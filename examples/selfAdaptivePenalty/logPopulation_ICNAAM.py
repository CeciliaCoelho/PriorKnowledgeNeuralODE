import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.optim as optim

from PriorKnowledgeNeuralODE import adaptiveFuncs
from PriorKnowledgeNeuralODE.adaptiveFuncs import *


parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=200)
parser.add_argument('--test_data_size', type=int, default=200)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--tf', type=int, default=300)
parser.add_argument('--tf_test', type=int, default=300)
parser.add_argument('--savePlot', type=str)
parser.add_argument('--saveModel', type=str)
parser.add_argument('--adaptiveFunc', type=str, choices=['self', 'lemonge', 'dynamic0', 'dynamic1'], default='self')

args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([2.518629]).to(device)

t = torch.linspace(0., args.tf, args.data_size).to(device)
t_test = torch.linspace(0., args.tf_test, args.test_data_size).to(device)



class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mul(torch.mul(0.026, y), torch.sub(1, torch.div(y,12)))


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    test_y = odeint(Lambda(), true_y0, t_test, method='dopri5')


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)





class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50,50),
            nn.ELU(),
            nn.Linear(50, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)



if __name__ == '__main__':
    ii = 0

    func = ODEFunc().to(device)
    optimizer = optim.Adam(func.parameters(), lr=1e-5)

    theta_best = list(func.parameters())
    theta = list(func.parameters())
    MSE_thetaf = torch.tensor(0.) 
    phi_best = math.inf

    for itr in range(1, args.niters + 1):
        pred_y = odeint(func, true_y0, t, method='rk4').to(device)

        MSE_theta = torch.square(torch.subtract(pred_y, true_y))
        v_j = [torch.maximum(torch.subtract(pred_y, 12), torch.Tensor([0]).to(device))]

        if args.adaptiveFunc == "self": phi_theta, MSE_theta = selfAdaptive(MSE_theta, v_j, itr, MSE_thetaf)
        if args.adaptiveFunc == "lemonge": phi_theta, MSE_theta = lemongeAdaptive(MSE_theta, v_j, itr, MSE_thetaf)
        if args.adaptiveFunc == "dynamic0": phi_theta, MSE_theta = dynamicAdaptive(MSE_theta, v_j, itr, MSE_thetaf, flag=0)
        if args.adaptiveFunc == "dynamic1": phi_theta, MSE_theta = dynamicAdaptive(MSE_theta, v_j, itr, MSE_thetaf, flag=1)
        

        if phi_theta < phi_best:
            theta_best = list(func.parameters())
            phi_best = phi_theta
        else:
            for prev, upd in zip(theta_best, theta):
                upd.data = prev.data

        if itr != args.niters:
            optimizer.zero_grad()
            phi_theta.backward(retain_graph=True)
            optimizer.step()

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                mse = nn.MSELoss()(pred_y, true_y)
                violation = torch.mean((torch.maximum(torch.subtract(pred_y, 12), torch.Tensor([0]).to(device))))
                print('Iter {:04d} | PHI_THETA {:.9f} | PHI_BEST {:.9f} | MSE {:.9f} | Penalty {:.9f}'.format(itr, phi_theta.item(), phi_best.item(), mse.item(), violation.item()))
                ii += 1

                end = time.time()
        
                
        if itr == args.niters:
            pred_y_test = odeint(func, true_y0, t_test)
            mse_t = nn.MSELoss()(pred_y_test, test_y)
            violation_t = torch.mean((torch.maximum(torch.subtract(pred_y_test, 12), torch.Tensor([0]).to(device))))
            print('MSE Test {:.9f} | Violation {:.9f}'.format(mse_t.item(), violation_t.item()))
            plt.plot(t_test.detach().cpu().numpy(), test_y.detach().cpu().numpy(), linestyle='dashed', label='real')
            plt.plot(t_test.detach().cpu().numpy(), pred_y_test.detach().cpu().numpy(), label='predicted')
            plt.xlabel("Time")
            plt.ylabel("Population")
            plt.legend()
            #plt.savefig(args.savePlot, format='eps')
            #torch.save(func, args.saveModel)
            plt.show()

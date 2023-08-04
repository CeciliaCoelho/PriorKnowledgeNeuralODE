import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.optim as optim

from adaptiveFuncs import * 




parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=100)
parser.add_argument('--test_data_size', type=int, default=100)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--tf', type=int, default=100)
parser.add_argument('--tf_test', type=int, default=100)
parser.add_argument('--savePlot', type=str)
parser.add_argument('--saveModel', type=str)
parser.add_argument('--adaptiveFunc', type=str, choices=['self', 'lemonge', 'dynamic0', 'dynamic1'], default='self')

args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

start_time = 0
end_time = args.tf
end_time_test = args.tf_test
time_steps = args.data_size
time_steps_test = args.test_data_size
dt = (end_time - start_time) / time_steps
dt_test = (end_time_test - start_time) / time_steps_test

# Define the initial masses of each element in the reaction
A = 1
B = 1
C = 0
D = 0

# Define the reaction rate constants
k1 = 0.1
k2 = 0.05

# Store the masses of each element in time
times = np.arange(start_time, end_time, dt)
times_test = np.arange(start_time, end_time_test, dt_test)
masses = np.zeros((time_steps, 4))
masses_test = np.zeros((time_steps_test, 4))
masses[0,:] = [A, B, C, D]
masses_test[0,:] = [A, B, C, D]

# Simulate the reaction over time
for i, t in enumerate(times[:-1]):
    a = masses[i, 0]
    b = masses[i, 1]
    c = masses[i, 2]
    d = masses[i, 3]
    masses[i+1,:] = [a + dt * (-k1 * a * b),
                     b + dt * (-k1 * a * b + k2 * c),
                     c + dt * (k1 * a * b - k2 * c),
                     d + dt * (k1 * a * b)]

for i, t in enumerate(times_test[:-1]):
    a = masses_test[i, 0]
    b = masses_test[i, 1]
    c = masses_test[i, 2]
    d = masses_test[i, 3]
    masses_test[i+1,:] = [a + dt_test * (-k1 * a * b),
                     b + dt_test * (-k1 * a * b + k2 * c),
                     c + dt_test * (k1 * a * b - k2 * c),
                     d + dt_test * (k1 * a * b)]



# Normalize the masses so that the total mass is 1 at each time step
masses = masses / np.sum(masses, axis=1, keepdims=True)
masses_test = masses_test / np.sum(masses_test, axis=1, keepdims=True)





class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
            nn.Linear(50,64),
            nn.ELU(inplace=True),
            nn.Linear(64,50),
            nn.Tanh(),
            nn.Linear(50, 4),
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

    true_y0 = torch.Tensor(masses[0,:]).to(device)
    true_y = torch.Tensor(masses).to(device)
    true_y_test = torch.Tensor(masses_test).to(device)
    t = torch.Tensor(times).to(device)
    t_test = torch.Tensor(times_test).to(device)
    tot_m = torch.sum(true_y0)

    theta_best = list(func.parameters())
    theta = list(func.parameters())
    MSE_thetaf = torch.tensor(0.) 
    phi_best = math.inf

    epsilon = 1e-4

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_y = odeint(func, true_y0, t, method='rk4').to(device)

#####################################################
        MSE_theta = torch.square(torch.subtract(pred_y, true_y))


        if args.adaptiveFunc == "self": 
            v_j = [torch.square(torch.subtract(torch.sum(pred_y, dim=1), tot_m))]
            phi_theta, MSE_theta = selfAdaptive(MSE_theta, v_j, itr, MSE_thetaf)
        if args.adaptiveFunc == "lemonge" or args.adaptiveFunc == 'dynamic0' or args.adaptiveFunc == 'dynamic1': 
            v_j = [torch.abs(torch.subtract(torch.sum(pred_y, dim=1), tot_m) - epsilon), torch.abs(-torch.subtract(torch.sum(pred_y, dim=1), tot_m) - epsilon)]
            if args.adaptiveFunc == 'lemonge': phi_theta, MSE_theta = lemongeAdaptive(MSE_theta, v_j, itr, MSE_thetaf)
            if args.adaptiveFunc == "dynamic0": phi_theta, MSE_theta = dynamicAdaptive(MSE_theta, v_j, itr, MSE_thetaf, flag=0)
            if args.adaptiveFunc == "dynamic1": phi_theta, MSE_theta = dynamicAdaptive(MSE_theta, v_j, itr, MSE_thetaf, flag=1)

        if phi_theta < phi_best:
            theta_best = list(func.parameters())
            phi_best = phi_theta
        else:
            curr_theta = list(func.parameters())
            for prev, upd in zip(theta_best, curr_theta):
                upd.data = prev.data

#####################################################
        if itr != args.niters:
            phi_theta.backward(retain_graph=True)
            optimizer.step()


        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t, method='rk4')
                mse = nn.MSELoss()(pred_y, true_y)
                violation = torch.mean(torch.square(torch.subtract(torch.sum(pred_y, dim=1), tot_m)))
                print('Iter {:04d} | PHI {:.9f} | MSE {:.9f} | Penalty {:.9f}'.format(itr, phi_theta.item(), mse.item(), violation.item()))

                
        if itr == args.niters:
            pred_y_test = odeint(func, true_y0, t_test)
            mse = nn.MSELoss()(pred_y_test, true_y_test)
            pen = torch.mean(torch.square(torch.subtract(torch.sum(pred_y_test, dim=1), tot_m)))
            print('MSE Test {:.9f} | Violation {:.9f}'.format(mse.item(), pen.item()))
            plt.plot(times_test, masses_test[:, 0], label="A", linestyle='dashed', color='green')
            plt.plot(times_test, masses_test[:, 1], label="B", linestyle='dashed', color='blue')
            plt.plot(times_test, masses_test[:, 2], label="C", linestyle='dashed', color='orange')
            plt.plot(times_test, masses_test[:, 3], label="D", linestyle='dashed', color='red')
            plt.plot(times_test, pred_y_test[:, 0].detach().cpu().numpy(), label="A", color='green')
            plt.plot(times_test, pred_y_test[:, 1].detach().cpu().numpy(), label="B", color='blue')
            plt.plot(times_test, pred_y_test[:, 2].detach().cpu().numpy(), label="C", color='orange')
            plt.plot(times_test, pred_y_test[:, 3].detach().cpu().numpy(), label="D", color='red')
            plt.xlabel("Time (s)")
            plt.ylabel("Mass")
            plt.legend()
            #plt.savefig(args.savePlot, format='eps')
            #torch.save(func, args.saveModel)
            plt.show()
                                
            total_mass = torch.sum(pred_y_test, dim=1)
            conservation = [abs(total_mass[i]-total_mass[0]) for i in range(1,len(total_mass))]
            print("Conservation of mass violated ", sum(list(map(lambda x: x>1e-6, conservation))).item(), " times!")
            print("Total difference is ", sum(conservation).item())




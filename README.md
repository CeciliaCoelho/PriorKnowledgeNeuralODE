# PriorKnowledgeNeuralODE

#### C. Coelho, M. F. P. Costa, and L.L. Ferrás, "A Study on Adaptive Penalty Functions in Neural ODEs for Real Systems Modeling" in Proceedings of the International Conference of Numerical Analysis and Applied Mathematics (ICNAAM-2023) (AIP Conference Proceedings, accepted)

[![License](https://img.shields.io/github/license/lululxvi/deepxde)](https://github.com/lululxvi/deepxde/blob/master/LICENSE) <img src="https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.svg" width="128"/>


This library provides a torch implementations of 3 adaptive penalty functions that can be used for training NN architectures. To learn more check the [paper](A_Study_on_Adaptive_Penalty_Functions_in_Neural_ODEs.pdf).

### Installation
```
pip install PriorKnowledgeNeuralODE
```

### Dependencies
<ol>
  <li>torchdiffeqq</li>
  <li>torch</li>
  <li>pandas</li>
  <li>numpy</li>
  <li>math</li>
  <li>matplotlib</li>
</ol>

### Examples
There are 2 case study examples that use a Neural ODE to model the [World Population Growth](https://www.kaggle.com/datasets/cici118/world-population-growth) and the evolution of a [Chemical Reaction](https://www.kaggle.com/datasets/cici118/synthetic-chemical-reaction) available [here](https://github.com/CeciliaCoelho/PriorKnowledgeNeuralODE/tree/master/examples/selfAdaptivePenalty)

##### Options
<ol>
  <li>--method :numerical method to solve the ODE, choices=['dopri5', 'adams']</li>
  <li>--data_size :number of training time steps/li>
  <li>--test_data_size :number of testing time steps</li>
  <li>--niters :number of iterations to train the NN</li>
  <li>--test_freq :frequency to compute and print the test metrics</li>
  <li>--gpu :turn on/off gpu</li>
  <li>--adjoint :use the adjoint method to compute the gradients</li>
  <li>--tf :value of the last time step for training</li>
  <li>tf_test :value of the last time step for testing</li>
  <li>--savePlot :path to store the plot of the real vs predicted curves</li>
  <li>--saveModel :path to store the weights of the trained model</li>
  <li>--adaptiveFunc :choice of the adaptive penalty function choices=['self', 'lemonge', 'dynamic0', 'dynamic1']</li>
 
</ol>

If you found this resource useful in your research, please consider citing.

```
@inproceedings{,
  title={A Study on Adaptive Penalty Functions in Neural ODEs for Real Systems Modeling},
  author={Coelho, C. and Costa, M. F. P. and Ferrás, L. L.},
  journal={International Conference of Numerical Analysis and Applied Mathematics (accepted)},
  year={2023}
}

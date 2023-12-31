PriorKnowledgeNeuralODE
=======================

C. Coelho, M. F. P. Costa, and L.L. Ferrás, “A Study on Adaptive Penalty Functions in Neural ODEs for Real Systems Modeling” in Proceedings of the International Conference of Numerical Analysis and Applied Mathematics (ICNAAM-2023) (AIP Conference Proceedings, accepted)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|License: MIT|

This library provides a torch implementations of 3 adaptive penalty
functions that can be used for training NN architectures. To learn more
check the
`paper <A_Study_on_Adaptive_Penalty_Functions_in_Neural_ODEs.pdf>`__.

Installation
~~~~~~~~~~~~

::

   pip install PriorKnowledgeNeuralODE

Dependencies
~~~~~~~~~~~~

.. raw:: html

   <ol>

.. raw:: html

   <li>

torchdiffeqq

.. raw:: html

   </li>

.. raw:: html

   <li>

torch

.. raw:: html

   </li>

.. raw:: html

   <li>

pandas

.. raw:: html

   </li>

.. raw:: html

   <li>

numpy

.. raw:: html

   </li>

.. raw:: html

   <li>

math

.. raw:: html

   </li>

.. raw:: html

   <li>

matplotlib

.. raw:: html

   </li>

.. raw:: html

   </ol>

Examples
~~~~~~~~

There are 2 case study examples that use a Neural ODE to model the
`World Population
Growth <https://www.kaggle.com/datasets/cici118/world-population-growth>`__
and the evolution of a `Chemical
Reaction <https://www.kaggle.com/datasets/cici118/synthetic-chemical-reaction>`__
available
`here <https://github.com/CeciliaCoelho/PriorKnowledgeNeuralODE/tree/master/examples/selfAdaptivePenalty>`__

Options
'''''''

.. raw:: html

   <ol>

.. raw:: html

   <li>

–method :numerical method to solve the ODE, choices=[‘dopri5’, ‘adams’]

.. raw:: html

   </li>

.. raw:: html

   <li>

–data_size :number of training time steps/li>

.. raw:: html

   <li>

–test_data_size :number of testing time steps

.. raw:: html

   </li>

.. raw:: html

   <li>

–niters :number of iterations to train the NN

.. raw:: html

   </li>

.. raw:: html

   <li>

–test_freq :frequency to compute and print the test metrics

.. raw:: html

   </li>

.. raw:: html

   <li>

–gpu :turn on/off gpu

.. raw:: html

   </li>

.. raw:: html

   <li>

–adjoint :use the adjoint method to compute the gradients

.. raw:: html

   </li>

.. raw:: html

   <li>

–tf :value of the last time step for training

.. raw:: html

   </li>

.. raw:: html

   <li>

tf_test :value of the last time step for testing

.. raw:: html

   </li>

.. raw:: html

   <li>

–savePlot :path to store the plot of the real vs predicted curves

.. raw:: html

   </li>

.. raw:: html

   <li>

–saveModel :path to store the weights of the trained model

.. raw:: html

   </li>

.. raw:: html

   <li>

–adaptiveFunc :choice of the adaptive penalty function choices=[‘self’,
‘lemonge’, ‘dynamic0’, ‘dynamic1’]

.. raw:: html

   </li>

.. raw:: html

   </ol>

If you found this resource useful in your research, please consider
citing.

\``\` @inproceedings{, title={A Study on Adaptive Penalty Functions in
Neural ODEs for Real Systems Modeling}, author={Coelho, C. and Costa, M.
F. P. and Ferrás, L. L.}, journal={International Conference of Numerical
Analysis and Applied Mathematics (accepted)}, year={2023} }
\``\`

.. |License: MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

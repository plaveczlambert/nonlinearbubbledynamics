# Nonlinear Bubble Dynamics with Deep Euler Method
Solving a nonlinear acoustic cavitation bubble model with the Dormand-Prince and Deep Euler Methods (DEM). 
The bubble model is based on the [Nonlinear Bubble Dynamics](https://www.semanticscholar.org/paper/Nonlinear-bubble-dynamics-Prosperetti-Crum/6cf87749e74b007c7263721c834def11be3dd59b) paper.

The repository has two main components:
* `Dopri` folder: contains Matlab and C++ implementations of the bubble model using the Dormand-Prince numerical solver. 
* `DEM` folder: implementation of the bubble model using the [Deep Euler Method](https://arxiv.org/abs/2003.09573) aka [HyperEuler](https://arxiv.org/abs/2007.09601). The solver incorporates a neural network, so the whole process of generating training data, training the network and testing the Deep Euler Method should be followed to get a working Deep Euler integration.
## Dopri
The `Dopri` folder has two subfolders, one for the C++, another for the MATLAB implementation. The C++ implementation is a single code file. It is based on the [boost](https://www.boost.org/) C++ library and the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) matrix algebra library.
The MATLAB implementation contains a script file in which the actual model and solution is coded, and a live script file to be used for visualization and quick testing.

## DEM
This folder contains the necessary infrastructure to train a Deep Euler Method for solving the bubble model.
The process is the following:
1. Generate datapoints using the C++ file in `data_gen`.
2. Create a usable training dataset using the `transform_data.ipynb` jupyter notebook.
3. Use the file `dem_train.py` to train a neural network. It accepts command line options. A useful command line instruction is the following:
```
python dem_train.py --name MyModel --epoch 5000 --early_stop --batch 100 --save_plots --print_epoch 50 --print_losses 50 --data data/data.hdf5
```
4. The Deep Euler Method is implemented in C++ with the help of the [Pytorch C++](https://pytorch.org/) library. The code is available in `run/normal`. You should use CMake to create the project. The "traced" version of the trained neural network can be loaded into C++ and used for integration.
5. The results of the Deep Euler intagration can be viewed with the `figures.ipynb` jupyter notebook

The `DEM` folder has similar structure to my other repository [Deep Euler Tests](https://github.com/plaveczlambert/deep_euler_tests). It also contains more information on the process of training a neural network in Python and using it in C++. The accepted command line arguments of `dem_train.py` are listed there, too.

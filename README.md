# Simple Gradient Boosting

This is a Python project that implements a basic version of gradient boosting trees for regression problems. 

The project uses numpy and simple DecisionTreeRegressors to implement gradient descent with momentum for boosting the simple trees and compute the mean squared error (MSE) loss function. 

The project also includes a Jupyter notebook that explores the gradient boosting algorithm on a synthetic dataset.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)

## Installation

To run this project, you need to have Python 3, sklearn and numpy installed. You can install numpy and sklearn using pip:

```
pip install numpy
pip install scikit-learn
```
You also need to clone this repository to your local machine:
```
git clone https://github.com/gurug-dev/simple-gradient-boosting.git
```
## Usage
To use the gradient boosting module, you need to import it in your Python script:
```
from gradient_boosting_mse import gradient_boosting_mse
```
Then, you can create the trees and fit it to your training data:
```
y_mean, trees = gradient_boosting_mse(X_train, y_train, 5, max_depth=2, nu=0.1)
```
You can also use the predict method to make predictions on new data:
```
y_hat = gradient_boosting_predict(X_val, trees, y_mean, nu=0.1)
```
To see an example of how to use the gradient boosting module on a synthetic dataset, you can open the "exploring GB.ipynb" notebook in Jupyter:
```
jupyter notebook `exploring\ GB.ipynb`
```
## Testing
To run the unit tests for the gradient boosting module, you need to have pytest installed. You can install it using pip:
```
pip install pytest
```
Then, you can run the test_gradient_boosting_mse.py script using pytest:
```
pytest test_gradient_boosting_mse.py
```

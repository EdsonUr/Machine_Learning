# Linear Regression from Scratch

This implementation builds a linear regression model from scratch using Python and NumPy.

Linear regression is a supervised learning algorithm used for **regression problems**, where the target value is continuous.

For example:

```math
y \in \mathbb{R}
```

## Table of Contents

- [Model](#model)
- [Cost Function](#cost-function)
- [Gradient Descent](#gradient-descent)
- [Cost History](#cost-history)
- [Prediction Line](#prediction-line)
- [Evaluation](#evaluation)
- [Project Files](#project-files)

## Model

The model predicts a value by computing a linear combination of the input features:

```math
f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b
```

For multiple training examples, this is implemented in vectorized form as:

```math
f_{\mathbf{w},b}(X) = X\mathbf{w} + b
```

Here, $X\mathbf{w}$ represents a matrix-vector multiplication. Each row of $X$ is one training example, and each prediction is computed as the dot product between that row and the weight vector $\mathbf{w}$:

```math
\mathbf{w} \cdot \mathbf{x}^{(i)}
=
w_1x_1^{(i)}
+
w_2x_2^{(i)}
+
\cdots
+
w_nx_n^{(i)}
```

where:

- $X$ is the input matrix with shape $(m,n)$
- $m$ is the number of training examples
- $n$ is the number of features
- $\mathbf{w}$ is the weight vector
- $b$ is the bias term
- $f_{\mathbf{w},b}(X)$ is the model prediction

When the dataset has only one feature, the model can be represented as a straight line:

```math
f_{w,b}(x) = wx + b
```

## Cost Function

For linear regression, the cost function is the **mean squared error cost**, written with a factor of $\frac{1}{2}$ to make the derivative simpler:

```math
J(\mathbf{w}, b)
=
\frac{1}{2m}
\sum_{i=0}^{m-1}
\left(
f_{\mathbf{w},b}(\mathbf{x}^{(i)})
-
y^{(i)}
\right)^2
```

This cost measures how far the model predictions are from the real target values.

A smaller cost means the model predictions are closer to the actual values.

## Gradient Descent

To minimize the cost function, the model uses gradient descent.

The gradient with respect to each weight is:

```math
\frac{\partial J}{\partial w_j}
=
\frac{1}{m}
\sum_{i=0}^{m-1}
\left(
f_{\mathbf{w},b}(\mathbf{x}^{(i)})
-
y^{(i)}
\right)
x_j^{(i)}
```

The gradient with respect to the bias is:

```math
\frac{\partial J}{\partial b}
=
\frac{1}{m}
\sum_{i=0}^{m-1}
\left(
f_{\mathbf{w},b}(\mathbf{x}^{(i)})
-
y^{(i)}
\right)
```

In vectorized form, the gradients are:

```math
d\mathbf{w}
=
\frac{1}{m}
X^T
\left(
f_{\mathbf{w},b}(X)-y
\right)
```

```math
db
=
\frac{1}{m}
\sum
\left(
f_{\mathbf{w},b}(X)-y
\right)
```

After calculating the gradients, the parameters are updated using:

```math
\mathbf{w}
=
\mathbf{w}
-
\alpha d\mathbf{w}
```

```math
b
=
b
-
\alpha db
```

where $\alpha$ is the learning rate.

This process is repeated for the number of iterations defined by `n_iters`.

## Cost History

The cost calculated after each parameter update is stored in `J_history`.

This allows the cost to be plotted over the training iterations and helps verify whether gradient descent is learning correctly.

If the model is training correctly, the cost should generally decrease over time.

## Prediction Line

When the dataset has only one feature, the learned model can be visualized as a straight line.

The prediction line represents:

```math
\hat{y} = wx + b
```

In the training notebook:

- The red points represent the training data
- The blue points represent the testing data
- The black line represents the predictions made by the model

If the line follows the general direction of the data points, the model learned a good linear relationship between the input feature and the target value.

The model also supports datasets with multiple features, although their predictions cannot be displayed as a simple two-dimensional line.

## Evaluation

After training, the model is evaluated on the test set using the **mean squared error**:

```math
MSE
=
\frac{1}{m}
\sum_{i=0}^{m-1}
\left(
y^{(i)}
-
\hat{y}^{(i)}
\right)^2
```

The MSE measures the average squared difference between the real values and the predicted values.

A lower MSE means the model is making better predictions.

## Project Files

This folder contains the following files:

```text
01 Linear_Regression/
├── LinearRegression.py
├── train.ipynb
└── README.md
```

### `LinearRegression.py`

Contains the linear regression model implemented from scratch using NumPy.

The model includes:

- Linear prediction
- Squared error cost calculation
- Gradient calculation
- Gradient descent
- Parameter updates
- Cost history
- Prediction method

### `train.ipynb`

Contains the practical experiment used to train and evaluate the model.

The notebook includes:

- Dataset generation
- Training and testing split
- Dataset visualization
- Model training
- Predictions
- Mean squared error calculation
- Learned weight and bias
- Regression line visualization
- Cost history visualization

### `README.md`

Contains the theoretical explanation of linear regression and documents how the model works.
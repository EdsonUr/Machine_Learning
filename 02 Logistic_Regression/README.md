# Logistic Regression from Scratch

This implementation builds a logistic regression model from scratch using Python and NumPy.

Logistic regression is a supervised learning algorithm used for **binary classification**, where the target value is usually:

```math
y \in \{0,1\}
```

## Table of Contents

- [Model](#model)
- [Sigmoid Function](#sigmoid-function)
- [Class Prediction](#class-prediction)
- [Cost Function](#cost-function)
- [Gradient Descent](#gradient-descent)
- [Cost History](#cost-history)
- [Decision Boundary](#decision-boundary)
- [Evaluation](#evaluation)
- [Project Files](#project-files)

## Model

The model first computes a linear combination of the input features:

```math
z = \mathbf{w} \cdot \mathbf{x} + b
```

For multiple training examples, this is implemented in vectorized form as:

```math
z = X\mathbf{w} + b
```

where:

- $X$ is the input matrix with shape $(m,n)$
- $m$ is the number of training examples
- $n$ is the number of features
- $\mathbf{w}$ is the weight vector
- $b$ is the bias term
- $z$ is the linear combination of the features and parameters

Unlike linear regression, logistic regression does not use $z$ directly as the final prediction. The sigmoid function transforms it into a probability between 0 and 1.

## Sigmoid Function

After computing $z$, logistic regression applies the **sigmoid function**:

```math
g(z) = \frac{1}{1+e^{-z}}
```

The model prediction is therefore:

```math
f_{\mathbf{w},b}(\mathbf{x})
=
g\left(\mathbf{w}\cdot\mathbf{x}+b\right)
```

For multiple training examples, the predictions are calculated in vectorized form as:

```math
f_{\mathbf{w},b}(X)
=
g\left(X\mathbf{w}+b\right)
```

The output of the sigmoid function is a value between 0 and 1.

It represents the model's estimated probability that an input belongs to class 1:

```math
f_{\mathbf{w},b}(\mathbf{x})
=
P\left(y=1\mid\mathbf{x}\right)
```

A value close to 1 indicates a higher probability of belonging to class 1, while a value close to 0 indicates a higher probability of belonging to class 0.

## Class Prediction

The probability returned by the sigmoid function is converted into a class using a threshold of 0.5.

```math
\hat{y}
=
\begin{cases}
1, & \text{if } f_{\mathbf{w},b}(\mathbf{x}) \geq 0.5 \\
0, & \text{if } f_{\mathbf{w},b}(\mathbf{x}) < 0.5
\end{cases}
```

Using this threshold:

- A probability greater than or equal to 0.5 produces class 1
- A probability below 0.5 produces class 0

The `predict_probability` method returns the probabilities, while the `predict` method converts those probabilities into class labels.

## Cost Function

Logistic regression does not use the squared error cost as its main cost function.

Instead, it uses **binary cross-entropy**, also known as **log loss**:

```math
J(\mathbf{w},b)
=
-\frac{1}{m}
\sum_{i=0}^{m-1}
\left[
y^{(i)}
\log\left(
f_{\mathbf{w},b}(\mathbf{x}^{(i)})
\right)
+
\left(1-y^{(i)}\right)
\log\left(
1-f_{\mathbf{w},b}(\mathbf{x}^{(i)})
\right)
\right]
```

This cost becomes smaller when the model assigns a high probability to the correct class.

For an example belonging to class 1, the cost decreases as the predicted probability approaches 1.

For an example belonging to class 0, the cost decreases as the predicted probability approaches 0.

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

## Decision Boundary

The **decision boundary** separates the region where the model predicts class 0 from the region where it predicts class 1.

The predicted class changes when the model probability is exactly 0.5:

```math
f_{\mathbf{w},b}(\mathbf{x}) = 0.5
```

The sigmoid function returns 0.5 when its input is 0. As a result, the decision boundary occurs when:

```math
z
=
\mathbf{w}\cdot\mathbf{x}+b
=
0
```

For a dataset with two features, the decision boundary is represented by:

```math
w_1x_1+w_2x_2+b=0
```

Solving for $x_2$ gives:

```math
x_2
=
-\frac{w_1x_1+b}{w_2}
```

This produces a straight line that separates the two predicted classes.

One side of the line is classified as class 0, while the other side is classified as class 1.

## Evaluation

After training, the model is evaluated on the test set using **accuracy**:

```math
\text{Accuracy}
=
\frac{\text{Number of correct predictions}}
{\text{Total number of predictions}}
```

Accuracy measures the proportion of test examples classified correctly by the model.

A higher accuracy means that the model correctly classified a greater proportion of the unseen examples.

The real test labels and predicted labels are also displayed side by side in the training notebook to make the results easier to compare.

## Project Files

This folder contains the following files:

```text
02_Logistic_Regression/
├── LogisticRegression.py
├── train.ipynb
└── README.md
```

### `LogisticRegression.py`

Contains the logistic regression model implemented from scratch using NumPy.

The model includes:

- Linear combination of the input features
- Sigmoid function
- Binary cross-entropy cost
- Gradient calculation
- Gradient descent
- Parameter updates
- Cost history
- Probability prediction
- Binary class prediction

### `train.ipynb`

Contains the practical experiment used to train and evaluate the model.

The notebook includes:

- Binary classification dataset generation
- Training and testing split
- Dataset visualization
- Model training
- Probability and class predictions
- Accuracy calculation
- Learned weights and bias
- Cost history visualization
- Comparison between real and predicted classes
- Decision region visualization
- Decision boundary visualization

### `README.md`

Contains the theoretical explanation of logistic regression and documents how the model works.
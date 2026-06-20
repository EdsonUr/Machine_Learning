# Decision Tree from Scratch

This implementation builds a decision tree classifier from scratch using Python and NumPy.

A decision tree is a supervised learning algorithm that classifies examples by repeatedly splitting the data according to feature values.

The model supports both **binary classification** and **multiclass classification**.

For binary classification:

```math
y \in \{0,1\}
```

For multiclass classification:

```math
y \in \{0,1,2,\ldots,K-1\}
```

## Table of Contents

- [Tree Structure](#tree-structure)
- [Entropy](#entropy)
- [Information Gain](#information-gain)
- [Finding the Best Split](#finding-the-best-split)
- [Feature Types](#feature-types)
- [Tree Construction](#tree-construction)
- [Stopping Criteria](#stopping-criteria)
- [Leaf Nodes](#leaf-nodes)
- [Prediction](#prediction)
- [Evaluation](#evaluation)
- [Project Files](#project-files)

## Tree Structure

A decision tree is composed of **internal decision nodes** and **leaf nodes**.

Each internal node stores:

- The index of the selected feature
- The threshold used for the split
- The information gain produced by the split
- A reference to the left child
- A reference to the right child

Each leaf node stores the class predicted by the tree.

For an internal node, the decision rule is:

```math
x_j \leq t
```

where:

- $x_j$ is the value of feature $j$
- $t$ is the selected threshold

Examples that satisfy the condition move to the left child:

```math
x_j \leq t
```

Examples that do not satisfy the condition move to the right child:

```math
x_j > t
```

## Entropy

Entropy measures the impurity of the target values contained in a node.

For a node containing $K$ classes, entropy is calculated as:

```math
H(y)
=
-\sum_{k=0}^{K-1}
p_k\log_2(p_k)
```

where:

- $y$ contains the target values of the examples in the node
- $K$ is the number of possible classes
- $p_k$ is the proportion of examples belonging to class $k$

The probability of class $k$ is:

```math
p_k
=
\frac{\text{number of examples belonging to class }k}
{\text{total number of examples in the node}}
```

A pure node contains examples from only one class and has entropy:

```math
H(y)=0
```

A node containing a more balanced mixture of classes has a higher entropy.

Because the calculation considers the probability of every class, the implementation supports both binary and multiclass classification.

## Information Gain

Information gain measures how much a split reduces the impurity of the current node.

Let:

- $y$ be the target values of all examples in the parent node
- $y_{\text{left}}$ be the target values sent to the left child
- $y_{\text{right}}$ be the target values sent to the right child
- $m$ be the total number of examples in the parent node
- $m_{\text{left}}$ be the number of examples in the left child
- $m_{\text{right}}$ be the number of examples in the right child

The total number of examples satisfies:

```math
m
=
m_{\text{left}}
+
m_{\text{right}}
```

The weight of each child is the proportion of parent examples sent to that child:

```math
w_{\text{left}}
=
\frac{m_{\text{left}}}{m}
```

```math
w_{\text{right}}
=
\frac{m_{\text{right}}}{m}
```

The weighted entropy of the child nodes is:

```math
H_{\text{children}}
=
w_{\text{left}}H(y_{\text{left}})
+
w_{\text{right}}H(y_{\text{right}})
```

where:

- $H(y)$ is the entropy of the parent node before the split
- $H(y_{\text{left}})$ is the entropy of the left child
- $H(y_{\text{right}})$ is the entropy of the right child
- $w_{\text{left}}$ is the proportion of examples sent to the left child
- $w_{\text{right}}$ is the proportion of examples sent to the right child

Information gain is the difference between the parent entropy and the weighted entropy of the child nodes:

```math
IG
=
H(y)
-
H_{\text{children}}
```

Substituting the weighted child entropy:

```math
IG
=
H(y)
-
\left[
w_{\text{left}}H(y_{\text{left}})
+
w_{\text{right}}H(y_{\text{right}})
\right]
```

A larger information gain means that the split produces purer child nodes.

The tree calculates the information gain for every candidate split and selects the feature and threshold with the highest value.

## Finding the Best Split

At each node, the model examines every feature.

For each feature, its unique values are sorted:

```math
v_1 < v_2 < \cdots < v_r
```

Candidate thresholds are generated using the midpoint between consecutive unique values:

```math
t_j
=
\frac{v_j+v_{j+1}}{2}
```

For example, if a feature contains the unique values:

```text
2, 4, 7
```

the candidate thresholds are:

```text
3.0, 5.5
```

For every candidate threshold, the examples are divided into two groups.

The left group contains the examples that satisfy:

```math
x_j^{(i)} \leq t
```

The right group contains the examples that satisfy:

```math
x_j^{(i)} > t
```

The information gain is calculated for each candidate split.

The feature and threshold with the highest information gain are stored in the decision node.

## Feature Types

The threshold strategy works with several kinds of numerical features.

### Continuous Features

For continuous values such as:

```text
3.2, 4.7, 5.1
```

the candidate thresholds are created between consecutive values:

```text
3.95, 4.9
```

### Integer Features

Integer features are handled in the same way.

For example:

```text
1, 2, 4
```

produces:

```text
1.5, 3.0
```

### Binary Features

For a binary feature containing `0` and `1`, the candidate threshold is:

```math
t
=
\frac{0+1}{2}
=
0.5
```

The split becomes:

```math
x_j \leq 0.5
```

This separates values `0` and `1`.

### One-Hot Encoded Features

One-hot encoded categorical features also contain `0` and `1`, so they are separated using a threshold of `0.5`.

Raw text categories must be converted into numerical features before being passed to the model.

Unordered categories with more than two values should generally be one-hot encoded instead of being represented using arbitrary integer values.

## Tree Construction

The tree is constructed recursively.

At each node, the algorithm:

1. Checks the stopping criteria
2. Finds the best feature and threshold
3. Splits the current examples into left and right groups
4. Builds the left subtree
5. Builds the right subtree
6. Stores the split in a new decision node

The same process continues for each child until a stopping condition is reached.

## Stopping Criteria

Tree construction stops when at least one of the following conditions is satisfied.

### Maximum Depth

The tree stops growing when the current depth reaches `max_depth`.

This limits the complexity of the tree and can help reduce overfitting.

### Minimum Number of Samples

A node is not divided when it contains fewer examples than `min_samples_split`.

### Pure Node

A node becomes a leaf when all its examples belong to the same class.

In this case:

```math
H(y)=0
```

### Minimum Information Gain

A split is rejected when its information gain is not greater than `min_information_gain`.

This prevents the tree from creating branches that provide little or no improvement in class separation.

### No Valid Split

If every feature is constant at the current node, no candidate threshold can be generated.

The current node then becomes a leaf.

## Leaf Nodes

When the tree stops growing, the current node becomes a leaf node.

The leaf predicts the most frequent class among the examples that reached it.

Let $N_k$ represent the number of examples belonging to class $k$:

```math
N_k
=
\sum_{i=1}^{m_{\text{leaf}}}
\mathbf{1}
\left(
y^{(i)}=k
\right)
```

where:

- $m_{\text{leaf}}$ is the number of examples in the leaf
- $\mathbf{1}(y^{(i)}=k)$ equals 1 when example $i$ belongs to class $k$
- $\mathbf{1}(y^{(i)}=k)$ equals 0 otherwise

The predicted class is:

```math
\hat{y}_{\text{leaf}}
=
\operatorname*{arg\,max}_{k}
N_k
```

This means that the leaf returns the class with the largest number of examples.

If the node is pure, all examples belong to the same class, so that class is returned directly.

## Prediction

To predict the class of a new example, the model begins at the root node.

At each internal node:

- It moves left when the feature value is less than or equal to the threshold
- It moves right when the feature value is greater than the threshold

The process continues until a leaf node is reached.

The class stored in that leaf is returned as the final prediction.

## Evaluation

The model is evaluated on the test set using accuracy:

```math
\text{Accuracy}
=
\frac{\text{Number of correct predictions}}
{\text{Total number of predictions}}
```

Accuracy measures the proportion of unseen examples classified correctly by the model.

The training notebook uses the Breast Cancer dataset, which is a binary classification dataset.

Although this experiment uses two classes, the entropy calculation and leaf prediction also support multiclass targets represented by nonnegative integer labels.

## Project Files

This folder contains the following files:

```text
03_Decision_Tree/
├── DecisionTree.py
├── train.ipynb
└── README.md
```

### `DecisionTree.py`

Contains the decision tree classifier implemented from scratch using NumPy.

The model includes:

- Decision-node and leaf-node representation
- Binary and multiclass entropy calculation
- Information gain calculation
- Midpoint threshold generation
- Best-split selection
- Recursive tree construction
- Stopping criteria
- Majority-class leaf prediction
- Tree traversal
- Class prediction

### `train.ipynb`

Contains the practical experiment used to train, evaluate, and visualize the model.

The notebook includes:

- Breast Cancer dataset loading
- Training and testing split
- Model training
- Test-set predictions
- Accuracy calculation
- Decision tree visualization

### `README.md`

Contains the theoretical explanation of the decision tree algorithm and documents how the implementation works.
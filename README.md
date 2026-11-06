# 🧠 ML Library from Scratch

A modular machine learning library built **from scratch with NumPy**, implementing Linear, Ridge, Lasso, and Polynomial Regression with decoupled **Optimizer** and **Regularizer** strategies, plus a fully hand-written **Neural Network (MLP)** with forward and backward propagation, for both regression and binary classification. Built by **[Mahamed Emad](https://www.linkedin.com/in/mahamed-emad/)** as a hands-on way to understand the math and the software design behind machine learning, without relying on scikit-learn.

---

## 📌 Features

### Classical models
- **From-Scratch Implementations:** Linear Regression and Polynomial Regression trained with vectorized mini-batch gradient descent, using only NumPy.
- **Pluggable Optimizers:** Optimizers (e.g. `Adam`) live in their own module and are passed into `fit`, so the training loop never changes when you swap them.
- **Pluggable Regularizers:** `Lasso` (L1) and `Ridge` (L2) penalties are separate strategy classes whose gradients are added to the loss gradient. Combine any model with any regularizer.
- **Strategy Pattern:** Optimizers and regularizers share abstract base classes (Python `abc`), making new ones easy to add.
- **Polynomial Feature Expansion:** Generates all polynomial and interaction terms up to a chosen degree using `combinations_with_replacement`.
- **K-Fold Cross-Validation:** A `Kfold` wrapper in `model_selection` that works with any base model.
- **scikit-learn-inspired API:** Familiar `fit` / `predict` / `score` methods, where `score` returns R².

### Neural network
- **Multi-Layer Perceptron from Scratch:** Any number of hidden layers, with hand-derived forward and backward propagation (no autograd, no frameworks).
- **Layer Objects:** `Input_Layer`, `Hiden_Layer`, and `Out_Layer` are linked together by a `Propagation` class; each layer implements its own `forward` and `backward` through an abstract `Layer` base class.
- **Pluggable Activations:** `Activation` (identity), `Relu`, `Sigmoid`, and `Softmax` share an abstract base class with `calc` (forward) and `derv` (derivative). The hidden activation is chosen with `hide_func` and the output activation with `acts_out`.
- **Bias as an Extra Node:** Every layer has a constant bias node that receives no input and only sends its own weights to the next layer, so the bias is trained exactly like any other weight.
- **Pluggable Losses:** `MSE` (regression) and `Bin_Cls` (binary cross-entropy, binary classification) share an abstract base class and turn the prediction and the target into the output-layer error signal. Choose one with `loss_fn`.
- **Built-in Scaling:** Inputs are standardized inside `fit`, and `predict` applies the same statistics. For regression the target is standardized too and predictions are converted back to the original scale; for classification the 0/1 labels are kept as they are.

---

## 📁 Repository Structure

```text
Machine-Learning-Library-from-Scratch/
├── linear_model/
│   ├── linear.py                  # Base class: shared init, predict, R² score
│   ├── linear_regression.py       # Mini-batch gradient descent training loop
│   └── polynomial_regression.py   # Polynomial feature expansion on top of linear regression
├── optimizers/
│   ├── _base.py                   # Abstract optimizer interface
│   ├── optimizer.py               # Default (plain gradient) optimizer
│   └── adam.py                    # Adam optimizer
├── regularizers/
│   ├── _base.py                   # Abstract regularizer interface
│   ├── regularizer.py             # Default (no regularization)
│   ├── lasso.py                   # L1 regularization
│   └── ridge.py                   # L2 regularization
├── model_selection/
│   └── kfold.py                   # K-Fold cross-validation
├── neural_network/
│   ├── neural_network.py          # Neural_Network class: fit / predict, input & target scaling
│   ├── mlp.py                     # Propagation: builds the linked layers, runs forward / backward
│   ├── layers/
│   │   ├── layer.py               # Abstract layer interface
│   │   ├── input_layer.py         # Input layer (features + bias node)
│   │   ├── hiden_layer.py         # Hidden layer with configurable activation
│   │   └── output_layer.py        # Output layer: activation + loss error signal
│   ├── activations/
│   │   ├── _base.py               # Abstract activation interface (calc / derv)
│   │   ├── activation.py          # Identity
│   │   ├── relu.py                # ReLU
│   │   ├── sigmoid.py             # Sigmoid
│   │   └── softmax.py             # Softmax
│   └── losses/
│       ├── _base.py               # Abstract loss interface
│       ├── mse.py                 # Mean squared error (regression)
│       └── bin_cls.py             # Binary cross-entropy (binary classification)
├── main.py                        # Neural network experiments (regression and classification)
└── README.md
```

---

## ⚙️ How It Works

### Linear models

Each training step of `Linear_Regression.fit` follows the same recipe, and only the strategies change:

1. Shuffle the data every epoch and split it into mini-batches.
2. Compute the prediction error and the MSE gradient: `Xᵀ(ŷ − y) / m`.
3. Add the **regularizer gradient** (`regularizer.grad(weights)`).
4. Pass the total gradient through the **optimizer** (`optimizer.update(grad)`).
5. Update the weights: `w ← w − lr · update`.

Because the loop only talks to the abstract optimizer and regularizer interfaces, adding a new strategy (e.g. Momentum or ElasticNet) never requires touching the model code.

`Polynomial_Regression` inherits from `Linear_Regression` and only adds a feature-expansion step before `fit` and `predict`, so it automatically supports every optimizer and regularizer.

### Neural network

**Data layout.** `fit` and `predict` receive `x` as `(samples, features)` like the other models. Internally the data is transposed so that **rows are features and columns are samples**, and a row of ones is added on top as the bias node.

**Shapes.** For a layer with `m` nodes feeding a layer with `h` nodes, the weight matrix is `W: (m+1) × h` (the `+1` is the bias node, whose weights form the first row of `W`):

```text
Wᵀ · X = (h × (m+1)) · ((m+1) × n) = h × n
```

The result has only the `h` real nodes of the next layer, so **the bias node never receives an input**. A new bias row is added before the next multiplication.

**Forward pass** (per layer):
1. `Z = Wᵀ · A_prev` (the previous layer's output, including its bias row)
2. `A = f(Z)` (hidden activation)

**Backward pass:**
1. Output error `δ`, computed by the loss strategy (see below)
2. Gradient of the weights: `dW = A_prev · δᵀ`
3. Error sent to the previous layer: `δ_prev = (W · δ) ⊙ f'(z_prev)`, **with the bias row dropped**, because no error can flow into a node that has no input
4. Update: `W ← W − lr · dW / n`

**Loss strategies.** The loss object (`loss_fn`) produces the output error `δ` that starts the backward pass:

| Loss                       | Task                  | Output activation | `δ`                 |
| -------------------------- | --------------------- | ----------------- | ------------------- |
| `MSE`                      | regression            | identity (or any) | `(ŷ − y) ⊙ f'(z)`   |
| `Bin_Cls` (cross-entropy)  | binary classification | `Sigmoid`         | `ŷ − y`             |

Binary cross-entropy is `L = −[y·log ŷ + (1 − y)·log(1 − ŷ)]`, so `∂L/∂ŷ = (ŷ − y) / (ŷ(1 − ŷ))`. The sigmoid derivative is `ŷ(1 − ŷ)`, and the product simplifies to `ŷ − y`. Unlike MSE combined with a sigmoid, this gradient does not shrink when the sigmoid saturates, so confident wrong predictions are still corrected quickly. The shortcut is only valid with a `Sigmoid` output layer and labels in `{0, 1}`.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas
```

### Installation

```bash
git clone https://github.com/mahamed-emad/Machine-Learning-Library-from-Scratch.git
cd Machine-Learning-Library-from-Scratch
```

### Quick Example: Linear Models

```python
import numpy as np
import pandas as pd
from linear_model import Linear_Regression, Polynomial_Regression
from optimizers import Adam
from regularizers import Lasso, Ridge

# Load data: features first, target in the last column
data = pd.read_csv("data.csv").to_numpy(dtype=np.float32)
x, y = data[:, :-1], data[:, -1:]

# Standardize features (important for gradient descent)
x = (x - x.mean(axis=0)) / x.std(axis=0)

# Linear regression with Adam + L1 regularization
model = Linear_Regression(lr=0.001, batches=32, itr=1000)
model.fit(x, y, optimizer=Adam(), regularizer=Lasso())
print("R²:", model.score(x, y))

# Polynomial regression (degree 2) with Adam + L2 regularization
poly = Polynomial_Regression(lr=0.001, batches=32, itr=1000)
poly.fit(x, y, degree=2, optimizer=Adam(), regularizer=Ridge())
print("R²:", poly.score(x, y))
predictions = poly.predict(x)
```

### Quick Example: Neural Network

```python
import numpy as np
from neural_network import Neural_Network, Relu

x = np.linspace(-3, 3, 200)
y = x ** 2                                    # a curve that no linear model can fit

model = Neural_Network(dimens=[1, 16, 8, 1], niter=20000, lr=0.1, hide_func=Relu)
model.fit(x, y)

print(model.predict(np.array([-2.0, 0.5, 2.5])))   # ≈ [4.08, 0.21, 6.33]  (true: 4, 0.25, 6.25)
```

### Quick Example: Binary Classification

```python
import numpy as np
from neural_network import Neural_Network, Relu, Sigmoid, Bin_Cls

np.random.seed(42)

x = np.arange(-50, 50)
y = np.where(x ** 2 >= 900, 1.0, 0.0)          # 1 if |x| >= 30 (a two-sided boundary)

model = Neural_Network(dimens=[1, 8, 1], niter=3000, lr=0.05,
                       hide_func=Relu, acts_out=Sigmoid, loss_fn=Bin_Cls)
model.fit(x, y)

probs = model.predict(np.array([-45.0, 0.0, 70.0]))   # probabilities in (0, 1)
print(probs >= 0.5)                                   # [ True False  True]
```

For several features pass `x` as a `(samples, features)` array, for example `Neural_Network(dimens=[0, 16, 8, 1])`. The first value of `dimens` is only a placeholder: it is replaced by the real input size.

### Hyperparameters

**Linear models**

| Parameter | Default | Description                                        |
| --------- | ------- | -------------------------------------------------- |
| `lr`      | `0.001` | Learning rate                                      |
| `batches` | `None`  | Mini-batch size (`None` = full-batch gradient descent) |
| `itr`     | `1e4`   | Number of epochs                                   |
| `seed`    | `42`    | Random seed                                        |

**Neural network**

| Parameter   | Default | Description                                                                 |
| ----------- | ------- | --------------------------------------------------------------------------- |
| `dimens`    | `[]`    | Layer sizes `[input, hidden..., output]` (the first value is a placeholder) |
| `acts_out`  | `None`  | Output activation, e.g. `Sigmoid` (`None` = identity)                       |
| `niter`     | `1e4`   | Number of full-batch gradient descent iterations                            |
| `lr`        | `0.1`   | Learning rate                                                               |
| `hide_func` | `Relu`  | Hidden activation class (`Relu`, `Sigmoid`), or `None` for no activation    |
| `loss_fn`   | `MSE`   | Loss: `MSE` (regression) or `Bin_Cls` (binary cross-entropy, needs `acts_out=Sigmoid`) |

---

## 📊 Experiments

Run the experiments with:

```bash
python main.py
```

`main.py` trains the neural network on known functions and boundaries. For regression it compares the network with a reference solution (the closed-form least-squares line from `numpy.polyfit`). All experiments fix the random seed (42).

**Test 1: linear function `y = 5x + 10`.** A network without a hidden activation is a linear model, so it should find the exact line, including at points outside the training range (153 and −15):

| x   | True | Neural network | Least squares |
| --- | ---- | -------------- | ------------- |
| 60  | 310  | 310.00         | 310.00        |
| 153 | 775  | 775.00         | 775.00        |
| 3   | 25   | 25.00          | 25.00         |
| -15 | -65  | -65.00         | -65.00        |

**Test 2: non-linear functions** on unseen points in `[-2.9, 2.9]`. The ReLU network learns the curves, while the network without a hidden activation behaves exactly like the best straight line (same MSE):

| Function     | Model            | MSE    | R²     |
| ------------ | ---------------- | ------ | ------ |
| `y = x²`     | ReLU network     | 0.0042 | 0.999  |
| `y = x²`     | No activation    | 6.7539 | -0.002 |
| `y = x²`     | Straight line    | 6.7539 | -0.002 |
| `y = sin(x)` | ReLU network     | 0.0002 | 1.000  |
| `y = sin(x)` | No activation    | 0.1554 | 0.708  |
| `y = sin(x)` | Straight line    | 0.1554 | 0.708  |

The identical numbers for "No activation" and "Straight line" show that stacking linear layers without a non-linearity adds no expressive power, which is exactly why activation functions matter.

**Test 3: binary classification with a step boundary** (`y = 1 if x ≥ 50 else 0`). The network (`Sigmoid` output + binary cross-entropy) is trained on 10,000 points in `[-1000, 1000]` and tested on 11 unseen points, including the extreme values `±10¹⁰` to check that the sigmoid saturates correctly. Result: **10 of 11 correct (90.9%)**. The only miss is `x = 48`, two units from the boundary (predicted probability 0.56): after 2,000 iterations the network has learned a soft boundary around `x = 50`. Training longer (`lr=0.5`, `niter=20000`) sharpens it and classifies all 11 points correctly.

**Test 4: binary classification with a two-sided boundary** (`y = 1 if x² ≥ 900`, i.e. `x ≤ -30` or `x ≥ 30`). A single linear boundary cannot represent this, so it needs the hidden layer. Result on 9 unseen points: **9 of 9 correct (100%)**, including points close to the boundaries (`x = 28` gets probability 0.48).

> **Note on stability.** With the default settings of Test 4 (`lr=0.05`, `niter=3000`) the result depends on the random initialization: across 10 seeds the mean accuracy is 96.7%, and one seed gets stuck at 66.7%. With `lr=0.2` and `niter=10000` all 10 seeds reach 100%.

---

## 🔭 Roadmap

- Neural network: support multi-class classification (Softmax with categorical cross-entropy) and multiple outputs. Currently the network has a single output (regression or binary classification).
- Neural network: add a `calc` method to the losses to report the loss value (a `loss_history`), plus `predict_classes` and an accuracy score.
- Neural network: reuse the `Optimizer` / `Adam` strategies and add mini-batch training.
- Add gradient-check unit tests for the neural network, and validate the linear models against scikit-learn (`LinearRegression`, `Ridge`, `Lasso`).
- Add more optimizers (SGD with momentum, RMSProp) and ElasticNet regularization.
- Add early stopping and learning-rate scheduling.
- Add `get_params` / `set_params` and make `fit` return `self` for full scikit-learn compatibility.
- Package the library with `pyproject.toml` so it can be installed with `pip`.

---

## 🔗 Author

**Mahamed Emad**

- GitHub: [Mahamed Emad](https://github.com/Mahamed-Emad)
- LinkedIn: [Mahamed Emad](https://www.linkedin.com/in/mahamed-emad/)

---

## 📜 License

Distributed under the MIT License.

# 🧠 ML Library from Scratch

A modular machine learning library built **from scratch with NumPy**, implementing Linear, Ridge, Lasso, and Polynomial Regression with decoupled **Optimizer** and **Regularizer** strategies, plus a fully hand-written **Neural Network (MLP)** with forward and backward propagation. Built by **[Mahamed Emad](https://www.linkedin.com/in/mahamed-emad/)** as a hands-on way to understand the math and the software design behind machine learning, without relying on scikit-learn.

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
- **Built-in Scaling:** Inputs and targets are standardized inside `fit`, and `predict` applies the same statistics and converts the output back to the original scale.

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
│   │   └── output_layer.py        # Output layer and error signal
│   └── activations/
│       ├── _base.py               # Abstract activation interface (calc / derv)
│       ├── activation.py          # Identity
│       ├── relu.py                # ReLU
│       ├── sigmoid.py             # Sigmoid
│       └── softmax.py             # Softmax
├── main.py                        # Neural network experiments and comparison
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
1. Output error: `δ = (ŷ − y) ⊙ f'(z)`
2. Gradient of the weights: `dW = A_prev · δᵀ`
3. Error sent to the previous layer: `δ_prev = (W · δ) ⊙ f'(z_prev)`, **with the bias row dropped**, because no error can flow into a node that has no input
4. Update: `W ← W − lr · dW / n`

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
| `acts_out`  | `None`  | Output activation: `"relu"`, `"sigmoid"`, `"softmax"`, or `None` (identity) |
| `niter`     | `1e4`   | Number of full-batch gradient descent iterations                            |
| `lr`        | `0.1`   | Learning rate                                                               |
| `hide_func` | `Relu`  | Hidden activation class (`Relu`, `Sigmoid`), or `None` for no activation    |

---

## 📊 Experiments

Run the experiments with:

```bash
python main.py
```

`main.py` trains the neural network on known functions and compares it with a reference solution (the closed-form least-squares line from `numpy.polyfit`).

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

---

## 🔭 Roadmap

- Neural network: support multiple outputs and classification (Softmax with cross-entropy). Currently the network supports single-output regression, because the targets are standardized.
- Neural network: reuse the `Optimizer` / `Adam` strategies and add mini-batch training.
- Add gradient-check unit tests for the neural network, and validate the linear models against scikit-learn (`LinearRegression`, `Ridge`, `Lasso`).
- Add more optimizers (SGD with momentum, RMSProp) and ElasticNet regularization.
- Add early stopping and learning-rate scheduling.
- Add `get_params` / `set_params` and make `fit` return `self` for full scikit-learn compatibility.
- Package the library with `pyproject.toml` so it can be installed with `pip`.

---

## 🔗 Author

**Mahamed Emad**

- GitHub: [Mahamed-Emad](https://github.com/Mahamed-Emad)
- LinkedIn: [mahamed-emad](https://www.linkedin.com/in/mahamed-emad/)

---

## 📜 License

Distributed under the MIT License.
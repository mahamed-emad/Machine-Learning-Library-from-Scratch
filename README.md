# 🧠 ML Library from Scratch

A modular machine learning library built **from scratch with NumPy**, implementing Linear, Ridge, Lasso, and Polynomial Regression with decoupled **Optimizer** and **Regularizer** strategies. Built by **[Mahamed Emad](https://www.linkedin.com/in/mahamed-emad/)** as a hands-on way to understand the math and the software design behind classical ML, without relying on scikit-learn.

---

## 📌 Features

- **From-Scratch Implementations:** Linear Regression and Polynomial Regression trained with vectorized mini-batch gradient descent, using only NumPy.
- **Pluggable Optimizers:** Optimizers (e.g. `Adam`) live in their own module and are passed into `fit`, so the training loop never changes when you swap them.
- **Pluggable Regularizers:** `Lasso` (L1) and `Ridge` (L2) penalties are separate strategy classes whose gradients are added to the loss gradient. Combine any model with any regularizer.
- **Strategy Pattern:** Optimizers and regularizers share abstract base classes (Python `abc`), making new ones easy to add.
- **Polynomial Feature Expansion:** Generates all polynomial and interaction terms up to a chosen degree using `combinations_with_replacement`.
- **K-Fold Cross-Validation:** A `Kfold` wrapper in `model_selection` that works with any base model.
- **scikit-learn-inspired API:** Familiar `fit` / `predict` / `score` methods, where `score` returns R².

---

## 📁 Repository Structure

```text
ML_library/
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
├── main.py                        # Example usage
└── README.md
```

---

## ⚙️ How It Works

Each training step of `Linear_Regression.fit` follows the same recipe, and only the strategies change:

1. Shuffle the data every epoch and split it into mini-batches.
2. Compute the prediction error and the MSE gradient: `Xᵀ(ŷ − y) / m`.
3. Add the **regularizer gradient** (`regularizer.grad(weights)`).
4. Pass the total gradient through the **optimizer** (`optimizer.update(grad)`).
5. Update the weights: `w ← w − lr · update`.

Because the loop only talks to the abstract optimizer and regularizer interfaces, adding a new strategy (e.g. Momentum or ElasticNet) never requires touching the model code.

`Polynomial_Regression` inherits from `Linear_Regression` and only adds a feature-expansion step before `fit` and `predict`, so it automatically supports every optimizer and regularizer.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas
```

### Installation

```bash
git clone https://github.com/Mahamed-Emad/ML_library.git
cd ML_library
```

### Quick Example

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

### Hyperparameters

| Parameter | Default | Description                                        |
| --------- | ------- | -------------------------------------------------- |
| `lr`      | `0.001` | Learning rate                                      |
| `batches` | `None`  | Mini-batch size (`None` = full-batch gradient descent) |
| `itr`     | `1e4`   | Number of epochs                                   |
| `seed`    | `42`    | Random seed                                        |

---

## 🔭 Roadmap

- Validate results against scikit-learn (`LinearRegression`, `Ridge`, `Lasso`) with unit tests.
- Add more optimizers (SGD with momentum, RMSProp) and ElasticNet regularization.
- Add early stopping and learning-rate scheduling.
- Add `get_params` / `set_params` and make `fit` return `self` for full scikit-learn compatibility.
- Package the library with `pyproject.toml` so it can be installed with `pip`.

---

## 🔗 Author

**Mahamed Emad**

- GitHub: [github.com/Mahamed-Emad](https://github.com/Mahamed-Emad)
- LinkedIn: [linkedin.com/in/mahamed-emad](https://www.linkedin.com/in/mahamed-emad/)

---

## 📜 License

Distributed under the MIT License.

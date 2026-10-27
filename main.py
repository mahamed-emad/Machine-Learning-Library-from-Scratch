import numpy as np
from neural_network import Neural_Network, Relu


def mse(y, p): return np.mean((y - p) ** 2)
def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


# ------------------------------------------------------------------
# Test 1: linear function  y = 5x + 10
# A network without a hidden activation (hide_func=None) is a linear model,
# so it must find the exact line (even outside the training range).
# ------------------------------------------------------------------
def func(i): return 5 * i + 10

X = np.array(list(range(100)), dtype=float)
Y = np.array([func(i) for i in range(100)], dtype=float)

np.random.seed(42)
nn = Neural_Network(dimens=[1, 16, 8, 1], niter=5000, hide_func=None)
nn.fit(X, Y)

lst = [60, 153, 3, -15]                      # 153 and -15 are outside the training range
test_x = np.array(lst, dtype=float)
true_y = np.array(list(map(func, lst)), dtype=float)
nn_pred = nn.predict(test_x)
ls_pred = np.polyval(np.polyfit(X, Y, 1), test_x)    # least-squares line (reference solution)

print("=== Test 1: y = 5x + 10 ===")
print(f"{'x':>8} {'true':>10} {'neural net':>12} {'least squares':>15}")
for x_i, t, a, b in zip(lst, true_y, nn_pred, ls_pred):
    print(f"{x_i:>8} {t:>10.2f} {a:>12.2f} {b:>15.2f}")
print(f"max |error|  neural net: {np.abs(true_y - nn_pred).max():.2e} | least squares: {np.abs(true_y - ls_pred).max():.2e}")


# ------------------------------------------------------------------
# Test 2: non-linear functions
# Compare: ReLU network  vs  network without hidden activation  vs  best straight line.
# Only the ReLU network can learn a curve.
# ------------------------------------------------------------------
functions = {
    "y = x^2":    lambda x: x ** 2,
    "y = sin(x)": lambda x: np.sin(x),
}

for name, f in functions.items():
    X = np.linspace(-3, 3, 200)
    Y = f(X)
    test_x = np.linspace(-2.9, 2.9, 57)      # points the network has never seen
    true_y = f(test_x)

    np.random.seed(42)
    relu_net = Neural_Network(dimens=[1, 16, 8, 1], niter=20000, lr=0.1, hide_func=Relu)
    relu_net.fit(X, Y)

    np.random.seed(42)
    lin_net = Neural_Network(dimens=[1, 16, 8, 1], niter=20000, lr=0.1, hide_func=None)
    lin_net.fit(X, Y)

    preds = {
        "ReLU network": relu_net.predict(test_x),
        "no activation": lin_net.predict(test_x),
        "straight line": np.polyval(np.polyfit(X, Y, 1), test_x),
    }

    print(f"\n=== Test 2: {name} ===")
    print(f"{'x':>7} {'true':>9} " + " ".join(f"{k:>14}" for k in preds))
    for i in range(0, len(test_x), 8):
        print(f"{test_x[i]:>7.2f} {true_y[i]:>9.3f} " + " ".join(f"{p[i]:>14.3f}" for p in preds.values()))
    print(f"{'model':>16} {'MSE':>10} {'R2':>8}")
    for k, p in preds.items():
        print(f"{k:>16} {mse(true_y, p):>10.4f} {r2(true_y, p):>8.3f}")
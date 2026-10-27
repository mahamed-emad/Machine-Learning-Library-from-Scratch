import numpy as np
from neural_network.neural_network import Neural_Network

# fit / predict take x = (samples, features)  (the network transposes it internally)

def layers_of(net):
    out, n = [], net.propagation.head
    while n is not None:
        if hasattr(n, "weights"): out.append(n)
        n = n.next
    return out

def loss(net, x, y, kind):
    yhat = net.propagation.forward(x)
    if kind == "ce": return -np.sum(y * np.log(yhat + 1e-12)) / x.shape[1]
    return 0.5 * np.sum((yhat - y) ** 2) / x.shape[1]

def grad_check(dimens, acts, kind, x, y):
    np.random.seed(0)
    net = Neural_Network(list(dimens), acts, niter=0, lr=1.0)
    net.fit(x, y)                                   # niter=0 -> only builds the layers
    xs = net.colm_x(x)                              # scaled, (features, samples)
    layers = layers_of(net)
    for l in layers: l.weights = np.random.randn(*l.weights.shape) * 0.7
    w0 = [l.weights.copy() for l in layers]
    y2 = y.reshape(1, -1) if y.ndim == 1 else y
    net.propagation.forward(xs); net.propagation.backward()     # lr = 1 -> step == gradient
    ana = [a - l.weights for a, l in zip(w0, layers)]
    for l, a in zip(layers, w0): l.weights = a.copy()
    worst = 0.0
    for l, g in zip(layers, ana):
        num = np.zeros_like(g)
        for idx in np.ndindex(g.shape):
            old = l.weights[idx]
            l.weights[idx] = old + 1e-6; p = loss(net, xs, y2, kind)
            l.weights[idx] = old - 1e-6; m = loss(net, xs, y2, kind)
            l.weights[idx] = old
            num[idx] = (p - m) / 2e-6
        worst = max(worst, np.linalg.norm(g - num) / (np.linalg.norm(g) + np.linalg.norm(num) + 1e-12))
    return worst

rng = np.random.default_rng(1)
x = rng.standard_normal((30, 4))
cases = [
    ("hidden x1 + sigmoid", [0, 5, 1], "sigmoid", "mse", (rng.random(30) > .5).astype(float)),
    ("hidden x3 + identity", [0, 8, 6, 4, 1], None, "mse", rng.standard_normal(30)),
    ("hidden x1 + softmax", [0, 5, 3], "softmax", "ce", np.eye(3)[rng.integers(0, 3, 30)].T),
    ("no hidden + sigmoid", [0, 1], "sigmoid", "mse", (rng.random(30) > .5).astype(float)),
]
ok = True
for name, d, a, k, y in cases:
    err = grad_check(d, a, k, x, y)
    good = err < 1e-6; ok &= good
    print(f"gradient check | {name:22s} | rel.err = {err:.1e} | {'PASS' if good else 'FAIL'}")

# your exact setup: 200 samples x 30 features, y = (200,)
xs, ys = rng.standard_normal((200, 30)), rng.standard_normal(200)
net = Neural_Network([5, 8, 6, 4, 1], "relu", niter=5); net.fit(xs, ys)
p = net.predict(xs); good = p.shape == (200,) and np.isfinite(p).all(); ok &= good
print(f"shape test     | predict -> {p.shape} | {'PASS' if good else 'FAIL'}")

# predict must use the TRAINING mean/std (a single new row must not give nan)
p1 = net.predict(xs[:1]); good = np.isfinite(p1).all() and np.allclose(p1, p[:1]); ok &= good
print(f"scaling test   | single-row predict == full predict | {'PASS' if good else 'FAIL'}")

print("\nALL OK" if ok else "\nSOMETHING IS WRONG: your files differ from the delivered ones")
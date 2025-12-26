import numpy as np

def generate():

    omega = 0.345
    xs = [0.1 * (t-50) for t in range(500)]
    ys = [7.9824 * np.cos(omega * x) + 2.8129 + np.random.normal(0,3) for x in xs]

    for x, y in zip(xs, ys):
        print(f"{x}, {y}")


if __name__ == "__main__":
    generate()


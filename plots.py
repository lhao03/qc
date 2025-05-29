import os.path

import matplotlib.pyplot as plt


def plot_energies(xpoints, points, labels, title, dir):
    plt.clf()
    for set_of_points in points:
        plt.plot(xpoints, set_of_points)
    plt.xlabel("Bond Length (Å)")
    plt.ylabel("Energy (Ha)")
    plt.legend(labels)
    plt.title(title)
    plt.savefig(os.path.join(dir, f"{title}.png"))

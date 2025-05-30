import os.path

import matplotlib.pyplot as plt

colors = [
    "#377eb8", # blue
    "#ff7f00", # orange
    "#f781bf", # pink
    "#a65628", # brown
    "#984ea3", # purple
    "#e41a1c", # red
]


def plot_energies(xpoints, points, labels, title, dir):
    plt.clf()
    for set_of_points in points:
        plt.plot(xpoints, set_of_points)
    plt.xlabel("Bond Length (Å)")
    plt.ylabel("Energy (Ha)")
    plt.legend(labels)
    plt.title(title)
    plt.savefig(os.path.join(dir, f"{title}.png"))

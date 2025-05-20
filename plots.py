import matplotlib.pyplot as plt

def plot_energies(xpoints, tru_points, min_points, title):
    plt.plot(xpoints, tru_points)
    plt.plot(xpoints, min_points)
    plt.xlabel("Bond Length (Å)")
    plt.ylabel("Energy (Ha)")
    plt.title(title)
    plt.savefig(f"{title}.png")
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# random seeds two dimensional points
P = [(random.random()*2.0, random.random()*2.0) for _ in range(100)]
P = np.array(P)


def calculate_sum(P, centroids, cluster):
    sum = 0

    for i, j in enumerate(P):
        x = centroids[int(cluster[i]), 0]-j[0]
        y = centroids[int(cluster[i]), 1]-j[1]
        sum += np.sqrt(x**2 +y**2)
    return sum


# Kmean function
def kmeans(P, k):

    cluster = np.zeros(P.shape[0])
    random_cen = np.random.choice(len(P), size=k, replace=False)
    centroids = P[random_cen, :]
    dif = 1

    while dif:
        for i, j in enumerate(P):
            min_distance = float('inf')
            for idx, centroid in enumerate(centroids):
                x_dis = centroid[0]-j[0]
                y_dis = centroid[1]-j[1]
                dis = np.sqrt(x_dis**2 + y_dis**2)
                if min_distance > dis:
                    min_distance = dis
                    cluster[i] = idx

        assign_centroids = pd.DataFrame(P).groupby(by=cluster).mean().values

        if np.count_nonzero(centroids-assign_centroids) == 0:
            dif = 0
        else:
            centroids = assign_centroids

    return centroids, cluster


error_list = []

for k in range(1, 10):
    centroids, cluster = kmeans(P, k)
    error = calculate_sum(P, centroids, cluster)
    error_list.append(error)

k = 4
centroids, cluster = kmeans(P, k)


def create_plot():
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(5, 5))

    sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=cluster)
    sns.scatterplot(centroids[:, 0], centroids[:, 1], s=100, color='y')

    return f


# simple interface using tkinter
root = tk.Tk()

label1 = tk.Label(root, text='Coordinates of centroids at the end of iteration', justify = 'center')
label1.pack(pady=5)
label2 = tk.Label(root, text=centroids, justify = 'center')
label2.pack(pady=5)

label3 = tk.Label(root, text='Cluster for each data points', justify = 'center')
label3.pack(pady=5)
label4 = tk.Label(root, text=cluster, justify = 'center')
label4.pack(pady=5)

fig = create_plot()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()
button = tk.Button(root, text="Quit", command=root.destroy)
button.pack()

root.mainloop()

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import numpy as np

# Adapted from example at https://matplotlib.org/3.1.1/users/event_handling.html
class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes != self.line.axes: return
        if event.button == MouseButton.LEFT:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
        elif self.xs:
            self.xs.pop()
            self.ys.pop()
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

def path_input_plot(data, naverage=100, cmap='Blues', cbar_label='frame index'):
    fig = plt.figure()
    
    for X in data:
        assert X.shape[1] == 2
        plt.scatter(*X.T, s=1, c=range(len(X)), cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)

    k = naverage // 2
    for i, X in enumerate(data):
        path = []
        for n in range(len(X)):
            start, stop = max(0, n - k), min(len(X), n + 1 + k)
            path.append(X[start:stop].mean(axis=0))
        path = np.asarray(path)
        plt.plot(path[:, 0], path[:, 1], color='tab:red')
    
    line, = plt.plot([], [], marker='o', markerfacecolor='lawngreen', markeredgewidth=1,
                    color='k', linewidth=1, zorder=10)
    linebuilder = LineBuilder(line)
    
    return fig, linebuilder.xs, linebuilder.ys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton


class LineBuilder:
    """An event handler for drawing Line2D objects.

    This is a slight adaptation of the LineBuilder example at 
    https://matplotlib.org/users/event_handling.html. In this 
    version, points may be deleted by right clicking.
    
    """

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


def path_input_plot(data, window=100, cmap='Blues'):
    """Create an interactive plot for manual path input.

    Parameters
    ----------
    data : ndarray (T, 2) list of ndarray (T_i, 2)
        2D trajectories. `T` is the number of frames.

    window : int, optional
        Width of the averaging window used for trajectory smoothing.

    cmap : str, optional
        Color map.

    Returns
    -------
    fig : matplotlib.figure.Figure

    ax : matplotlib.figure.Axes

    line : matplotlib.lines.Line2D
        The manually input path. The x- and y-coordinates may be 
        accessed via the `get_xdata()` and `get_ydata()` methods.

    """
    if isinstance(data, np.ndarray):
        data = [data]    
    
    fig, ax = plt.subplots() 

    for X in data:
        s = ax.scatter(*X.T, s=1, c=range(len(X)), cmap=cmap)
    cbar = fig.colorbar(s)
    cbar.set_label('frame index')

    k = window // 2
    for X in data:
        path = []
        for n in range(len(X)):
            start, stop = max(0, n - k), min(len(X), n + 1 + k)
            path.append(X[start:stop].mean(axis=0))
        path = np.asarray(path)
        ax.plot(path[:, 0], path[:, 1], color='tab:red')

    line, = ax.plot([], [], marker='o', markerfacecolor='lawngreen', 
                    markeredgewidth=1, color='k', linewidth=1, zorder=10)
    linebuilder = LineBuilder(line)
        
    return fig, ax, line


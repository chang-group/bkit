from scipy.interpolate import interp1d
import numpy as np

def interpolate_path(path, npoints=1000, image_spacing=1.0):
    t = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    t = np.insert(t, 0, 0)
    
    f = interp1d(t, path.T, kind='cubic')
    path_new = f(np.linspace(0, t[-1], npoints)).T
    
    s = np.cumsum(np.linalg.norm(np.diff(path_new, axis=0), axis=1))
    s = np.insert(s, 0, 0)
    g = interp1d(s, path_new.T, kind='linear')
    images = g(np.arange(0, s[-1], image_spacing)).T
    
    return path_new, images

import numpy as np
import scipy.spatial
import scipy.stats
import itertools
import networkx as nx

class Schedule:
    def __init__(self, a=None):
        if a:
            self.labels = [a]
            self.lengths = np.array([0])
        else:
            self.labels = []
            self.lengths = np.array([])
    
    def append(self, a, t):
        assert t >= 0
        self.labels.append(a)
        self.lengths = np.append(self.lengths, [t])

    def length(self):
        return self.lengths.sum()

    def scale(s):
        assert s >= 0
        self.lengths = self.lengths * s
        
    def reduce(self):
        selectors = self.lengths != 0
        self.labels = list(itertools.compress(self.labels, selectors))
        self.lengths = self.lengths[selectors]
    
    def is_reduced(self):
        return not np.any(self.lengths == 0)
    
    def __str__(self):
        return ''.join([f'({a}, {t})' for a, t in zip(self.labels, self.lengths)])

    def __getitem__(self, key):
        if isinstance(key, int):
            return (self.labels[key], self.lengths[key])
        else:
            return Schedule(self.labels[key], self.lengths[key])
    
    def __delitem__(self, key):
        del self.labels[key]
        self.lengths = np.delete(self.lengths, key)
    
    def __iter__(self):
        return zip(self.labels, self.lengths)
        
    def __len__(self):
        return len(self.labels)

def milestone_schedule(dtraj, dt=1):
    schedule = Schedule(frozenset({None, dtraj[0]}))
    for i, j in zip(dtraj[:-1], dtraj[1:]):
        if j in schedule.labels[-1]:
            schedule.lengths[-1] += dt
        else:
            schedule.append(frozenset({i, j}), dt)
    schedule.reduce()
    return schedule

class MilestoningModel:
    def __init__(self, anchors, cutoff=np.inf):
        nanchors, ndim = anchors.shape

        self._graph = nx.Graph()
        if ndim > 1:
            tri = scipy.spatial.Delaunay(anchors)
            indptr, indices = tri.vertex_neighbor_vertices
            self._graph.add_edges_from([(i, j) for i in range(nanchors-1) for
                                        j in indices[indptr[i]:indptr[i+1]]])
        else:
            self._graph.add_edges_from([(i, i+1) for i in range(nanchors-1)])

        self._anchor_kdtree = scipy.spatial.cKDTree(anchors, copy_data=True)    
        self._cutoff = cutoff
        self._parent_cell = list(range(nanchors))
        if np.isfinite(cutoff):
            self._parent_cell.append(None)

        self._dtrajs = []
        self._schedules = []
        self._dts = []
    
    @property
    def milestones(self):
        return [set(e) for e in self._graph.edges]  
    
    def load_trajectory_data(self, trajs, dt=1):        
        for traj in trajs:
            _, indices = self._anchor_kdtree.query(traj, distance_upper_bound=self._cutoff)
            dtraj = [self._parent_cell[n] for n in indices]
            self._dtrajs.append(dtraj)
            self._schedules.append(milestone_schedule(dtraj, dt))
            self._dts.append(dt)
 
    def remove_milestone(self, i, j):
        assert self._graph.has_edge(i, j)
        self._graph = nx.contracted_nodes(self._graph, i, j, self_loops=False)
        self._parent_cell = [i if k == j else k for k in self._parent_cell]
        for q, dtraj in enumerate(self._dtrajs):
            self._dtrajs[q] = [i if k == j else k for k in dtraj]
            self._schedules[q] = milestone_schedule(dtraj, self._dts[q])
    
    def estimate(self):
        # TODO: Check for jumps between non-adjacent cells.

        M = self._graph.number_of_edges()
        self._counts = np.zeros((M, M), dtype=np.int32)
        self._lifetimes = [[] for m in range(M)]

        index = dict((frozenset(e), m) 
                     for m, e in enumerate(self._graph.edges))
        for schedule in self._schedules:
            it = zip(schedule[:-1], schedule[1:])
            for (source, lifetime), (target, _) in enumerate(it):
                if source in index and target in index:
                    self._counts[index[source], index[target]] += 1
                    self._lifetimes[index[source]].append(lifetime)
 
        self.K = self._counts / np.sum(self._counts, axis=1)[:, np.newaxis]
        self.t = np.array(list(map(np.mean, self._lifetimes)))
        self.t_stderr = np.array(list(map(scipy.stats.sem, self._lifetimes)))


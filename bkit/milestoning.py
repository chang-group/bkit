import msmtools.analysis as analysis
import networkx as nx
import numpy as np
import scipy.spatial as spatial
import scipy.stats as stats
from _schedule import Schedule


class MilestoningModel:
    """Milestoning model with given rate matrix"""

    def __init__(self, rate_matrix):
        """Initialize milestoning model.

        Parameters
        ----------
        rate_matrix : ndarray (m, m)

        """

        assert analysis.is_rate_matrix(rate_matrix)
        self._Q = rate_matrix
        self._t = -1 / np.diag(self._Q)
        self._P = np.fill_diagonal(self._Q * self._t[:, np.newaxis], 0)
        self._q = analysis.stationary_distribution(self._P)
        self._p = self._q * self._t
        self._p /= sum(self._p)

    @property
    def rate_matrix(self):
        return self._Q

    @property
    def equilibrium_populations(self):
        return self._p

    @property
    def transition_matrix(self):
        return self._P

    @property
    def stationary_fluxes(self):
        return self._q

    @property
    def mean_lifetimes(self):
        return self._t
    
    def mfpts(self, target):
        """Mean first passage times to a set of target milestones

        Parameters
        ----------
        target : int or list of int
            indices of target milestones (sinks)

        Returns
        -------
        mfpts : ndarray (m,)
            mean first passage times to target set

        """

        P = np.copy(self._P)
        P[target, :] = 0
        return np.linalg.solve(np.fill_diagonal(-P, 1), self._t)


class MilestoningMLEstimator:
    """Maximum likelihood milestoning estimator"""

    def __init__(self, anchors, cutoff=np.inf, boxsize=None):
        """Initialize milestoning estimator.

        Parameters
        ----------
        anchors : ndarray (N, d) or list of ndarray (N_i, d)
            Generating points for Voronoi tessellation. 
            If a list of ndarrays is given, each group of anchors
            corresponds to a union of Voronoi cells that should be
            treated as a single cell.

        cutoff : positive float, optional
            Maximum distance to nearest anchor. The region beyond
            this cutoff is treated as an absorbing (cemetery) state.

        boxsize : array_like or scalar, optional (not yet implemented)
            Apply d-dimensional toroidal topology (periodic boundaries)

        """

        if type(anchors) is np.ndarray:
            nanchors, ndim = anchors.shape
            self._parent_cell = list(range(nanchors))
        else:
            ndim = anchors[0].shape[1]
            assert all(a.shape[1] == ndim for a in anchors[1:])
            self._parent_cell = [i for i, a in enumerate(anchors) for x in a]
            anchors = np.concatenate(anchors)

        G = nx.Graph()
        if ndim > 1:
            tri = spatial.Delaunay(anchors)
            indptr, indices = tri.vertex_neighbor_vertices
            G.add_edges_from([(k, l) for k in range(nanchors-1) 
                              for l in indices[indptr[k]:indptr[k+1]]])
        else:
            G.add_edges_from([(k, k+1) for k in range(nanchors-1)])
        partition = lambda k, l: self._parent_cell[k] == self.parent_cell[l]
        self._graph = nx.quotient_graph(G, partition, relabel=True)

        self._anchor_kdtree = spatial.cKDTree(anchors)    
        self._cutoff = cutoff
        if np.isfinite(cutoff):
            self._parent_cell.append(None)

        self._dtrajs = []
        self._dts = []
        self._schedules = []
    
    @property
    def milestones(self):
        return [set(e) for e in self._graph.edges]  
 
    def load_trajectory_data(self, trajs, dt=1.):        
        for traj in trajs:
            _, indices = self._anchor_kdtree.query(traj, distance_upper_bound=self._cutoff)
            dtraj = np.fromiter((self._parent_cell[k] for k in indices), int)
            self._dtrajs.append(dtraj)
            self._dts.append(dt)
            self._schedules.append(_milestone_schedule(dtraj, dt))
 
    def remove_milestone(self, i, j):
        assert self._graph.has_edge(i, j)
        self._graph = nx.contracted_nodes(self._graph, i, j, self_loops=False)
        self._parent_cell = [i if k == j else k for k in self._parent_cell]
        for q, dtraj in enumerate(self._dtrajs):
            self._dtrajs[q] = [i if k == j else k for k in dtraj]
            self._schedules[q] = _milestone_schedule(dtraj, self._dts[q])
    
    def estimate(self):
        # TODO: Check for jumps between non-adjacent cells.

        M = self._graph.number_of_edges()
        self._count_matrix = np.zeros((M, M), dtype=np.int32)
        self._lifetimes = [[] for m in range(M)]

        index = dict((frozenset(e), m) 
                     for m, e in enumerate(self._graph.edges))
        for schedule in self._schedules:
            it = zip(schedule[:-1], schedule[1:])
            for (source, lifetime), (target, _) in enumerate(it):
                if source in index and target in index:
                    self._count_matrix[index[source], index[target]] += 1
                    self._lifetimes[index[source]].append(lifetime)
 
        self.K = self._count_matrix / np.sum(self._count_matrix, 
                                             axis=1)[:, np.newaxis]
        self.t = np.fromiter(map(np.mean, self._lifetimes), float)
        self.t_stderr = np.fromiter(map(stats.sem, self._lifetimes), float)


def _milestone_schedule(dtraj, dt=1.):
    initial_milestone = frozenset({None, dtraj[0]})
    schedule = Schedule([initial_milestone], [0])
    for i, j in zip(dtraj[:-1], dtraj[1:]):
        if j in schedule.labels[-1]:
            schedule.lengths[-1] += dt
        else:
            schedule.append(frozenset({i, j}), dt)
    schedule.reduce()
    return schedule


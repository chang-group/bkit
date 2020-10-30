import networkx as nx
import numpy as np
import scipy.spatial as spatial
from bkit.markov import ContinousTimeMarkovChain


class MarkovianMilestoningModel(ContinuousTimeMarkovChain):
    """Milestoning process governed by a continuous-time Markov chain."""

    def __init__(self, rate_matrix, milestones):
        """Model with a given rate matrix and indexed set of milestones.

        Parameters
        ----------
        rate_matrix : (M, M) ndarray
            Matrix :math:`Q` of transition rates between milestones.

        milestones : list, length M
            The milestones corresponding to the indices of :math:`Q`.

        """
        super().__init__(rate_matrix)
        self.milestones = milestones

    @property
    def milestones(self):
        """The milestones in indexed order."""
        return self._milestones

    @milestones.setter
    def milestones(self, value):
        if len(value) != self.n_states:
            msg = 'number of milestones must match dimension of rate matrix'
            raise ValueError(msg)
        self._milestones = milestones

    @property
    def transition_kernel(self):
        """Transition probability matrix :math:`K`."""
        return self.jump_chain.transition_matrix

    @property
    def mean_lifetimes(self):
        """The mean waiting time associated with each milestone.""" 
        return -1 / np.diag(self.rate_matrix)

    @property
    def stationary_fluxes(self):
        """The stationary flux vector :math:`q`."""
        return self.jump_chain.stationary_distribution


class MilestoningEstimator:

    def __init__(self, milestones):
        """Initialize estimator.

        Parameters
        ----------
        milestones : List[Hashable]
            The milestones over which trajectories are decomposed.

        """
        self._milestones = milestones
        self._ix = dict((a, ix) for ix, a in enumerate(milestones))
        self._schedules = []
        self._count_matrix = np.zeros((len(milestones), len(milestones),
                                      dtype=int)
        self._total_times = np.zeros(len(milestones))

    def load_schedules(self, schedules, forward=False):
        """Incorporate data in the form of milestone schedules.

        Parameters
        ----------
        schedules : list of lists of pairs
            Lists of (milestone, lifetime) pairs obtained by 
            trajectory decomposition.

        """
        if forward:
            raise NotImplementedError()

        for schedule in schedules:
            it = zip(schedule[:-1], schedule[1:])
            for (a, t), (b, _) in it:
                if a not in self._ix or b not in self._ix:
                    continue
                self._count_matrix[self._ix[a], self._ix[b]] += 1
                self._total_times[self._ix[a]] += t
        self._schedules += schedules

    def fetch_markov_model(self):
        """Return the maximum likelihood Markovian milestoning model."""
        total_counts = np.sum(self._count_matrix, axis=1)
        Q = ((self._count_matrix - np.diag(total_counts))
             / self._total_times[:, np.newaxis])
        return MarkovianMilestoningModel(Q, self._milestones)


class TrajectoryDecomposer:
    """Path decomposition by milestoning with Voronoi tessellations."""

    def __init__(self, anchors, cutoff=np.inf, boxsize=None):
        """Trajectory decomposer with given (partitioned) set of anchors.

        Parameters
        ----------
        anchors : ndarray (N, d) or list of ndarray (N_i, d)
            Generating points for Voronoi tessellation. 
            If a list of ndarrays is given, each subset of anchors
            indicates a union of Voronoi cells that should be treated 
            as a single cell.

        cutoff : positive float, optional
            Maximum distance to nearest anchor. The region of space 
            beyond the cutoff is treated as a cell labeled `None`.
 
       boxsize : array_like or scalar, optional (not yet implemented)
            Apply d-dimensional toroidal topology (periodic boundaries).

        """
        if boxsize:
            raise NotImplementedError('patience')

        if type(anchors) is np.ndarray:
            self._parent_cell = dict((k, k) for k in range(len(anchors))
        else:
            self._parent_cell = dict((k, i) for i, arr in enumerate(anchors)
                                            for k in range(len(arr))]
            anchors = np.concatenate(anchors)
        n_anchors, n_dim = anchors.shape 

        G = nx.Graph()
        if n_dim > 1:
            tri = spatial.Delaunay(anchors)
            indptr, indices = tri.vertex_neighbor_vertices
            G.add_edges_from([(k, l) for k in range(n_anchors-1) 
                              for l in indices[indptr[k]:indptr[k+1]]])
        else:
            G.add_edges_from([(k, k+1) for k in range(n_anchors-1)])
        partition = lambda k, l: self._parent_cell[k] == self._parent_cell[l]
        self._graph = nx.quotient_graph(G, partition, relabel=True)

        self._kdtree = spatial.cKDTree(anchors)    
        
        self._cutoff = cutoff
        if np.isfinite(cutoff):
            self._parent_cell.append(None)
    
    @property
    def cell_anchor_mapping(self):
        """Mapping from cells to anchor points."""
        mapping = dict((i, []) for i in self._graph.nodes)
        for k, i in self._parent_cell.items():
            mapping[i] += self._kdtree.data[k]
        return mapping

    @property
    def milestones(self):
        """List of milestones."""
        return list(frozenset(e) for e in self._graph.edges)
 
    def remove_milestone(self, i, j):
        """Remove the milestone between cells i and j.

        Parameters
        ----------
        i, j : int
            Indices of cells to merge. Cell `j` is merged into cell `i`, 
            and `j` is removed from the cell index set.

        """
        if not self._graph.has_edge(i, j):
            raise ValueError('milestone does not exist; cannot remove it')
        self._graph = nx.contracted_nodes(self._graph, i, j, self_loops=False)
        for k in self._parent_cell:
            if self._parent_cell[k] == j:
                self._parent_cell[k] = i

    def decompose(self, trajs, dt=1, forward=False):
        """Map trajectories to milestone schedules.

        Parameters
        ----------
        trajs : ndarray (T, d) or list of ndarray (T_i, d)
            Trajectories to be decomposed.

        dt : int or float, optional
            Trajectory sampling interval, positive (>0)

        forward : bool, optional
            If true, track the next milestone hit (forward commitment),
            rather than the last milestone hit (backward commitment).
    
        Returns
        -------
        schedules : list of list of tuple
            Sequences of (milestone, lifetime) pairs obtained by 
            trajectory decomposition.

        """ 
        if type(trajs) is np.ndarray:
            trajs = [trajs]
        return [self._traj_to_milestone_schedule(traj, dt, forward)
                for traj in trajs]

    def _traj_to_milestone_schedule(self, traj, dt=1, forward=False):
        _, ktraj = self._kdtree.query(traj, distance_upper_bound=self._cutoff)
        dtraj = [self._parent_cell[k] for k in ktraj]
        return dtraj_to_milestone_schedule(dtraj, dt, forward)
 

def dtraj_to_milestone_schedule(dtraj, dt=1, forward=False):
    """'Milestone' a discrete trajectory.

    Parameters
    ----------
    dtraj : list of int or ndarray(T, dtype=int)
        A discrete trajectory.

    dt : int or float, optional
        Trajectory sampling interval, positive (>0)

    forward : bool, optional
        If true, track the next milestone hit (forward commitment),
        rather than the last milestone hit (backward commitment).

    Returns
    -------
    schedule : list of tuple
        Sequence of (milestone, lifetime) pairs. For backward milestoning,
        the first milestone is set to `{None, dtraj[0]}`. For forward
        milestoning, the last milestone is set to `{dtraj[-1], None}`.
        (In fact the milestones are `frozenset`s, which are hashable.)
    
    """
    if forward_milestoning:
        dtraj = reversed(dtraj)
    milestones = [frozenset({None, dtraj[0]})]
    lifetimes = [0]
    for i, j in zip(dtraj[:-1], dtraj[1:]):
        lifetimes[-1] += dt
        if j not in milestones[-1]:
            milestones.append(frozenset({i, j}))
            lifetimes.append(0)
    schedule = list(zip(milestones, lifetimes))
    if forward_milestoning:
        return reversed(schedule)
    return schedule


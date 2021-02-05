"""Tools for estimation and analysis of Markovian milestoning models."""

import collections
import msmtools.estimation as estimation
import msmtools.util.types
import numpy as np
import scipy.spatial
import bkit.ctmc as ctmc


class MarkovianMilestoningModel(ctmc.ContinuousTimeMarkovChain):
    """A milestoning process governed by a continuous-time Markov chain.

    Parameters
    ----------
    transition_kernel : (M, M) array_like
        Matrix of milestone-to-milestone transition probabilities. Must 
        be a row stochastic matrix (each row sums to 1) with all zeros on
        the diagonal. 

    mean_lifetimes : (M,) array_like
        Vector of average milestone lifetimes.

    milestones : sequence
        Milestone labels. Values must be unique and hashable.
           
    """

    def __init__(self, transition_kernel, mean_lifetimes, milestones):
        Q = ctmc.rate_matrix(transition_kernel, 1/np.asarray(mean_lifetimes))
        super().__init__(Q, states=milestones)

    @property
    def transition_kernel(self):
        """(M, M) ndarray: Alias for ``jump_matrix``."""
        return self.jump_matrix

    @property
    def mean_lifetimes(self):
        """(M,) ndarray: Reciprocal of ``jump_rates``."""
        return 1 / self.jump_rates

    @property
    def stationary_flux(self):
        """(M,) ndarray: Stationary flux vector, normalized to 1."""
        q = self.stationary_distribution * self.jump_rates
        return q / q.sum()

    @property
    def stationary_probability(self):
        """(M,) ndarray: Alias for ``stationary_distribution``."""
        return self.stationary_distribution


class MarkovianMilestoningEstimator:
    """Maximum likelihood and Bayesian estimation of Markovian 
    milestoning models.
        
    Parameters
    ----------
    reversible : bool, optional
        If True, restrict the space of transition matrices to those 
        satisfying detailed balance.

    """

    def __init__(self, reversible=True):
        self._reversible = bool(reversible)
        self._model = None

    @property
    def reversible(self):
        """bool: If True, perform reversible estimation."""
        return self._reversible

    def max_likelihood_estimate(self):
        """Return the maximum likelihood estimate.

        Returns
        -------
        MarkovianMilestoningModel
            The model that maximizes the likelihood of the data.

        """
        return self._model
    
    @property
    def count_matrix(self):
        ...

    @property
    def total_times(self):
        ...

    @property
    def milestones(self):
        ...        

    def fit(self, data):
        """Fit the estimator to data.

        Parameters
        ----------
        data : iterable of Sequence[tuple] or dict[tuple, Collection]
            Milestone schedules, or a mapping from pairs of milestone 
            indices to samples of first passage times (FPTs).

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        See Also
        --------
        fit_to_schedules, fit_to_fpts

        """
        if isinstance(data, dict):
            return self.fit_to_fpts(data)
        return self.fit_to_schedules(data)

    def fit_to_schedules(self, schedules):
        """Fit the estimator to milestone schedule data.

        Parameters
        ----------
        schedules : iterable of Sequence[tuple[frozenset, int]]
            Sequences of (milestone index, lifetime) pairs. *Note:*
            No transition statistics will be computed for milestones
            bordering unassigned cells (index -1).

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        """
        first_passage_times = collections.defaultdict(list)
        for schedule in schedules:
            a, t = schedule[0]
            for b, s in schedule[1:]:
                if -1 not in a and -1 not in b:
                    first_passage_times[a, b].append(t)
                a, t = b, s
        return self.fit_to_fpts(first_passage_times)

    def fit_to_fpts(self, first_passage_times):
        """Fit the estimator to first passage time data.

        Parameters
        ----------
        first_passage_times : dict[tuple, Collection]
            Mapping from ordered pairs of milestone indices to samples
            of first passage times. `first_passage_times[a, b]` is 
            a collection of first passage times from from milestone `a` 
            to milestone `b`.

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        """
        milestones = ({a for a, _ in first_passage_times}
                      | {b for _, b in first_passage_times})
        milestones = sorted(milestones, key=lambda a: sorted(a))
        ix = {a: i for i, a in enumerate(milestones)}
        m = len(milestones)

        count_matrix = np.zeros((m, m), dtype=int)
        total_times = np.zeros(m)
        for (a, b), times in first_passage_times.items():
            count_matrix[ix[a], ix[b]] = len(times)
            total_times[ix[a]] += sum(times)
        
        connected = estimation.largest_connected_set(count_matrix,
            directed=(True if self.reversible else False))
        milestones = [milestones[i] for i in connected]
        count_matrix = count_matrix[connected, :][:, connected]
        total_counts = count_matrix.sum(axis=1)
        total_times = total_times[connected]

        K = estimation.transition_matrix(
            count_matrix, reversible=self.reversible)
        np.fill_diagonal(K, 0)
        t = total_times / total_counts
        self._model = MarkovianMilestoningModel(K, t, milestones)

        self._count_matrix = count_matrix
        self._total_counts = total_counts
        self._total_times = total_times

        return self

    def posterior_sample(self, size=100):
        """Generate a sample from the posterior distribution.

        Parameters
        ----------
        size : int, optional
            The sample size, i.e., the number of models to generate.

        Returns
        -------
        Collection[MarkovianMilestoningModel]
            The sampled models, or ``None`` if the estimator has not 
            been fit.

        """
        if self._model is None:
            return None

        sampler = estimation.tmatrix_sampler(
            self._count_matrix, reversible=self.reversible,
            T0=self._model.transition_kernel)
        Ks = sampler.sample(nsamples=size)
        for K in Ks:
            np.fill_diagonal(K, 0)

        rng = np.random.default_rng()
        vs = np.zeros((size, self._model.n_states))
        for i, (n, r) in enumerate(zip(self._total_counts, self._total_times)):
            vs[:, i] = rng.gamma(n, scale=1/r, size=size)

        return [MarkovianMilestoningModel(K, 1/v, self._model.states) 
                for K, v in zip(Ks, vs)]


class TrajectoryColoring:
    """Mapping from continuous-space dynamics to milestoning dynamics.
        
    Parameters
    ----------
    anchors : (N, d) array_like
        Generating points for Voronoi tessellation.

    parent_cell : (N,) array_like of int, optional
        The cell index associated with each anchor. Can be used to 
        define a Voronoi diagram with sites that are sets of anchors
        rather than single anchors. Will default to :code:`range(N)`
        if not provided. 

    boxsize : (d,) array_like or scalar, optional
        Apply `d`-dimensional toroidal topology (periodic boundary 
        conditions).

    cutoff : positive float, optional
        Maximum distance to the nearest anchor. The region of state space
        outside the cutoff is treated as a cell with index -1.

    forward : bool, optional
        If true, track the next milestone hit (forward commitment),
        rather than the last milestone hit (backward commitment).

    """

    def __init__(self, anchors, parent_cell=None, boxsize=None, cutoff=None,
                 forward=False):
        self._kdtree = scipy.spatial.cKDTree(anchors, boxsize=boxsize)
        self.cutoff = cutoff
        self.parent_cell = parent_cell
        self.forward = forward

    @property
    def anchors(self):
        """(N, d) ndarray: Generating points of the Voronoi tessellation."""
        return self._kdtree.data

    @property
    def boxsize(self):
        """(d,) ndarray: Periodic box lengths."""
        return self._kdtree.boxsize

    @property
    def cutoff(self):
        """float: Maximum distance to nearest anchor."""
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        if value is None:
            self._cutoff = np.inf
        else:
            if value <= 0:
                raise ValueError('cutoff must be positive')
            self._cutoff = float(value)

    @property
    def parent_cell(self):
        """(N,) ndarray of int: Cell index associated with each anchor."""
        if np.isfinite(self.cutoff):
            return self._parent_cell[:-1]
        return self._parent_cell

    @parent_cell.setter
    def parent_cell(self, value):
        if value is None:
            value = np.arange(self._kdtree.n)
        else:
            value = msmtools.util.types.ensure_dtraj(value)
            if len(value) != self._kdtree.n:
                msg = 'number of cell indices much match number of anchors'
                raise ValueError(msg)
        if np.isfinite(self.cutoff):
            value = np.append(value, -1) 
        self._parent_cell = value

    @property
    def forward(self):
        """bool: Whether to map to a forward milestoning process."""
        return self._forward

    @forward.setter
    def forward(self, value):
        self._forward = bool(value)

    def transform(self, trajs):
        """Color trajectories according to milestone state.

        Parameters
        ----------
        trajs : (T, d) array_like or list of (T_i, d) array_like
            Trajectories in `d`-dimensional space. (The trajectory lengths
            `T_i` may differ.)

        Returns
        -------
        schedules : list of Sequence[tuple[frozenset, int]]
            Sequences of (milestone index, lifetime) pairs.

        """
        trajs = msmtools.util.types.ensure_traj_list(trajs)
        return [self._traj_to_milestone_schedule(traj) for traj in trajs]

    def __call__(self, trajs):
        return self.transform(trajs)

    def _traj_to_milestone_schedule(self, traj):
        dtraj = self._assign_cells(traj)
        return dtraj_to_milestone_schedule(dtraj, forward=self.forward)
    
    def _assign_cells(self, x):
        _, k = self._kdtree.query(x, distance_upper_bound=self.cutoff)
        return self.parent_cell[k]
 

def dtraj_to_milestone_schedule(dtraj, forward=False):
    """Map a discrete-state trajectory to a milestone schedule.

    Parameters
    ----------
    dtraj : sequence of int
        A discrete-state trajectory, e.g., a sequence of cell or cluster
        indices. The index -1 is reserved to indicate an undefined state.

    forward : bool, optional
        If true, track the next milestone hit (forward commitment),
        rather than the last milestone hit (backward commitment).

    Returns
    -------
    Sequence[tuple[frozenset, int]]
        A sequence of (milestone index, lifetime) pairs. When `forward` 
        is false (ordinary milestoning), the initial milestone index is
        set to ``frozenset({-1, dtraj[0]})``. When `forward` is true, the
        final milestone index is set to ``frozenset({dtraj[-1], -1})``.
 
    """
    dtraj = msmtools.util.types.ensure_dtraj(dtraj)
    dtraj_it = reversed(dtraj) if forward else iter(dtraj)
    i = next(dtraj_it)
    milestones = [frozenset({-1, i})]
    lifetimes = [0]
    for j in dtraj_it:
        lifetimes[-1] += 1
        if j not in milestones[-1]:
            milestones.append(frozenset({i, j}))
            lifetimes.append(0)
        i = j
    if forward:
        return list(zip(reversed(milestones), reversed(lifetimes))) 
    return list(zip(milestones, lifetimes))
 

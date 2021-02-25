"""Tools for estimation and analysis of Markovian milestoning models."""

import collections
import msmtools.estimation as estimation
import msmtools.util.types
import numpy as np
import scipy.spatial
import bkit.ctmc as ctmc


class MilestoneState(frozenset):
    """A milestone state identified by a pair of cells.

    Parameters
    ----------
    i : hashable
        Index of the first cell.
    j : hashable
        Index of the second cell.
 
    Notes
    -----
    A milestone state is a :py:obj:`frozenset` with two elements: the 
    cell indices `i` and `j`. Some useful things to remember:

    * ``MilestoneState(i, j)`` is equal to ``MilestoneState(j, i)``.

    * Two milestone states ``a`` and ``b`` are adjacent (incident) if 
      their intersection ``a & b`` is nonempty.

    * If ``states`` is a collection of milestone states, the set of
      underlying cells is given by ``frozenset.union(*states)``.

    """

    def __new__(cls, i, j):
        return super().__new__(cls, {i, j})

    def __init__(self, i, j):
        self._cells = (i, j)

    def __repr__(self):
        return f'{self.__class__.__name__}{self._cells}'


class MarkovianMilestoningModel(ctmc.ContinuousTimeMarkovChain):
    """A milestoning process governed by a continuous-time Markov chain.

    Parameters
    ----------
    transition_kernel : (M, M) array_like
        Matrix of milestone-to-milestone jump probabilities. Must be a row
        stochastic matrix (each row sums to one) with all zeros on the 
        diagonal.
    mean_lifetimes : (M,) array_like
        Vector of average milestone lifetimes.
    stationary_flux : (M,) array_like, optional
        Stationary flux vector. Must be stationary with respect to
        `transition_kernel`. If not provided, the stationary flux will be
        computed during initialization.
    states : sequence of MilestoneState, optional
        Milestone state labels. Values must be unique and consistent with
        the elements of `transition_kernel`. (A jump from milestone state 
        ``a`` to milestone state ``b`` can occur only if their 
        intersection ``a & b`` is nonempty.) Default is
        ``[MilestoneState(i, i+1) for i in range(M)]``, in which case
        `transition_kernel` must be tridiagonal.
    estimator : MarkovianMilestoningEstimator, optional
        The estimator that produced the model. Default is None. 

    """

    def __init__(self, transition_kernel, mean_lifetimes, stationary_flux=None,
                 states=None, estimator=None):
        Q = ctmc.rate_matrix(transition_kernel, np.reciprocal(mean_lifetimes))

        if stationary_flux is None:
            pi = None
        else:
            pi = np.multiply(stationary_flux, mean_lifetimes)

        if states is None:
            states = [MilestoneState(i, i+1) for i in range(len(Q))]
        elif any(type(a) != MilestoneState for a in states):
            raise TypeError('states must be of type MilestoneState')

        for i, j in np.argwhere(transition_kernel):
            a, b = states[i], states[j]
            if not (a & b):
                raise ValueError(f'cannot jump from {a} to {b}')

        super().__init__(Q, stationary_distribution=pi, states=states)
        self.estimator = estimator

    @property
    def transition_kernel(self):
        """(M, M) ndarray: Alias self.jump_matrix."""
        return self.jump_matrix

    @property
    def mean_lifetimes(self):
        """(M,) ndarray: Reciprocal of self.jump_rates."""
        return 1. / self.jump_rates

    @property
    def stationary_flux(self):
        """(M,) ndarray: Stationary flux vector."""
        return self.stationary_probability * self.jump_rates

    @property
    def stationary_probability(self):
        """(M,) ndarray: Alias self.stationary_distribution."""
        return self.stationary_distribution

    @property
    def estimator(self):
        """MarkovianMilestoningEstimator: Estimator that produced self."""
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        if value != None:
            cls = MarkovianMilestoningEstimator
            if not isinstance(value, cls):
                raise TypeError(f'estimator must be of type {cls.__name__}')
        self._estimator = value 
            

class MarkovianMilestoningEstimator:
    """Maximum likelihood and Bayesian estimation of Markovian 
    milestoning models.
        
    Parameters
    ----------
    reversible : bool, default True
        If True, enforce detailed balance. In this case estimation will 
        be performed on the maximal *strongly* connected set of milestone
        states.
    observation_interval : positive float, default 1.0
        The time resolution of the data used for fitting, that is, the 
        time interval between analyzed trajectory frames. This interval
        represents a lower bound on observed first passage times.

    See Also
    --------
    MarkovianMilestoningModel
        Class of models generated by this estimator.

    Notes
    -----
    Detailed balance means that the transition kernel :math:`K` of the
    milestoning model satisfies the condition 
    :math:`q_a K_{ab} = q_b K_{ba}`, where :math:`q` is the stationary
    flux vector.

    Estimation of the transition kernel (which is the transition matrix 
    of an embedded discrete-time Markov chain) is performed using the 
    methods discussed by Trendelkamp-Schroer et al. [1]_ and implemented 
    in the :mod:`msmtools.estimation` module.

    References
    ----------
    .. [1] B. Trendelkamp-Schroer, H. Wu, F. Paul, and F. Noe. Estimation
        and uncertainty of reversible Markov models. J. Chem. Phys. 
        143, 174101 (2015).
    
    """

    def __init__(self, reversible=True, observation_interval=1.):
        self.reversible = reversible
        self.observation_interval = observation_interval

    @property
    def reversible(self):
        """bool: Whether to enforce detailed balance."""
        return self._reversible

    @reversible.setter
    def reversible(self, value):
        self._reversible = bool(value)
 
    @property
    def observation_interval(self):
        """float: Time resolution of the data."""
        return self._observation_interval

    @observation_interval.setter
    def observation_interval(self, value):
        if value <= 0:
            raise ValueError('observation interval must be positive')
        self._observation_interval = value

    def fit(self, schedules):
        """Fit the estimator to milestone schedule data.

        Parameters
        ----------
        schedules : iterable of Sequence[tuple[MilestoneState, float]]
            Sequences of (milestone state, lifetime) pairs.

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        Notes
        -----
        For users with data in the form of individual 
        milestone-to-milestone first-passage times, a first-passage event 
        from milestone ``a`` to milestone ``b`` after a time ``t`` can be 
        represented by a schedule ``((a, t), (b, 0))``.
        
        Transitions to or from milestones bordering cells labeled None 
        will be ignored.

        """
        # Build a mapping from ordered pairs of milestone states to lists
        # of first passage times.
        # TODO(Jeff): Separate input validation from processing logic.
        first_passage_times = collections.defaultdict(list)
        for schedule in schedules:
            a, t = schedule[0]
            if type(a) != MilestoneState:
                raise TypeError('states must be of type MilestoneState')
            for b, s in schedule[1:]:
                if type(b) != MilestoneState:
                    raise TypeError('states must be of type MilestoneState')
                if t <= 0:
                    msg = 'nonterminal milestone lifetimes must be positive'
                    raise ValueError(msg)
                if None not in a and None not in b:
                    first_passage_times[a, b].append(t)
                a, t = b, s
            if t < 0:
                msg = 'terminal milestone lifetime must be nonnegative'
                raise ValueError(msg)

        states = ({a for a, _ in first_passage_times} 
                  | {b for _, b in first_passage_times})
        states = sorted(states, key=lambda a: sorted(a))
        ix = {a: i for i, a in enumerate(states)}
        n_states = len(states)

        count_matrix = np.zeros((n_states, n_states), dtype=int)
        total_times = np.zeros(n_states)
        for (a, b), times in first_passage_times.items():
            count_matrix[ix[a], ix[b]] = len(times)
            total_times[ix[a]] += sum(times)
        
        self.first_passage_times_ = first_passage_times
        self.states_ = states
        self.count_matrix_ = count_matrix
        self.total_times_ = total_times

        return self

    def connected_sets(self):
        """Compute the connected sets of states.

        If :attr:`self.reversible` is True, the connected sets are the 
        strongly connected components of the directed graph with adjacency 
        matrix :attr:`self.count_matrix_`. Otherwise, they are the weakly 
        connected components.  

        Returns
        -------
        connected_sets : list of ndarray(dtype=int)
            Arrays of zero-based state indices.

        See Also
        --------
        :func:`msmtools.estimation.connected_sets`
            Low-level function used to compute connected sets.
 
        """
        return estimation.connected_sets(
            self.count_matrix_, directed=(True if self.reversible else False))

    def max_likelihood_estimate(self):
        r"""Return the maximum likelihood estimate.

        Returns
        -------
        MarkovianMilestoningModel
            The model that maximizes the likelihood of the data.

        See Also
        --------
        :func:`msmtools.estimation.transition_matrix` :
            Low-level function used to estimate the transition kernel.

        Notes
        -----
        The transition kernel is estimated from the observed transition 
        count matrix :math:`N` by maximizing the likelihood

        .. math:: \mathbb{P}(N|K)\propto\prod_{a,b}K_{ab}^{N_{ab}}.

        In the nonreversible case, this gives the estimate 
        :math:`\hat{K}_{ab}=N_{ab}/N_a`, where :math:`N_a=\sum_{b}N_{ab}` 
        is the total number of transitions starting from milestone 
        :math:`a`. In the reversible case, the maximization is subject to
        the constraint of detailed balance. For details see Section III 
        of Trendelkamp-Schroer et al. [1]_ 

        The mean lifetime of milestone :math:`a` is estimated by
        :math:`\hat{\tau}_a=T_a/N_a`, where :math:`T_a` is the total time 
        spent in milestone state :math:`a`.

        """
        lcc = self.connected_sets()[0]  # largest connected component
        states = [self.states_[i] for i in lcc]
        count_matrix = self.count_matrix_[lcc, :][:, lcc]
        total_times = self.total_times_[lcc]

        t = total_times / count_matrix.sum(axis=1)  # mean lifetimes

        # Estimate transition kernel, and return MLE model.
        # -- Reversible case
        if self.reversible:
            K, q = estimation.transition_matrix(count_matrix, reversible=True,
                                                return_statdist=True)
            np.fill_diagonal(K, 0)
            return MarkovianMilestoningModel(K, t, stationary_flux=q, 
                                             states=states, estimator=self)
        # -- Nonreversible case
        K = estimation.transition_matrix(count_matrix, reversible=False)
        np.fill_diagonal(K, 0)
        return MarkovianMilestoningModel(K, t, states=states, estimator=self)

    def posterior_sample(self, size=100):
        r"""Generate a sample from the posterior distribution.

        Parameters
        ----------
        size : int, optional
            The sample size, i.e., the number of models to generate.

        Returns
        -------
        Collection[MarkovianMilestoningModel]
            The sampled models.

        See Also
        --------
        :func:`msmtools.estimation.tmatrix_sampler` :
            Low-level function used to sample transition kernels.

        Notes
        -----
        Transition kernels are sampled from the posterior distribution

        .. math:: \mathbb{P}(K|N) \propto \mathbb{P}(K)
                                          \prod_{a,b} K_{ab}^{N_{ab}},

        where the prior :math:`\mathbb{P}(K)` depends on whether detailed
        balance is assumed. For details see Section IV of
        Trendelkamp-Schroer et al. [1]_ Sampling is initiated from the
        maximum likelihood estimate of :math:`K`.

        The mean lifetime of milestone :math:`a` is sampled from an 
        inverse Gamma distribution with shape :math:`N_a` and scale
        :math:`T_a`.

        """
        lcc = self.connected_sets()[0]  # largest connected component
        states = [self.states_[i] for i in lcc]
        count_matrix = self.count_matrix_[lcc, :][:, lcc]
        total_times = self.total_times_[lcc]
        total_counts = count_matrix.sum(axis=1)

        # Sample jump rates (inverse mean lifetimes).
        rng = np.random.default_rng()
        vs = np.zeros((size, len(states)))
        for i, (n, r) in enumerate(zip(total_counts, total_times)):
            vs[:, i] = rng.gamma(n, scale=1/r, size=size)
        
        # Initialize transition matrix sampler.
        K_mle = estimation.transition_matrix(
            count_matrix, reversible=self.reversible)
        sampler = estimation.tmatrix_sampler(
            count_matrix, reversible=self.reversible, T0=K_mle)

        # Sample transition kernels, and return sampled models.
        # -- Reversible case
        if self.reversible:
            Ks, qs = sampler.sample(nsamples=size, return_statdist=True)
            for K in Ks:
                np.fill_diagonal(K, 0)
            return [MarkovianMilestoningModel(K, 1/v, stationary_flux=q,
                                              states=states, estimator=self)
                    for K, v, q in zip(Ks, vs, qs)] 
        # -- Nonreversible case
        Ks = sampler.sample(nsamples=size)
        for K in Ks:
            np.fill_diagonal(K, 0)
        return [MarkovianMilestoningModel(K, 1/v, 
                                          states=states, estimator=self) 
                for K, v in zip(Ks, vs)]


class TrajectoryColoring:
    """Mapping of continuous trajectories to milestone schedules.
        
    Parameters
    ----------
    anchors : (N, d) array_like
        Generating points for Voronoi tessellation of `d`-dimensional
        state space.
    boxsize : (d,) array_like or scalar, optional
        Apply `d`-dimensional toroidal topology (periodic boundary 
        conditions).
    parent_cell : (N,) array_like of int, optional
        The cell index associated with each anchor. Can be used to 
        define a Voronoi diagram with sites that are sets of anchors
        rather than single anchors. By default, there are `N` cells,
        one for each anchor.
    forward : bool, optional
        If True, track the next milestone hit (forward commitment),
        rather than the last milestone hit (backward commitment). Default
        is False.
    cutoff : positive float, optional
        Maximum distance to the nearest anchor. The region of state space
        outside the cutoff is treated as a cell labeled None. This is 
        primarily an ad hoc device.

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
            value = np.append(value, None) 
        self._parent_cell = value

    @property
    def forward(self):
        """bool: Whether to map to a forward milestoning process."""
        return self._forward

    @forward.setter
    def forward(self, value):
        self._forward = bool(value)

    def transform(self, traj):
        """Map a trajectory to its milestone schedule.

        Parameters
        ----------
        traj : sequence of (d,) array_like
            A trajectory in `d`-dimensional space.

        Returns
        -------
        schedule : Sequence[tuple[MilestoneState, int]]
            A sequence of (milestone state, lifetime) pairs.

        """
        dtraj = self._assign_cells(traj)
        return color_discrete_trajectory(dtraj, forward=self.forward)

    def __call__(self, traj):
        """Alias self.transform(traj)."""
        return self.transform(traj)
 
    def _assign_cells(self, x):
        _, k = self._kdtree.query(x, distance_upper_bound=self.cutoff)
        return self._parent_cell[k]
 

def color_discrete_trajectory(dtraj, forward=False):
    """Map a discrete-state trajectory to a milestone schedule.

    Parameters
    ----------
    dtraj : sequence
        A discrete-state trajectory, e.g., a sequence of cell indices. 
        Values must be hashable. The value None is reserved to indicate
        an undefined state.
    forward : bool, optional
        If True, track the next milestone hit (forward commitment),
        rather than the last milestone hit (backward commitment). 
        Default is False.

    Returns
    -------
    schedule : Sequence[tuple[MilestoneState, int]]
        A sequence of (milestone state, lifetime) pairs. 

    Notes
    -----
    When `forward` is False (ordinary milestoning), the initial milestone
    state is set to ``MilestoneState(None, dtraj[0])``. When `forward` is
    True, the final milestone state is set to 
    ``MilestoneState(dtraj[-1], None)``.
 
    """
    dtraj_it = reversed(dtraj) if forward else iter(dtraj)
    i = next(dtraj_it)
    states = [MilestoneState(None, i)]
    lifetimes = [0]
    for j in dtraj_it:
        lifetimes[-1] += 1
        if j not in states[-1]:
            states.append(MilestoneState(i, j))
            lifetimes.append(0)
        i = j
    if forward:
        return tuple(zip(reversed(states), reversed(lifetimes))) 
    return tuple(zip(states, lifetimes))
 

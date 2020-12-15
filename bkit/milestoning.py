import collections
import msmtools.estimation as estimation
import msmtools.util.types
import numpy as np
import scipy.spatial
from bkit.markov import ContinuousTimeMarkovChain


class Milestone(frozenset):
    """A milestone indexed by a set of cells."""

    @property
    def cells(self):
        """Cells associated with the milestone."""
        return set(self)


class MarkovianMilestoningModel(ContinuousTimeMarkovChain):
    """Milestoning process governed by a continuous-time Markov chain."""

    def __init__(self, transition_kernel, mean_lifetimes, milestones=None):
        """Create a new MarkovianMilestoningModel.

        Parameters
        ----------
        transition_kernel : (M, M) array_like
            Transition probability kernel. Must be a row stochastic 
            matrix with all diagonal elements equal to zero.

        mean_lifetimes : (M,) array_like
            Average milestone lifetimes, positive (>0).

        milestones : (M,) array_like, optional
            Milestone labels, assumed to be hashable. Will default to 
            np.arange(M) if not provided.            
           
        """
        super().__init__(transition_kernel, 1/mean_lifetimes, milestones)

    @property
    def milestones(self):
        """Milestone labels in indexed order."""
        return self.states

    @property
    def n_milestones(self):
        """Number of milestones."""
        return self.n_states

    @property
    def milestone_to_index(self):
        """A dictionary mapping each milestone label to its index."""
        return self.state_to_index

    @property
    def transition_kernel(self):
        """Transition probability kernel."""
        return self.embedded_tmatrix

    @property
    def mean_lifetimes(self):
        """Mean lifetime associated with each milestone.""" 
        return 1 / self.jump_rates

    @property
    def stationary_flux(self):
        """Stationary flux distribution, normalized to 1."""
        q = self.stationary_distribution * self.jump_rates
        return q / q.sum()

    @property
    def stationary_population(self):
        """Stationary population distribution, normalized to 1."""
        return self.stationary_distribution


class MarkovianMilestoningEstimator:
    """Estimator for Markovian milestoning models."""

    def __init__(self, reversible=True, dt=1):
        """Estimator for Markovian milestoning models.

        Parameters
        ----------
        reversible : bool, optional
            If True, restrict the ensemble of transition matrices
            to those satisfying detailed balance.

        dt : float, optional
            Observation interval (time resolution) of MD data.
            
        """
        super().__init__()
        self._reversible = reversible
        self._dt = dt
        self._model = None

    @property
    def reversible(self):
        """If True, perform reversible estimation."""
        return self._reversible

    @property
    def dt(self):
        """Observation interval (time resolution) of MD data."""
        return self._dt

    @property
    def maximum_likelihood_model(self):
        """The maximum likelihood MarkovianMilestoningModel."""
        return self._model

    def fit(self, data):
        """Fit estimator to coarse-grained trajectory data.

        Parameters
        ----------
        data : list of lists of tuples, dict
            Milestone schedules, i.e., lists of (milestone, lifetime) pairs,
            or a mapping from ordered pairs of milestones to lists of
            lag times. Times are assumed to be in units of `self.dt`.

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        """
        if type(data) is list:
            return self.fit_from_schedules(data)
        return self.fit_from_lagtimes(data)

    def fit_from_schedules(self, schedules):
        """Fit estimator to milestone schedule data.

        Parameters
        ----------
        schedules : list of lists of tuples
            Sequences of (milestone, lifetime) pairs obtained by
            trajectory decomposition. Lifetimes are assumed 
            to be in units of `self.dt`. Transitions to or from 
            milestones associated with unassigned cells (index -1) 
            are ignored.

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        """ 
        lagtimes = collections.defaultdict(list)
        for schedule in schedules:
            a, t = schedule[0]
            for b, s in schedule[1:]:
                if -1 not in a and -1 not in b:
                    lagtimes[a, b].append(t)
                a, t = b, s
        return self.fit_from_lagtimes(lagtimes)

    def fit_from_lagtimes(self, lagtimes):
        """Fit estimator to lagtime data.

        Parameters
        ----------
        lagtimes : dict
            Mapping from ordered pairs of milestones to lag times:
            `lagtimes[a, b]` is a list of lag times for transitions 
            from source milestone `a` to target milestone `b`. 
            Times are assumed to be in units of `self.dt`.

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        """
        milestones = sorted(
            ({a for a, b in lagtimes} | {b for a, b in lagtimes}), 
            key=lambda a: sorted(a))
        ix = {a: i for i, a in enumerate(milestones)}
        m = len(milestones)

        count_matrix = np.zeros((m, m), dtype=int)
        total_times = np.zeros(m)
        for (a, b), times in lagtimes.items():
            count_matrix[ix[a], ix[b]] = len(times)
            total_times[ix[a]] += sum(times)
        total_counts = count_matrix.sum(axis=1)
        
        total_times *= self.dt  # should this be done here?
 
        self._count_matrix = count_matrix
        self._total_times = total_times
        self._total_counts = total_counts

        K = estimation.transition_matrix(
            count_matrix, reversible=self.reversible)
        np.fill_diagonal(K, 0)
        t = total_times / total_counts
        self._model = MarkovianMilestoningModel(K, t, milestones)

        return self

    def sample_posterior(self, n_samples=100):
        """Sample models from the posterior distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw.

        Returns
        -------
        samples : list of MarkovianMilestoningModels
            Sampled models, or `None` if the estimator has not been fit.

        """
        if self._model is None:
            return None

        sampler = estimation.tmatrix_sampler(
            self._count_matrix, reversible=self.reversible,
            T0=self._model.transition_kernel)
        Ks = sampler.sample(nsamples=n_samples)
        for K in Ks:
            np.fill_diagonal(K, 0)

        rng = np.random.default_rng()
        vs = np.zeros((n_samples, self._model.n_states))
        for i, (n, r) in enumerate(zip(self._total_counts, self._total_times)):
            vs[:, i] = rng.gamma(n, scale=1/r, size=n_samples)

        return [MarkovianMilestoningModel(K, 1/v, self._model.milestones) 
                for K, v in zip(Ks, vs)]


class CoarseGraining:
    """Mapping from space-continuous dynamics to milestoning dynamics."""

    def __init__(self, anchors, boxsize=None, cutoff=np.inf, forward=False):
        """Mapping from space-continuous dynamics to milestoning dynamics.

        Parameters
        ----------
        anchors : ndarray (N, d) or list of ndarray (N_i, d)
            Generating points for Voronoi tessellation. 
            If a list of ndarrays is given, each subset of anchors
            indicates a union of Voronoi cells that should be treated 
            as a single cell.

        boxsize : array_like or scalar, optional
            Apply d-dimensional toroidal topology (periodic boundaries).

        cutoff : positive float, optional
            Maximum distance to nearest anchor. The region of space 
            beyond the cutoff is treated as a cell with index -1.

        forward : bool, optional
            If true, track the next milestone hit (forward commitment),
            rather than the last milestone hit (backward commitment).

        """
        self._anchors = anchors
        self._boxsize = boxsize
        self._cutoff = cutoff
        self._forward = forward

        if type(anchors) is np.ndarray:
            parent_cell = list(range(len(anchors)))
        else:
            parent_cell = [i for i, a in enumerate(anchors) for x in a]
            anchors = np.concatenate(anchors)
        if np.isfinite(cutoff):
            parent_cell.append(-1)
        self._parent_cell = np.asarray(parent_cell, dtype=int)
        self._kdtree = scipy.spatial.cKDTree(anchors, boxsize=boxsize)    

    @property
    def anchors(self):
        """Sites of the Voronoi diagram."""
        return self._anchors

    @property
    def boxsize(self):
        """Periodic dimensions."""
        return self._boxsize

    @property
    def cutoff(self):
        """Maximum distance to nearest anchor."""
        return self._cutoff

    @property
    def forward(self):
        """Whether to map to a forward milestoning process."""
        return self._forward

    def transform(self, trajs):
        """Map space-continuous dynamics to milestoning dynamics.

        Parameters
        ----------
        trajs : ndarray (T, d) or list of ndarray (T_i, d)
            Trajectories to be coarse grained.

        Returns
        -------
        schedules : list of lists of tuples
            Sequences of (milestone, lifetime) pairs obtained by
            coarse graining. Milestones are unordered pairs of integers
            (cell indices). Lifetimes are positive integers.

        """
        trajs = msmtools.util.types.ensure_traj_list(trajs)
        return [self._traj_to_milestone_schedule(traj) for traj in trajs]

    def __call__(self, trajs):
        return self.transform(trajs)

    def _traj_to_milestone_schedule(self, traj):
        dtraj = self._assign_cells(traj)
        return dtraj_to_milestone_schedule(dtraj, forward=self.forward)
    
    def _assign_cells(self, x):
        _, k = self._kdtree.query(x, distance_upper_bound=self._cutoff)
        return self._parent_cell[k]
 

def dtraj_to_milestone_schedule(dtraj, forward=False):
    """Map cell-based dynamics to milestoning dynamics.

    Parameters
    ----------
    dtraj : ndarray(T, dtype=int)
        A discrete trajectory, i.e., a sequence of cell indices.

    forward : bool, optional
        If true, track the next milestone hit (forward commitment),
        rather than the last milestone hit (backward commitment).

    Returns
    -------
    schedule : list of tuples
        Sequence of (milestone, lifetime) pairs. Milestones are 
        unordered pairs of integers (cell indices). Lifetimes are 
        positive integers. For ordinary milestoning, the initial 
        milestone is set to `{-1, dtraj[0]}`. For forward milestoning, 
        the final milestone is set to `{dtraj[-1], -1}`.
 
    """
    dtraj = msmtools.util.types.ensure_dtraj(dtraj)
    dtraj_it = reversed(dtraj) if forward else iter(dtraj)
    i = next(dtraj_it)
    milestones = [Milestone({-1, i})]
    lifetimes = [0]
    for j in dtraj_it:
        lifetimes[-1] += 1
        if j not in milestones[-1]:
            milestones.append(Milestone({i, j}))
            lifetimes.append(0)
        i = j
    if forward:
        return list(zip(reversed(milestones), reversed(lifetimes))) 
    return list(zip(milestones, lifetimes))
 

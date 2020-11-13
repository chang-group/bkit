import bkit.markov
import deeptime.base
import deeptime.markov.msm
import deeptime.markov.tools.estimation as estimation
import numpy as np
import scipy.spatial

BayesianPosterior = deeptime.markov.msm.BayesianPosterior


class MarkovianMilestoningModel(bkit.markov.ContinuousTimeMarkovModel):
    """Milestoning process governed by a continuous-time Markov chain."""

    def __init__(self, rate_matrix, milestones):
        """Milestoning model with given rate matrix and set of milestones.

        Parameters
        ----------
        rate_matrix : (M, M) ndarray
            Transition rate matrix, row infinitesimal stochastic.

        milestones : array_like
            Ordered set of milestones indexed by the states of the 
            underlying Markov model.
           
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
        self._milestones = value

    @property
    def transition_kernel(self):
        """Transition probability kernel of the embedded Markov chain."""
        return self.embedded_markov_model.transition_matrix

    @property
    def mean_lifetimes(self):
        """The mean lifetime associated with each milestone.""" 
        return 1 / self.jump_rates

    @property
    def stationary_fluxes(self):
        """The stationary flux vector."""
        return self.embedded_markov_model.stationary_distribution


class MarkovianMilestoningEstimator(deeptime.base.Estimator):
    """Estimator for Markovian milestoning models."""

    def __init__(self, reversible=True, n_samples=None):
        """Estimator for Markovian milestoning models.

        Parameters
        ----------
        reversible : bool
            If True, restrict the ensemble of transition probability
            kernels to those satisfying detailed balance.

        n_samples : int
            Number of samples to draw from the posterior distribution. 
            If `None`, compute only a maximum likelihood estimate.
            
        """
        super().__init__()
        self.reversible = reversible
        self.n_samples = n_samples

    def fit(self, data):
        return self.fit_from_schedules(data)

    def fit_from_discrete_timeseries(self, timeseries):
        pass

    def fit_from_schedules(self, schedules):
        """Estimate model from data in the form of milestone schedules.

        Parameters
        ----------
        schedules : list of lists of tuples
            Lists of (milestone, lifetime) pairs obtained by 
            trajectory decomposition.

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        """
        milestones = sorted({a for schedule in schedules for a, t in schedule
                             if None not in a}, key=lambda a: sorted(a))
        ix = {a: i for i, a in enumerate(milestones)}
        m = len(milestones)

        count_matrix = np.zeros((m, m))
        total_times = np.zeros(m)
        for schedule in schedules:
            for (a, t), (b, _) in zip(schedule[:-1], schedule[1:]):
                if a not in ix or b not in ix:
                    continue
                count_matrix[ix[a], ix[b]] += 1
                total_times[ix[a]] += t
        total_counts = np.sum(count_matrix, axis=1)   

        # Maximum likelihood estimation
        if not n_samples:
            K = estimation.transition_matrix(count_matrix, 
                                             reversible=self.reversible)
            t = total_times / total_counts
            Q = K / t[:, np.newaxis] - np.diag(1/t)
            self._model = MarkovianMilestoningModel(Q, milestones) 
            return self

        # Sampling from posterior distribution 
        Ks = estimation.tmatrix_sampler(count_matrix, 
            reversible=self.reversible).sample(nsamples=self.n_samples)
        vs = np.zeros((self.n_samples, m)) # v = 1/t = vector of jump rates
        for i, (n, r) in enumerate(zip(total_counts, total_times)):
            rng = np.random.default_rng()
            vs[:, i] = rng.gamma(n, scale=1/r, size=self.n_samples)
        Qs = [K * v[:, np.newaxis] - np.diag(v) for K, v in zip(Ks, vs)]
        samples = [MarkovianMilestoningModel(Q, milestones) for Q in Qs]
        self._model = BayesianPosterior(samples=samples)
        return self


class CoarseGrainer(deeptime.base.Transformer):
    """Mapping from trajectories to schedules of a milestoning process."""

    def __init__(self, anchors, boxsize=None, cutoff=np.inf):
        """Initialize trajectory decomposer.

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
            beyond the cutoff is treated as a cell labeled `None`.

        """
        if type(anchors) is np.ndarray:
            self._parent_cell = list(range(len(anchors)))
        else:
            self._parent_cell = [i for i, arr in enumerate(anchors) 
                                   for k in range(len(arr))]
            anchors = np.concatenate(anchors)
        self._kdtree = scipy.spatial.cKDTree(anchors, boxsize=boxsize)    
        self._cutoff = cutoff
        if np.isfinite(cutoff):
            self._parent_cell.append(None)
     
    def transform(self, trajs, dt=1, forward=False):
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
    if forward:
        dtraj = reversed(dtraj)
    milestones = [frozenset({None, dtraj[0]})]
    lifetimes = [0]
    for i, j in zip(dtraj[:-1], dtraj[1:]):
        lifetimes[-1] += dt
        if j not in milestones[-1]:
            milestones.append(frozenset({i, j}))
            lifetimes.append(0)
    schedule = list(zip(milestones, lifetimes))
    if forward:
        return reversed(schedule)
    return schedule


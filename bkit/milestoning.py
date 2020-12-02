import bkit.markov
import collections
import deeptime.base
import msmtools.estimation as estimation
import numpy as np
import scipy.spatial
from deeptime.markov.msm import BayesianPosterior


class MarkovianMilestoningModel(bkit.markov.ContinuousTimeMarkovChain):
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
    def index_by_milestone(self):
        """Dictionary mapping milestone labels to indices."""
        return self.index_by_state

    @property
    def transition_kernel(self):
        """Transition probability kernel."""
        return self.embedded_tmatrix

    @property
    def mean_lifetimes(self):
        """Mean lifetime associated with each milestone.""" 
        return 1/self.jump_rates

    @property
    def stationary_flux(self):
        """Stationary flux distribution, normalized to 1."""
        q = self.stationary_distribution * self.jump_rates
        return q / q.sum()

    def free_energy(self, kT=1):
        """Free energies of the milestone states in given units.
    
        Parameters
        ----------
        kT : float, optional
            Energy scale factor.

        Returns
        -------
        (M,) ndarray
            Free energies of the milestone states.

        """
        return -kT * np.log(self.stationary_distribution)


class MarkovianMilestoningEstimator(deeptime.base.Estimator):
    """Estimator for Markovian milestoning models."""

    def __init__(self, reversible=True, dt_obs=1):
        """Estimator for Markovian milestoning models.

        Parameters
        ----------
        reversible : bool, optional
            If True, restrict the ensemble of transition matrices
            to those satisfying detailed balance.

        dt_obs : float, optional
            Observation interval (time resolution) of MD data.
            
        """
        super().__init__()
        self._reversible = reversible
        self._dt_obs = dt_obs

    @property
    def reversible(self):
        """If True, perform reversible estimation."""
        return self._reversible

    @property
    def dt_obs(self):
        """Observation interval."""
        return self._dt_obs

    def fetch_model(self):
        """Return the maximum likelihood model.

        Returns
        -------
        model : MarkovianMilestoningModel
            Model obtained by maximum likelihood estimation.
        
        """
        return self._model

    def fit(self, data):
        """Fit maximum likelihood model to coarse-grained trajectory data.

        Parameters
        ----------
        data : list of tuples, list of lists of tuples, or dict
            Milestone schedules, i.e., lists of (milestone, lifetime) pairs,
            or a mapping from ordered pairs of milestones to lists of
            lag times. Times are assumed to be in units of `self.dt_obs`.

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        """
        if type(data) is list:
            return self.fit_from_schedules(data)
        return self.fit_from_lagtimes(data)

    def fit_from_schedules(self, schedules):
        """Fit maximum likelihood model to milestone schedule data.

        Parameters
        ----------
        schedules : list of tuples or list of lists of tuples
            Sequences of (milestone, lifetime) pairs obtained by
            trajectory decomposition. Milestones are assumed to be 
            `frozenset`s of cell indices; lifetimes are assumed to 
            be in units of `self.dt_obs`. Transitions to or from 
            milestones with unassigned cells (index -1) are ignored.

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        """ 
        if type(schedules[0]) is tuple:
            schedules = [schedules]
        lagtimes = collections.defaultdict(list)
        for schedule in schedules:
            a, t = schedule[0]
            for b, s in schedule[1:]:
                if -1 not in a and -1 not in b:
                    lagtimes[a, b].append(t)
                a, t = b, s
        return self.fit_from_lagtimes(lagtimes)

    def fit_from_lagtimes(self, lagtimes):
        """Fit maximum likelihood model to lagtime data.

        Parameters
        ----------
        lagtimes : dict
            Map from ordered pairs of milestones to lists of lag times:
            `lagtimes[a, b]` are the lag times for transitions from 
            source milestone `a` to target milestone `b`. Times are 
            assumed to be in units of `self.dt_obs`.

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        """
        milestones = sorted(({a for a, b in lagtimes} 
            | {b for a, b in lagtimes}), key=lambda a: sorted(a))
        ix = {a: i for i, a in enumerate(milestones)}
        m = len(milestones)

        count_matrix = np.zeros((m, m), dtype=int)
        total_times = np.zeros(m)
        for (a, b), ts in lagtimes.items():
            count_matrix[ix[a], ix[b]] = len(ts)
            total_times[ix[a]] += sum(ts)
        total_times *= self.dt_obs

        K = estimation.transition_matrix(count_matrix, 
                                         reversible=self.reversible)
        t = total_times / count_matrix.sum(axis=1)

        self._model = MarkovianMilestoningModel(K, t, milestones) 
        self._lagtimes = lagtimes
        self._count_matrix = count_matrix
        self._total_times = total_times

        return self

    def sample_posterior(self, n_samples=100):
        """Sample models from the posterior distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw.

        Returns
        -------
        posterior : BayesianPosterior
            Object containing sampled models (or `None` if the estimator
            has not yet been fit).

        """
        if not self.has_model:
            return None
        Ks = estimation.tmatrix_sampler(self._count_matrix, 
            reversible=self.reversible).sample(nsamples=n_samples)
        vs = np.zeros((n_samples, self._model.n_states))
        for i, (n, r) in enumerate(zip(self._count_matrix.sum(axis=1), 
                                       self._total_times)):
            rng = np.random.default_rng()
            vs[:, i] = rng.gamma(n, scale=1/r, size=n_samples)
        samples = [MarkovianMilestoningModel(K, 1/v, milestones) 
                   for K, v in zip(Ks, vs)]
        return BayesianPosterior(samples=samples)


class CoarseGrainer(deeptime.base.Transformer):
    """Mapping from space-continuous dynamics to milestoning dynamics."""

    def __init__(self, anchors, boxsize=None, cutoff=np.inf):
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

        """
        self._anchors = anchors
        self._boxsize = boxsize
        self._cutoff = cutoff

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

    def transform(self, trajs, forward=False):
        """Map space-continuous dynamics to milestoning dynamics.

        Parameters
        ----------
        trajs : ndarray (T, d) or list of ndarray (T_i, d)
            Trajectories to be decomposed.

        forward : bool, optional
            If true, track the next milestone hit (forward commitment),
            rather than the last milestone hit (backward commitment).

        Returns
        -------
        schedules : list of tuples or list of lists of tuples
            Sequences of (milestone, lifetime) pairs obtained by
            trajectory decomposition. Milestones are `frozenset`s
            of cell indices. Lifetimes are in units of the sampling 
            interval of the trajectory data.

        """ 
        if type(trajs) is np.ndarray:
            return self._traj_to_milestone_schedule(trajs, forward)
        return [self._traj_to_milestone_schedule(traj, forward)
                for traj in trajs]

    def _traj_to_milestone_schedule(self, traj, forward=False):
        dtraj = self._assign_cells(traj)
        return dtraj_to_milestone_schedule(dtraj, forward)
    
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
        unordered pairs of cell indices. For ordinary milestoning, 
        the first milestone is set to `{-1, dtraj[0]}`. For forward 
        milestoning, the last milestone is set to `{dtraj[-1], -1}`. 
        Lifetimes are in units of the time step of `dtraj`.
 
    """
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
 

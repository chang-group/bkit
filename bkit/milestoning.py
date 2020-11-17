import bkit.markov
import deeptime.base
import deeptime.markov.msm
from deeptime.markov.tools import estimation
import numpy as np
import scipy.spatial


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
        """Mean lifetime associated with each milestone.""" 
        return 1 / self.jump_rates

    @property
    def stationary_fluxes(self):
        """Stationary flux distribution, normalized to 1."""
        return self.embedded_markov_model.stationary_distribution

    def _free_energy(self, kT=1):
        return -kT * np.log(self.stationary_distribution)


class MarkovianMilestoningEstimator(deeptime.base.Estimator):
    """Estimator for Markovian milestoning models."""

    def __init__(self, dt=1, reversible=True, n_samples=None):
        """Estimator for Markovian milestoning models.

        Parameters
        ----------
        dt : float, optional
            Trajectory sampling interval, positive (>0).

        reversible : bool, optional
            If True, restrict the ensemble of transition probability
            kernels to those satisfying detailed balance.

        n_samples : int, optional
            Number of samples to draw from the posterior distribution. 
            If `None`, compute only a maximum likelihood estimate.
            
        """
        super().__init__()
        self.dt = dt
        self.reversible = reversible
        self.n_samples = n_samples

    def fit(self, schedules):
        """Estimate model from realizations of a milestoning process.

        Parameters
        ----------
        schedules : list of tuples or list of lists of tuples
            Sequences of (milestone, lifetime) pairs obtained by
            trajectory decomposition. Milestones are pairs of 
            cell indices. Lifetimes are in units of `self.dt`.

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        """
        if type(schedules[0]) is tuple:
            schedules = [schedules]

        edges = set.union(*({a for a, t in schedule if -1 not in a} 
                            for schedule in schedules))
        milestones = sorted({tuple(sorted(a)) for a in edges}) 
        ix = {a: i for i, a in enumerate(milestones)}
        ix.update((tuple(reversed(a)), i) for i, a in enumerate(milestones))
        m = len(milestones)

        count_matrix = np.zeros((m, m), dtype=int)
        total_times = np.zeros(m)
        for schedule in schedules:
            a, t = schedule[0]
            for b, s in schedule[1:]:
                if a in edges and b in edges:
                    count_matrix[ix[a], ix[b]] += 1
                    total_times[ix[a]] += t
                a, t = b, s
        total_counts = np.sum(count_matrix, axis=1)
        total_times *= self.dt
                
        self.count_matrix_ = count_matrix
        self.total_times_ = total_times
        self.milestones_ = milestones

        # Maximum likelihood estimate
        if not self.n_samples:
            K = estimation.transition_matrix(count_matrix, 
                                             reversible=self.reversible)
            t = total_times / total_counts
            Q = K / t[:, np.newaxis] - np.diag(1/t)
            self._model = MarkovianMilestoningModel(Q, milestones) 
            return self

        # Sample from posterior distribution 
        Ks = estimation.tmatrix_sampler(count_matrix, 
            reversible=self.reversible).sample(nsamples=self.n_samples)
        vs = np.zeros((self.n_samples, m)) # v = 1/t = jump rates
        for i, (n, r) in enumerate(zip(total_counts, total_times)):
            rng = np.random.default_rng()
            vs[:, i] = rng.gamma(n, scale=1/r, size=self.n_samples)
        Qs = [K * v[:, np.newaxis] - np.diag(v) for K, v in zip(Ks, vs)]
        samples = [MarkovianMilestoningModel(Q, milestones) for Q in Qs]
        self._model = deeptime.markov.msm.BayesianPosterior(samples=samples)
        return self


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
            trajectory decomposition. Milestones are pairs (2-tuples) 
            of cell indices, with (i, j) and (j, i) identified as the
            same milestone. Lifetimes are in units of the sampling 
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
        pairs (2-tuples) of cell indices. For ordinary milestoning, 
        the first milestone is set to `(-1, dtraj[0])`. For forward 
        milestoning, the last milestone is set to `(dtraj[-1], -1)`. 
        Lifetimes are in units of the time step of `dtraj`.
 
    """
    dtraj_it = reversed(dtraj) if forward else iter(dtraj)
    i = next(dtraj_it)
    milestones = [(-1, i)]
    lifetimes = [0]
    for j in dtraj_it:
        lifetimes[-1] += 1
        if j not in milestones[-1]:
            milestones.append((i, j))
            lifetimes.append(0)
        i = j
    if forward:
        return list(zip((tuple(reversed(a)) for a in reversed(milestones)),
                        reversed(lifetimes))) 
    return list(zip(milestones, lifetimes))
 

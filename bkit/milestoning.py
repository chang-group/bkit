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

    def fit(self, mtrajs):
        """Estimate model from trajectories of a milestoning process.

        Parameters
        ----------
        mtrajs : ndarray (T, 2) or list of ndarray (T_i, 2), dtype=int
            Time series of milestone labels (pairs of cell indices).

        Returns
        -------
        self : MarkovianMilestoningEstimator
            Reference to self.

        """
        if type(mtrajs) is np.ndarray:
            mtrajs = [mtrajs]

        edges = {(i, j) for y in mtrajs for i, j in np.unique(y, axis=0)}
        milestones = sorted({tuple(sorted(e)) for e in edges if -1 not in e})
        m = len(milestones)
        
        ix = {a: i for i, a in enumerate(milestones)}
        ix.update((tuple(reversed(a)), i) for i, a in enumerate(milestones))

        count_matrix = np.zeros((m, m), dtype=int)
        total_times = np.zeros(m)
        for y in mtrajs:
            lag = 0
            for n in range(1, len(y)):
                lag += self.dt   
                if all(y[n] == y[n-1]):
                    continue
                a, b = tuple(y[n-1]), tuple(y[n])
                if a in ix and b in ix:
                    count_matrix[ix[a], ix[b]] += 1
                    total_times[ix[a]] += lag
                lag = 0
        total_counts = np.sum(count_matrix, axis=1)   

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
        vs = np.zeros((self.n_samples, m)) # v = 1/t = vector of jump rates
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
            self._parent_cell = list(range(len(anchors)))
        else:
            self._parent_cell = [i for i, arr in enumerate(anchors) 
                                   for k in range(len(arr))]
            anchors = np.concatenate(anchors)
        self._kdtree = scipy.spatial.cKDTree(anchors, boxsize=boxsize)    
        if np.isfinite(cutoff):
            self._parent_cell.append(-1)
    
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
        """Map original dynamics to milestoning index process.

        Parameters
        ----------
        trajs : ndarray (T, d) or list of ndarray (T_i, d)
            Trajectories to be decomposed.

        forward : bool, optional
            If true, track the next milestone hit (forward commitment),
            rather than the last milestone hit (backward commitment).
    
        Returns
        -------
        mtrajs : ndarray (T, 2) or list of ndarray (T_i, 2), dtype=int
            Time series of milestone labels (pairs of cell indices).

        """ 
        if type(trajs) is np.ndarray:
            return self._traj_to_mtraj(trajs, forward=forward)
        return [self._traj_to_mtraj(traj, forward=forward) for traj in trajs]

    def _traj_to_mtraj(self, traj, forward=False):
        return _dtraj_to_mtraj(self._traj_to_dtraj(traj), forward=forward)
    
    def _traj_to_dtraj(self, traj):
        _, ktraj = self._kdtree.query(traj, distance_upper_bound=self._cutoff)
        return np.fromiter((self._parent_cell[k] for k in ktraj), dtype=int)
 

def _dtraj_to_mtraj(dtraj, forward=False):
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
    mtraj : ndarray((T, 2), dtype=int)
        Sequence of milestone labels. For ordinary (backward) milestoning,
        the first milestone is set to `(-1, dtraj[0])`. For forward
        milestoning, the last milestone is set to `(dtraj[-1], -1)`.
    
    """
    mtraj = np.zeros((len(dtraj), 2), dtype=int)
    if forward:
        mtraj[-1] = dtraj[-1], -1
        start, stop, step = len(dtraj)-2, -1, -1
    else:
        mtraj[0] = -1, dtraj[0]
        start, stop, step = 1, len(dtraj), 1
    for n in range(start, stop, step):
        if dtraj[n] in mtraj[n-step]:
            mtraj[n] = mtraj[n-step]
        else:
            mtraj[n] = dtraj[n-step], dtraj[n]
    return mtraj


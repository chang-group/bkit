import numpy as np
import msmtools.analysis as msmana


class ContinuousTimeMarkovChain:
    """A continuous-time Markov chain (i.e., Markov jump process)."""

    def __init__(self, embedded_tmatrix, jump_rates, states=None):
        """Create a new ContinuousTimeMarkovChain.

        Parameters
        ----------
        embedded_tmatrix : (M, M) array_like
            Transition matrix of the embedded discrete-time Markov
            chain. Must be row stochastic with diagonal elements
            all equal to zero.

        jump_rates: (M,) array_like
            Exponential rate parameters, positive (>0).

        states : (M,) array_like, optional
            State labels, assumed to be hashable. Will default to 
            np.arange(M) if no labels are provided.
            
        """
        self.embedded_tmatrix = embedded_tmatrix
        self.jump_rates = jump_rates
        self.states = states

    @property
    def embedded_tmatrix(self):
        return self._embedded_tmatrix

    @embedded_tmatrix.setter
    def embedded_tmatrix(self, value):
        value = np.asarray(value)
        if not msmana.is_tmatrix(value):
            raise ValueError('matrix must be row stochastic')
        if np.count_nonzero(value.diagonal()):
            raise ValueError('diagonal elements must be equal to zero')
        self._embedded_tmatrix = value

    @property
    def jump_rates(self):
        return self._jump_rates

    @jump_rates.setter
    def jump_rates(self, value):
        if value.shape != (self.embedded_tmatrix.shape[0],):
            msg = 'number of jump rates must match number of states'
            raise ValueError(msg)
        if not (value > 0).all():
            raise ValueError('jump rates must be positive')
        self._jump_rates = value

    @property
    def rate_matrix(self):
        """Transition rate matrix (infinitesimal generator)."""
        Q = self.jump_rates[:, np.newaxis] * self.embedded_tmatrix
        Q[np.diag_indices(self.n_states)] = -self.jump_rates
        return Q

    @property
    def stationary_distribution(self): 
        """Stationary population distribution, normalized to 1."""
        p = (msmana.stationary_distribution(self.embedded_tmatrix) 
             / self.jump_rates)
        return p / p.sum()

    @property
    def is_reversible(self):
        """Whether the Markov chain is reversible."""
        return msmana.is_reversible(self.embedded_tmatrix)
    
    @property
    def states(self):
        """State labels."""
        return self._states

    @states.setter
    def states(self, value):
        if value is None:
            self._states = np.arange(self.embedded_tmatrix.shape[0])
            return
        value = np.asarray(value)
        if value.shape != (self.embedded_tmatrix.shape[0],):
            msg = 'number of labels must match number of states'
            raise ValueError(msg)
        self._states = value
        self._index_by_state = {x: i for i, x in enumerate(value)}

    @property
    def n_states(self):
        """Number of states."""
        return len(self.states)

    @property
    def index_by_state(self):
        """Dictionary mapping each state label to its index."""
        return self._index_by_state 

    def mfpt(self, target_indices):
        """Mean first passage times to a target set of states.

        Parameters
        ----------
        target_indices : int or array_like of int
            Indices of the target states.

        Returns
        -------
        (M,) ndarray
            Mean first passage time from each state to the target set.

        """
        is_source = np.ones(self.n_states, dtype=bool)
        is_source[target_indices] = False
        Q = self.rate_matrix[is_source, :][:, is_source]
        mfpt = np.zeros(self.n_states)
        mfpt[is_source] = np.linalg.solve(Q, -np.ones(len(Q)))
        return mfpt


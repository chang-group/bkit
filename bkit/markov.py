import numpy as np
import msmtools.analysis as msmana
import msmtools.flux


class ContinuousTimeMarkovChain:
    """A continuous-time Markov chain (i.e., Markov jump process).
        
    Parameters
    ----------
    embedded_tmatrix : array_like, shape (M, M)
        Transition matrix of the embedded discrete-time Markov chain. 
        Must be row stochastic with diagonal elements all equal to zero.

    jump_rates: array_like, shape (M,)
        Exponential rate parameters.

    states : iterable, optional
        State labels, assumed to be hashable. Will default to 
        ``range(M)`` if not provided.
 
    """

    def __init__(self, embedded_tmatrix, jump_rates, states=None):
        self.embedded_tmatrix = embedded_tmatrix
        self.jump_rates = jump_rates
        self.states = states

    @property
    def embedded_tmatrix(self):
        """ndarray: Transition matrix of the embedded chain."""
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
        """ndarray: Rate parameters."""
        return self._jump_rates

    @jump_rates.setter
    def jump_rates(self, value):
        value = np.asarray(value)
        if value.shape != (self.embedded_tmatrix.shape[0],):
            msg = 'number of jump rates must match number of states'
            raise ValueError(msg)
        if not (value > 0).all():
            raise ValueError('jump rates must be positive')
        self._jump_rates = value

    @property
    def rate_matrix(self):
        """ndarray: Transition rate matrix (infinitesimal generator)."""
        rate_matrix = self.jump_rates[:, np.newaxis] * self.embedded_tmatrix
        rate_matrix[np.diag_indices(self.n_states)] = -self.jump_rates
        return rate_matrix

    @property
    def stationary_distribution(self): 
        """ndarray: Stationary distribution, normalized to 1."""
        p = (msmana.stationary_distribution(self.embedded_tmatrix) 
             / self.jump_rates)
        return p / p.sum()

    @property
    def is_reversible(self):
        """bool: Whether the Markov chain is reversible."""
        return msmana.is_reversible(self.embedded_tmatrix)
    
    @property
    def states(self):
        """list: State labels."""
        return self._states

    @states.setter
    def states(self, value):
        if value is None:
            value = range(self.embedded_tmatrix.shape[0])
        value = list(value)
        if len(value) != self.embedded_tmatrix.shape[0]:
            msg = 'number of labels must match number of states'
            raise ValueError(msg)
        self._states = value
        self._state_to_index = {x: i for i, x in enumerate(value)}

    @property
    def n_states(self):
        """int: The number of states."""
        return len(self.states)

    @property
    def state_to_index(self):
        """dict: Mapping from state labels to integer indices."""
        return self._state_to_index 

    def mfpt(self, target):
        """Mean first passage time to a target set of states.

        Parameters
        ----------
        target : int or list of int
            Indices of the target states.

        Returns
        -------
        ndarray, shape (M,)
            Mean first passage time from each state to the target set.

        """
        is_source = np.ones(self.n_states, dtype=bool)
        is_source[target] = False
        Q = self.rate_matrix[is_source, :][:, is_source]
        mfpt = np.zeros(self.n_states)
        mfpt[is_source] = np.linalg.solve(Q, -np.ones(len(Q)))
        return mfpt

    def reactive_flux(self, source, target):
        """Reactive flux from transition path theory (TPT).

        Parameters
        ----------
        source : int or list of int
            Indices of the source states.

        target : int or list of int
            Indices of the target states.

        Returns
        -------
        msmtools.flux.ReactiveFlux
            An object describing the reactive flux. See MSMTools 
            documentation for full details.

        """
        return msmtools.flux.tpt(self.rate_matrix, source, target, 
                                 rate_matrix=True)


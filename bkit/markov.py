import numpy as np
import scipy.linalg as linalg
from deeptime.markov.tools import analysis


class DTMC:
    """Discrete-time Markov chain with given transition matrix."""

    def __init__(self, transition_matrix, timestep=1.):
        """Construct a new model from a transition matrix.

        Parameters
        ----------
        transition_matrix : (M, M) ndarray
            Right stochastic matrix of 1-step transition probabilities.
        timestep : positive float, optional
            Time corresponding to one step of the chain.

        """
        self.transition_matrix = transition_matrix
        self.timestep = timestep

    @property
    def transition_matrix(self):
        """Transition matrix."""
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, value):
        if not analysis.is_transition_matrix(value):
            raise ValueError('Invalid transition matrix.')
        self._transition_matrix = np.asarray(value)
    
    @property
    def timestep(self):
        """Time corresponding to one step of the chain."""
        return self._timestep

    @timestep.setter
    def timestep(self, value):
        if not value > 0:
            raise ValueError('Timestep must be positive.')
        self._timestep = value

    @property
    def stationary_distribution(self):
        """Stationary distribution (may not be unique)."""
        return analysis.stationary_distribution(self.transition_matrix)

    @property
    def nstates(self):
        """Number of states in the chain."""
        return len(self.transition_matrix)

    def mfpt(self, target):
        """Mean first passage times to a target set of states.

        Parameters
        ----------
        target : int or list of int
            Index or indices of the target states.

        Returns
        -------
        mfpt : (M,) ndarray
            Mean first passage time from each state to the target set.

        """
        return analysis.mfpt(self.transition_matrix, target, tau=self.timestep)


class CTMC:
    """Continuous-time Markov chain with given rate matrix."""

    def __init__(self, rate_matrix):
        """Construct a new model from a rate matrix.

        Parameters
        ----------
        rate_matrix : (M, M) ndarray
            Transition rate matrix. For j != i, rate_matrix[i, j] is
            the transition rate from state i to state j. Diagonal
            elements are defined such that each row sums to zero.

        """
        self.rate_matrix = rate_matrix

    @property
    def rate_matrix(self):
        """Transition rate matrix."""
        return self._rate_matrix

    @rate_matrix.setter
    def rate_matrix(self, value):
        if not analysis.is_rate_matrix(value):
            raise ValueError('Invalid rate matrix.')
        self._rate_matrix = np.asarray(value)

    @property
    def stationary_distribution(self):
        """Stationary distribution (may not be unique)."""
        basis = linalg.null_space(self.rate_matrix.T).T
        return basis[0] / basis[0].sum()

    @property
    def nstates(self):
        """Number of states in the chain."""
        return len(self.rate_matrix)

    @property
    def jump_chain(self):
        """Embedded discrete-time Markov chain."""
        P = -self.rate_matrix / np.diag(self.rate_matrix)[:, np.newaxis]
        np.fill_diagonal(P, 0.)
        return DTMC(P)

    def mfpt(self, target):
        """Mean first passage times to a target set of states.

        Parameters
        ----------
        target : int or list of int
            Index or indices of the target states.

        Returns
        -------
        mfpt : (M,) ndarray
            Mean first passage time from each state to the target set.

        """
        is_source = np.ones(self.nstates, dtype=bool)
        is_source[target] = False
        Q = self.rate_matrix[is_source, :][:, is_source]
        mfpt = np.zeros(self.nstates)
        mfpt[is_source] = linalg.solve(Q, -np.ones(len(Q)))
        return mfpt


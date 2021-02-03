"""Algorithms for analysis of continuous-time Markov chains."""

import numpy as np
import msmtools.analysis as msmana
import msmtools.flux


class ContinuousTimeMarkovChain:
    """A continuous-time Markov chain.
        
    Parameters
    ----------
    rate_matrix: (M, M) array_like
        Transition rate matrix, row infinitesimal stochastic.

    stationary_distribution : (M,) array_like, optional
        Stationary distribution. Must be invariant with respect to the
        given rate matrix. If not provided, the stationary distribution
        will be computed from the rate matrix.

    states : sequence, optional
        State labels. Values must be unique and hashable. Will default 
        to ``range(M)`` if not provided.
 
    """

    def __init__(self, rate_matrix, stationary_distribution=None, states=None):
        self.rate_matrix = rate_matrix
        self.stationary_distribution = stationary_distribution
        self.states = states

    @property
    def rate_matrix(self):
        """(M, M) ndarray: Transition rate matrix."""
        return self._rate_matrix

    @rate_matrix.setter
    def rate_matrix(self, value):
        value = np.asarray(value)
        if not msmana.is_rate_matrix(value):
          raise ValueError('matrix must be row infinitesimal stochastic')
        self._rate_matrix = value
    
    @property
    def stationary_distribution(self): 
        """(M,) ndarray: Stationary distribution."""
        return self._statdist

    @stationary_distribution.setter
    def stationary_distribution(self, value):
        if value is None:
            self._statdist = stationary_distribution(self.rate_matrix)
            return
        if not np.allclose(np.dot(value, self.rate_matrix), 
                           np.zeros_like(value)):
            msg = 'vector must be invariant under the infinitesimal generator'
            raise ValueError(msg)
        self._statdist = np.asarray(value) / np.sum(value)

    @property
    def jump_matrix(self):
        """(M, M) ndarray: Transition matrix of the embedded Markov chain."""
        return jump_matrix(self.rate_matrix)

    @property
    def jump_rates(self):
        """(M,) ndarray: Total transition rate out of each state."""
        return -self.rate_matrix.diagonal()

    @property
    def is_reversible(self):
        """bool: Whether the Markov chain is reversible."""
        X = self.stationary_distribution[:, np.newaxis] * self.rate_matrix
        return np.allclose(X, X.T)
    
    @property
    def states(self):
        """list: State labels."""
        return self._states

    @states.setter
    def states(self, value):
        if value is None:
            value = range(self.rate_matrix.shape[0])
        value = list(value)
        if len(value) > len(set(value)):
            raise ValueError('state labels must be unique')
        if len(value) != self.n_states:
            msg = 'number of labels must match number of states'
            raise ValueError(msg)
        self._states = value
        self._index = {x: i for i, x in enumerate(value)}

    @property
    def n_states(self):
        """int: The number of states."""
        return self.rate_matrix.shape[0]

    @property
    def state_to_index(self):
        """dict: Mapping from state labels to integer indices."""
        return self._index

    def committor(self, source, target, forward=True):
        """Committor probability between two sets of states.

        Parameters
        ----------
        source : list of int
            Indices of the source states.

        target : list of int
            Indices of the target states.

        forward : bool, optional
            If true, compute the forward committor (default). If false,
            compute the backward committor.

        Returns
        -------
        (M,) ndarray
            Vector of committor probabilities.

        """
        return msmana.committor(self.jump_matrix, source, target, 
                                forward=forward)

    def expectation(self, observable):
        """Stationary expectation of an observable.

        Parameters
        ----------
        observable : (M,) array_like
            A function on the state space of the Markov chain.

        Returns
        -------
        float
            Expected value of the observable with respect to the 
            stationary probability distribution.
        
        """
        return np.dot(observable, self.stationary_distribution)

    def mfpt(self, target):
        """Mean first passage time to a target set of states.

        Parameters
        ----------
        target : int or list of int
            Indices of the target states.

        Returns
        -------
        (M,) ndarray
            Mean first passage time from each state to the target set.

        """
        is_source = np.ones(self.n_states, dtype=bool)
        is_source[target] = False
        Q = self.rate_matrix[is_source, :][:, is_source]
        mfpt = np.zeros(self.n_states)
        mfpt[is_source] = np.linalg.solve(Q, -np.ones(Q.shape[0]))
        return mfpt

    def reactive_flux(self, source, target):
        """Reactive flux from transition path theory (TPT).

        Parameters
        ----------
        source : list of int
            Indices of the source states.

        target : list of int
            Indices of the target states.

        Returns
        -------
        msmtools.flux.ReactiveFlux
            An object describing the reactive flux. See MSMTools 
            documentation for full details.

        """
        qplus = self.committor(source, target)
        if self.is_reversible:
            qminus = 1.0 - qplus
        else:
            qminus = self.committor(source, target, forward=False)

        flux_matrix = msmtools.flux.flux_matrix(self.rate_matrix,
            self.stationary_distribution, qminus, qplus)

        return msmtools.flux.ReactiveFlux(source, target, flux_matrix,
            mu=self.stationary_distribution, qminus=qminus, qplus=qplus)


def jump_matrix(rate_matrix):
    """Extract the jump matrix from a rate matrix.

    Parameters
    ----------
    rate_matrix : (M, M) array_like
        Transition rate matrix, row infinitesimal stochastic.

    Returns
    -------
    (M, M) ndarray
        The jump matrix (embedded transition matrix) of the
        continuous-time Markov chain `rate_matrix`.

    """
    rate_matrix = np.asarray(rate_matrix)
    if not msmana.is_rate_matrix(rate_matrix):
          raise ValueError('matrix must be row infinitesimal stochastic')
    jump_rates = -rate_matrix.diagonal()
    jump_matrix = rate_matrix / jump_rates[:, np.newaxis]
    np.fill_diagonal(jump_matrix, 0)
    return jump_matrix


def rate_matrix(jump_matrix, jump_rates):
    """Return the rate matrix with given jump matrix and jump rates.

    Parameters
    ----------
    jump_matrix : (M, M) array_like
        Jump probability matrix (embedded transition matrix). Must be 
        row stochastic with all zeros on the diagonal.

    jump_rates : (M,) array_like
        Total transition rate out of each state.

    Returns
    -------
    (M, M) ndarray
        The transition rate matrix of the continuous-time Markov chain
        with the given jump matrix (embedded chain) and jump rates.

    """
    jump_matrix = np.asarray(jump_matrix)
    if not msmana.is_tmatrix(jump_matrix):
        raise ValueError('matrix must be row stochastic')
    if np.count_nonzero(jump_matrix.diagonal()):
        raise ValueError('diagonal elements must be equal to zero')

    jump_rates = np.asarray(jump_rates)
    if jump_rates.shape != (jump_matrix.shape[0],):
        msg = 'number of jump rates must match number of states'
        raise ValueError(msg)
    if not (jump_rates > 0).all():
        raise ValueError('jump rates must be positive')
    
    rate_matrix = jump_rates[:, np.newaxis] * jump_matrix
    rate_matrix[np.diag_indices_from(rate_matrix)] = -jump_rates

    return rate_matrix


def stationary_distribution(rate_matrix):
    """Compute the stationary distribution of a rate matrix.

    Parameters
    ----------
    rate_matrix : (M, M) array_like
        Transition rate matrix, row infinitesimal stochastic.

    Returns
    -------
    (M,) ndarray
        Stationary distribution of `rate_matrix`.

    """
    P = jump_matrix(rate_matrix)
    mu = -msmana.stationary_distribution(P) / np.diagonal(rate_matrix)
    return mu / mu.sum()


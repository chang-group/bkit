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

    states : iterable, optional
        State labels. Values must be unique and hashable. Will default 
        to ``range(M)`` if not provided.
 
    """

    def __init__(self, rate_matrix, stationary_distribution=None, states=None):
        self.rate_matrix = rate_matrix
        self.stationary_distribution = stationary_distribution
        self.states = states

    @property
    def rate_matrix(self):
        """ndarray: Transition rate matrix (infinitesimal generator)."""
        return self._rate_matrix

    @rate_matrix.setter
    def rate_matrix(self, value):
        value = np.asarray(value)
        if not msmana.is_rate_matrix(value):
          raise ValueError('matrix must be row infinitesimal stochastic')
        self._rate_matrix = value
    
    @property
    def stationary_distribution(self): 
        """ndarray: Stationary distribution, normalized to 1."""
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
    def embedded_tmatrix(self):
        """ndarray: Transition matrix of the embedded chain."""
        return embedded_tmatrix(self.rate_matrix)

    @property
    def jump_rates(self):
        """ndarray: Rate parameters."""
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
        return msmana.committor(self.embedded_tmatrix, source, target, 
                                forward=forward)

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


def embedded_tmatrix(rate_matrix):
    """Embedded transition matrix of a continuous-time Markov chain.

    Parameters
    ----------
    rate_matrix : (M, M) array_like
        Transition rate matrix, row infinitesimal stochastic.

    Returns
    -------
    (M, M) ndarray
        Transition matrix of the embedded chain of the continuous-time
        Markov chain with rate matrix `rate_matrix`.

    """
    rate_matrix = np.asarray(rate_matrix)
    if not msmana.is_rate_matrix(rate_matrix):
          raise ValueError('matrix must be row infinitesimal stochastic')
    jump_rates = -rate_matrix.diagonal()
    embedded_tmatrix = rate_matrix / jump_rates[:, np.newaxis]
    np.fill_diagonal(embedded_tmatrix, 0)
    return embedded_tmatrix


def rate_matrix(embedded_tmatrix, jump_rates):
    """Transition rate matrix of the continuous-time Markov chain with
    given embedded chain and jump rates.

    Parameters
    ----------
    embedded_tmatrix : (M, M) array_like
        Transition matrix of the embedded discrete-time Markov chain. 
        Must be row stochastic with diagonal elements all equal to zero.

    jump_rates : (M,) array_like
        Exponential rate parameters.

    Returns
    -------
    (M, M) ndarray
        Rate matrix of the continuous-time Markov chain with embedded 
        chain `embedded_tmatrix` and jump rates `jump_rates`.

    """
    embedded_tmatrix = np.asarray(embedded_tmatrix)
    if not msmana.is_tmatrix(embedded_tmatrix):
        raise ValueError('matrix must be row stochastic')
    if np.count_nonzero(embedded_tmatrix.diagonal()):
        raise ValueError('diagonal elements must be equal to zero')

    jump_rates = np.asarray(jump_rates)
    if jump_rates.shape != (embedded_tmatrix.shape[0],):
        msg = 'number of jump rates must match number of states'
        raise ValueError(msg)
    if not (jump_rates > 0).all():
        raise ValueError('jump rates must be positive')
    
    rate_matrix = jump_rates[:, np.newaxis] * embedded_tmatrix
    rate_matrix[np.diag_indices_from(rate_matrix)] = -jump_rates

    return rate_matrix


def stationary_distribution(rate_matrix):
    """Stationary distribution of a transition rate matrix..

    Parameters
    ----------
    rate_matrix : (M, M) array_like
        Transition rate matrix, row infinitesimal stochastic.

    Returns
    -------
    (M,) ndarray
        Stationary distribution of `rate_matrix`, normalized to 1.

    """
    P = embedded_tmatrix(rate_matrix)
    mu = -msmana.stationary_distribution(P) / np.diagonal(rate_matrix)
    return mu / mu.sum()


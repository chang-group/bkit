"""Algorithms for analysis of continuous-time Markov chains."""

import numpy as np
import msmtools.analysis as msmana
import msmtools.flux
import msmtools.generation as msmgen
import sys


class ContinuousTimeMarkovChain:
    """A continuous-time Markov chain.
        
    Parameters
    ----------
    rate_matrix: (M, M) array_like
        A transition rate (infinitesimal generator) matrix, with row sums
        equal to zero.
    stationary_distribution : (M,) array_like, optional
        Stationary distribution. Must be invariant with respect to the
        rate matrix. If not provided, the stationary distribution will be
        computed during initialization.
    states : (M,) array_like, optional
        State labels. Values must be unique and hashable. Will default 
        to range(M) if not provided.
 
    """

    def __init__(self, rate_matrix, stationary_distribution=None, states=None):
        self.rate_matrix = rate_matrix
        self.stationary_distribution = stationary_distribution
        self.states = states

    @property
    def rate_matrix(self):
        """(M, M) ndarray: Infinitesimal generator matrix."""
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
        """bool: Whether the chain satisfies detailed balance."""
        X = self.stationary_distribution[:, np.newaxis] * self.rate_matrix
        return np.allclose(X, X.T)
    
    @property
    def states(self):
        """(M,) ndarray of objects: State labels."""
        return np.asarray(self._states)

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

    def index(self, state):
        """Return the zero-based index of a state.

        Parameters
        ----------
        state : hashable
            A state label.

        Returns
        -------
        int
            The zero-based index of `state`. Raises a :py:exc:`ValueError`
            if there is no state with the given label.

        """
        try:
            return self._index[state]
        except KeyError:
            raise ValueError(f'state {state} is not in the chain')

    def committor(self, source, target, forward=True):
        """Compute the committor between sets of states.

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
        """Compute the stationary expectation of an observable.

        Parameters
        ----------
        observable : (M,) array_like
            Observable vector on the state space of the Markov chain.

        Returns
        -------
        float
            The expected value of `observable` with respect to the 
            stationary probability distribution.
        
        """
        return np.dot(observable, self.stationary_distribution)

    def mfpt(self, target):
        """Compute the mean first passage time to a set of states.

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
        """Compute the reactive flux between sets of states.

        Parameters
        ----------
        source : list of int
            Indices of the source states.
        target : list of int
            Indices of the target states.

        Returns
        -------
        msmtools.flux.ReactiveFlux
            An object describing the reactive flux. 

        See Also
        --------
        :class:`msmtools.flux.ReactiveFlux`
            The type of object returned by this method.

        """
        qplus = self.committor(source, target)
        if self.is_reversible:
            qminus = 1. - qplus
        else:
            qminus = self.committor(source, target, forward=False)

        flux_matrix = msmtools.flux.flux_matrix(self.rate_matrix,
            self.stationary_distribution, qminus, qplus)

        return msmtools.flux.ReactiveFlux(source, target, flux_matrix,
            mu=self.stationary_distribution, qminus=qminus, qplus=qplus)

    def simulate(self, n_jumps=None, start=None, target=None):
        """Generate a realization of the chain.

        The simulation will stop after a given number of jumps or when
        a given target is reached. If both are provided, the simulation
        length is determined by the earlier of the two stopping times.

        Parameters
        ----------
        n_jumps : int, optional
            Number of jumps to simulate. Required when `target` is None.
        start : int, optional
            Index of the starting state. If not provided, it will be drawn
            from the stationary distribution of :attr:`jump_matrix`.
        target : int or list of int, optional
            Indices of the target states. Required when `n_jumps` is None.

        Returns
        -------
        dtraj : sequence
            The sequence of states visited by the chain.
        arrival_times : sequence of floats
            The increasing sequence of arrival (jump) times. The arrival
            time at the starting state is defined to be zero.

        See Also
        --------
        :func:`msmtools.generation.generate_traj`
            Used to generate a realization of the embedded Markov chain.

        """
        if n_jumps is None:
            if target is None:
                raise ValueError('must provide a stopping criterion')
            n_jumps = sys.maxsize
        elif n_jumps < 0:
                raise ValueError('number of jumps must be nonnegative')

        dtraj = msmgen.generate_traj(self.jump_matrix, n_jumps+1,
                                     start=start, stop=target)

        arrival_times = np.zeros_like(dtraj, dtype=float)
        if len(dtraj) > 1:
            rng = np.random.default_rng()
            mean_lifetimes = 1. / self.jump_rates[dtraj[:-1]]
            lifetimes = rng.exponential(scale=mean_lifetimes)
            arrival_times[1:] = np.cumsum(lifetimes)

        return self.states[dtraj], arrival_times


def jump_matrix(rate_matrix):
    """Extract the jump probabilities from a rate matrix.

    Parameters
    ----------
    rate_matrix : (M, M) array_like
        A transition rate matrix, with row sums equal to zero.

    Returns
    -------
    (M, M) ndarray
        The jump matrix (embedded transition matrix) derived from 
        `rate_matrix`.

    """
    rate_matrix = np.asarray(rate_matrix)
    if not msmana.is_rate_matrix(rate_matrix):
          raise ValueError('matrix must be row infinitesimal stochastic')
    jump_rates = -rate_matrix.diagonal()
    jump_matrix = rate_matrix / jump_rates[:, np.newaxis]
    np.fill_diagonal(jump_matrix, 0)
    return jump_matrix


def rate_matrix(jump_matrix, jump_rates):
    """Return a rate matrix with given jump probabilities and jump rates.

    Parameters
    ----------
    jump_matrix : (M, M) array_like
        A jump probability matrix (embedded transition matrix). Must be 
        row stochastic with all zeros on the diagonal.
    jump_rates : (M,) array_like
        The total transition rate out of each state.

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
        A transition rate matrix, with row sums equal to zero.

    Returns
    -------
    (M,) ndarray
        The stationary distribution of `rate_matrix`.

    """
    P = jump_matrix(rate_matrix)
    mu = -msmana.stationary_distribution(P) / np.diagonal(rate_matrix)
    return mu / mu.sum()


import numpy as np
import deeptime.base
import deeptime.markov.msm as msm
import deeptime.markov.tools.analysis as msmana


class ContinuousTimeMarkovModel(deeptime.base.Model):
    """Continuous-time Markov model (i.e., Markov jump process)."""

    def __init__(self, rate_matrix):
        """Continuous-time Markov model with a given rate matrix.

        Parameters
        ----------
        rate_matrix : (M, M) ndarray
            Transition rate matrix, row infinitesimal stochastic.

        """
        self.rate_matrix = rate_matrix

    @property
    def rate_matrix(self):
        """Rate matrix (infinitesimal generator)."""
        return self._rate_matrix

    @rate_matrix.setter
    def rate_matrix(self, value):
        if not msmana.is_rate_matrix(value):
            raise ValueError('matrix must be row infinitesimal stochastic')
        self._rate_matrix = np.asarray(value)
        # Update embedded Markov chain
        P = self.rate_matrix / self.jump_rates[:, np.newaxis]
        np.fill_diagonal(P, 0)
        self._embedded_markov_model = msm.MarkovStateModel(P)

    @property
    def embedded_markov_model(self):
        """Embedded discrete-time Markov model."""
        return self._embedded_markov_model

    @property
    def n_states(self):
        """Number of states in the model."""
        return self.embedded_markov_model.n_states

    @property
    def jump_rates(self):
        """Exponential rate parameter associated with each state."""
        return -np.diag(self.rate_matrix)

    @property
    def stationary_distribution(self): 
        """Stationary distribution on the model states."""
        return (self.embedded_markov_model.stationary_distribution 
                / self.jump_rates)

    @property
    def reversible(self):
        """Whether the Markov chain is reversible."""
        return self.embedded_markov_model.reversible

    def mfpt(self, target):
        """Mean first passage times to a target set of states.

        Parameters
        ----------
        target : int or list of int
            Indices of the target states.

        Returns
        -------
        mfpt : (M,) ndarray
            Mean first passage time from each state to the target set.

        """
        is_source = np.ones(self.n_states, dtype=bool)
        is_source[target] = False
        Q = self.rate_matrix[is_source, :][:, is_source]
        mfpt = np.zeros(self.n_states)
        mfpt[is_source] = np.linalg.solve(Q, -np.ones(len(Q)))
        return mfpt


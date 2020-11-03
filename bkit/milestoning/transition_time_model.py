import numpy as np
from deeptime.markov import TransitionCountModel


class TransitionTimeModel(TransitionCountModel):
    """Transition count model with individually timed transitions."""
    
    def __init__(self, transition_times, state_symbols=None):
        """
        Parameters
        ----------
        transition_times : list of list of ndarray 
            Timed transition events between each pair of states. The
            element `transition_times[i][j]` store the times associated 
            with transitions from state `i` to state `j`. 

        """
        count_matrix = np.array([[len(a) for a in l] 
                                 for l in transition_times], dtype=int)
        super().__init__(count_matrix)
        self._transition_times = transition_times

    @property
    def transition_times(self):
        "The transition times for each pair of states."""
        return self._transition_times



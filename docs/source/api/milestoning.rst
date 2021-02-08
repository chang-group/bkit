=====================================
Milestoning (:mod:`bkit.milestoning`)
=====================================

.. currentmodule:: bkit.milestoning

Trajectory decomposition
========================

Mapping of continuous or discrete-state trajectories to 
*milestone schedules*. A milestone schedule is a list of tuples

.. math:: [(a_0,t_0),(a_1,t_1),\cdots,(a_N,t_N)],

where :math:`a_0,a_1,\dots,a_N` is the sequence of milestones visited,
and :math:`t_0,t_1,\dots,t_n` are the corresponding lifetimes.

.. autosummary::
   :toctree: generated/

   TrajectoryColoring
   color_discrete_trajectory

For users with data in the form of individual milestone-to-milestone
first-passage times, note that a first-passage event from milestone 
:math:`a` to milestone :math:`b` after a time :math:`t` can be 
represented by a schedule :math:`[(a,t),(b,0)]`.


Markovian milestoning
=====================

.. autosummary::
   :toctree: generated/

   MarkovianMilestoningEstimator
   MarkovianMilestoningModel


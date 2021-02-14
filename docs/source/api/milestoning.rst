=====================================
Milestoning (:mod:`bkit.milestoning`)
=====================================

.. currentmodule:: bkit.milestoning

Trajectory decomposition
========================

Mapping of trajectories to milestone 
`schedules <https://ncatlab.org/nlab/show/schedule>`_.

.. autosummary::
   :toctree: generated/

   TrajectoryColoring
   color_discrete_trajectory

A *milestone schedule* is a sequence of pairs 
:math:`((a_1,t_1),\cdots,(a_n,t_n))`, where :math:`a_1,\dots,a_n` are the
successive milestone states of a trajectory, and :math:`t_1,\dots,t_n` 
are the corresponding state lifetimes.

Model estimation
================

Estimation of dynamical models from milestone-schedule data.

.. autosummary::
   :toctree: generated/

   MarkovianMilestoningEstimator

.. note:: For users with data in the form of individual 
    milestone-to-milestone first-passage times, a first-passage event from
    milestone ``a`` to milestone ``b`` after a time ``t`` can 
    be represented by a schedule ``((a, t), (b, 0))``.

Model analysis
==============

Approximate models of milestone-to-milestone dynamics that can be queried
for various dynamical and stationary properties.

.. autosummary::
   :toctree: generated/

   MarkovianMilestoningModel





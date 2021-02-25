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
are the corresponding lifetimes.

Model estimation
================

Estimation of kinetic models from milestone schedule data.

.. autosummary::
   :toctree: generated/

   MarkovianMilestoningEstimator

Model analysis
==============

Models of milestoning dynamics with queryable properties and observables.

.. autosummary::
   :toctree: generated/

   MarkovianMilestoningModel

Utilities
=========

Common types, etc.

.. autosummary::
   :toctree: generated/

   MilestoneState


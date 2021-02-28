.. BKiT documentation master file, created by
   sphinx-quickstart on Tue Dec 15 20:47:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BKiT
====

BKiT is a Python package for estimation and analysis of
`milestoning <https://doi.org/10.1146/annurev-biophys-121219-081528>`_
models. It relies heavily on the functionality of the 
`MSMTools <https://msmtools.readthedocs.io>`_ library for low-level
computations.

Currently, BKiT supports the construction of *Markovian* milestoning 
models. Future versions may include support for additional model types
(e.g., semi-Markov and hidden Markov models).

A Markovian milestoning model is a continuous-time Markov chain (CTMC) 
with some extra structure. In case they might be useful, some tools 
for working with general CTMCs are also made public in the API.

Documentation
=============

.. toctree::
   :maxdepth: 2

   INSTALL
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

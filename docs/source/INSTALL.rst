============
Installation
============

We are working toward making BKiT available via 
`conda-forge <https://conda-forge.org/>`_ and `PyPI <https://pypi.org/>`_. 
In the meantime, to install the latest version, we recommend the following
approach. For this route you will need a working 
`conda <https://docs.conda.io/en/latest/>`_ installation.

1.  Create and activate a fresh conda environment, here named 'bkit':

    .. code::

        (base) $ conda create --name bkit
        (base) $ conda activate bkit

2.  Inside this environment, install `PyEMMA <http://pyemma.org>`_:

    .. code::
    
        (bkit) $ conda install pyemma --channel conda-forge
    
    This will install all BKiT dependencies, as well as provide you with 
    tools for manipulating and selecting features from MD trajectories 
    (namely, `MDTraj <https://mdtraj.org/>`_ and the PyEMMA 
    `coordinates <http://www.emma-project.org/latest/api/index_coor.html>`_
    package).

3.  Clone the bkit repository and install the package with pip:

    .. code::

        (bkit) $ git clone https://github.com/chang-group/bkit
        (bkit) $ pip install bkit/

4.  To run the example notebook, you will also need 
    `Jupyter <https://jupyter.org/>`_ notebook and 
    `ipympl <https://github.com/matplotlib/ipympl>`_:

    .. code::
    
        (bkit) $ conda install jupyter ipympl --channel conda-forge


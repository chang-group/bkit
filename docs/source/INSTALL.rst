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
    (namely, `MDTraj <https://mdtraj.org/>`_ and PyEMMA's
    `coordinates <http://www.emma-project.org/latest/api/index_coor.html>`_
    package). To install only the bare essentials, replace `pyemma` with 
    `msmtools` in the above command.

3.  Install BKiT from the GitHub repository using pip:

    .. code::

        (bkit) $ pip install git+https://github.com/chang-group/bkit

4.  To run the example notebooks, you will also need 
    `Jupyter <https://jupyter.org/>`_ notebook and 
    `ipympl <https://github.com/matplotlib/ipympl>`_:

    .. code::
    
        (bkit) $ conda install notebook ipympl --channel conda-forge


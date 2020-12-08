from setuptools import setup, find_packages

setup(
    name='bkit',
    version='0.1.0',
    description='Binding Kinetics Toolkit',
    url='https://github.com/chang-group/bkit',
    author='Jeff Thompson',
    author_email='jeffreyt@ucr.edu',
    classifiers=[  
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'msmtools']
)


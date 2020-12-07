from pyemma.coordinates.data import MDFeaturizer
from pyemma.coordinates.transform import PCA, TICA


class MDProjector:
    """Composition of a featurizer and a fitted transformer."""
    
    def __init__(self, featurizer, transformer):
        """Composition of a featurizer and a fitted transformer.
        
        Parameters
        ----------
        featurizer : MDFeaturizer
            Maps `mdtraj.Trajectory` data to a set of features.
        
        transformer : PCA or TICA
            Projects featurized data onto a set of eigenvectors.
        
        """
        if featurizer.dimension() != len(transformer.eigenvalues):
            msg = 'number of features must match transform input dimension'
            raise ValueError(msg)
        self.featurizer = featurizer
        self.transformer = transformer
        
    def dimension(self):
        """Output dimension."""
        return self.transformer.dimension()
        
    def transform(self, traj):
        """Project featurized MD data onto a set of eigenvectors."""
        return self.transformer.transform(self.featurizer.transform(traj))i


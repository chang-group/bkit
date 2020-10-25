import collections.abc


class Schedule(collections.abc.Sequence):
    """A 'schedule' as defined at https://ncatlab.org/nlab/show/schedule"""

    def __init__(self, labels=[], lengths=[]):
        assert len(labels) == len(lengths)
        assert all(t >= 0 for t in lengths)
        self.labels = labels
        self.lengths = lengths
    
    def append(self, label, length):
        assert length >= 0
        self.labels.append(label)
        self.lengths.append(length)

    def length(self):
        return sum(self.lengths)

    def scale(factor):
        assert factor >= 0
        self.lengths = [factor * t for t in self.lengths]
        
    def reduce(self):
        self.labels = [a for a, t in zip(self.labels, self.lengths) if t > 0]
        self.lengths = [t for t in self.lengths if t > 0] 

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return (self.labels[key], self.lengths[key])
        return Schedule(self.labels[key], self.lengths[key])

    def __str__(self):
        return ''.join(map(str, self))


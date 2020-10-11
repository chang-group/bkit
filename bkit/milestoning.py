import numpy as np
import scipy.spatial
import scipy.stats
import itertools
import networkx as nx

class Schedule:
    
    def __init__(self, labels=None, lengths=None):
        if labels:
            assert len(labels) == len(lengths)
            self.labels = labels
            self.lengths = np.array(lengths)
        else:
            self.labels = []
            self.lengths = np.array([])
    
    def append(self, a, t):
        self.labels.append(a)
        self.lengths = np.append(self.lengths, [t])

    def length(self):
        return self.lengths.sum()
    
    def scale_lengths(t):
        self.lengths =  self.lengths * t
        
    def reduce(self):
        selectors = self.lengths != 0
        self.labels = list(itertools.compress(self.labels, selectors))
        self.lengths = self.lengths[selectors]
    
    def is_reduced(self):
        return not np.any(self.lengths == 0)
    
    def __str__(self):
        return ''.join(['({}, {})'.format(a, t) for a, t in zip(self.labels, self.lengths)])

    def __getitem__(self, key):
        if isinstance(key, int):
            return (self.labels[key], self.lengths[key])
        else:
            return Schedule(self.labels[key], self.lengths[key])
    
    def __delitem__(self, key):
        del self.labels[key]
        self.lengths = np.delete(self.lengths, key)
    
    def __iter__(self):
        return zip(self.labels, self.lengths)
        
    def __len__(self):
        return len(self.labels)
    
class MilestoningModel:

    def __init__(self, anchors):
        nanchors, ndim = anchors.shape
        
        self._graph = nx.Graph()
        if ndim > 1:
            tri = scipy.spatial.Delaunay(anchors)
            indptr, indices = tri.vertex_neighbor_vertices
            for i in range(nanchors-1):
                self._graph.add_edges_from([(i, j) for j in indices[indptr[i]:indptr[i+1]]])
        else:
            self._graph.add_edges_from([(i, i+1) for i in range(nanchors-1)])
        
        self._anchor_kdtree = scipy.spatial.cKDTree(anchors)
        self._parent_node = list(range(nanchors))
        self._node_anchors = {i: set([tuple(anchors[i])]) for i in range(nanchors)}
        self._sampled_edges = set()
  
        self._schedules = []
        self._dtrajs = []
        self._cutoff = np.Inf
    
    @property
    def milestones(self):
        return [set(e) for e in self._graph.edges]
    
    @property
    def cell_anchors(self):
        return self._node_anchors
    
    @property
    def is_resolved(self):
        return all([self._graph.has_edge(i, j) for i, j in self._sampled_edges])
    
    @property
    def unresolved_jumps(self):
        return sorted([(i, j) for i, j in self._sampled_edges 
                       if not self._graph.has_edge(i, j)])
    
    def load_trajectory_data(self, trajs, dt=1):
        
        for traj in trajs:
            _, indices = self._anchor_kdtree.query(traj, distance_upper_bound=self._cutoff)
            
            node_path = [self._parent_node[k] for k in indices]
            self._dtrajs.append(node_path)
            
            edge_path = list(zip(node_path[:-1], node_path[1:]))
            
            schedule = Schedule()
            
            start = 1
            for source, target in edge_path:
                if target != source:
                    schedule.append((source, target), dt)
                    break
                start += 1
                
            for source, target in edge_path[start:]:
                if target in schedule.labels[-1]:
                    schedule.lengths[-1] += dt
                    continue
                schedule.append((source, target), dt)
            
            if schedule:
                self._schedules.append(schedule)
            self._sampled_edges |= set(schedule.labels)
            
    def remove_milestone(self, i, j):
        
        self._graph = nx.contracted_nodes(self._graph, i, j, self_loops=False)
        for a in self._node_anchors[j]:
            self._parent_node = i
        self._node_anchors[i] |= self._node_anchors.pop(j)
        
        self._sampled_edges = set()
        
        schedules = []
        for schedule_old in self._schedules:

            edge = schedule_old.labels[0]
            if edge == (i, j) or edge == (j, i):
                if not schedule_old[1:]:
                    continue
                schedule_old = schedule_old[1:]
            
            schedule = Schedule()
            
            (source, target), lifetime = schedule_old[0]
            if source == j:
                schedule.append((i, target), lifetime)
            elif target == j:
                schedule.append((source, i), lifetime)
            else:
                schedule.append((source, target), lifetime)

            for edge, lifetime in schedule_old[1:]:
                
                if edge == (i, j) or edge == (j, i):
                    schedule.lengths[-1] += lifetime
                    continue

                source, target = edge
                if target in schedule.labels[-1]:
                    schedule.lengths[-1] += lifetime
                    continue
                
                if source == j:
                    schedule.append((i, target), lifetime)
                elif target == j:
                    schedule.append((source, i), lifetime)
                else:
                    schedule.append((source, target), lifetime)
            
            if schedule:
                schedules.append(schedule)
            self._sampled_edges |= set(schedule.labels)
        
        self._schedules = schedules
    
    def estimate(self):
        
        milestone_index = {}
        for k, (i, j) in enumerate(self._graph.edges):
            milestone_index[(i, j)] = k
            milestone_index[(j, i)] = k
        
        nmilestones = self._graph.number_of_edges()
        self._N = np.zeros((nmilestones, nmilestones), dtype=np.int32)
        self._lifetimes = [[] for k in range(nmilestones)]
        
        for schedule in self._schedules:
            it = zip(schedule[:-1], schedule[1:])
            for i, ((a, t), (b, _)) in enumerate(it):
                if a in milestone_index and b in milestone_index:
                    self._N[milestone_index[a], milestone_index[b]] += 1

                    # If the source milestone is not the first milestone, 
                    # record the lifetime.
                    if i:
                        self._lifetimes[milestone_index[a]].append(t)
                    # (The time spent in the first milestone is undetermined,
                    # since we don't know whether the observed first crossing
                    # is a true first crossing.)
 
        self.K = self._N / np.sum(self._N, axis=1)[:, np.newaxis]

        self.t = np.array(list(map(np.mean, self._lifetimes)))
        self.t_stderr = np.array(list(map(scipy.stats.sem, self._lifetimes)))


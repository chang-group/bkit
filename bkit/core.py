import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d

import numpy as np
import scipy.spatial
import itertools as it
import networkx as nx

class Schedule:
    
    def __init__(self, labels=None, lengths=None):
        if labels and lengths:
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
        self.labels = list(it.compress(self.labels, selectors))
        self.lengths = self.lengths[selectors]
    
    def is_reduced(self):
        return not np.any(self.lengths == 0)
    
    def __str__(self):
        return ''.join(['({}, {})'.format(a, t) for a, t in zip(self.labels, self.lengths)])

    def __getitem__(self, key):
        if isinstance(key, int):
            return Schedule([self.labels[key]], [self.lengths[key]])
        else:
            return Schedule(self.labels[key], self.lengths[key])
    
    def __delitem__(self, key):
        del self.labels[key]
        self.lengths = np.delete(self.lengths, key)
    
    def __len__(self):
        return len(self.labels)
    
    
class MilestoningModel:

    def __init__(self, anchors):
        nanchors, ndim = anchors.shape
        
        self._graph = nx.Graph()
        if ndim > 1:
            tri = scipy.spatial.Delaunay(points)
            indptr, indices = tri.vertex_neighbor_vertices
            for i in range(nanchors-1):
                self._graph.add_edges_from([(i, j) for j in indices[indptr[i]:indptr[i+1]]])
        else:
            self._graph.add_edges_from([(i, i+1) for i in range(nanchors-1)])
            
        self._parent_node = list(range(nanchors))
        self._anchor_kdtree = scipy.spatial.cKDTree(anchors)
        
        self._schedules = []
    
    def is_resolved():
        pass
    
    def load_trajectory_data(self, trajs, dt=1):
        
        for traj in trajs:
            node_path = [self._parent_node[a] for a in self._anchor_kdtree.query(traj)[1]]
            edge_path = list(zip(node_path[:-1], node_path[1:]))
            
            schedule = Schedule()
            
            start = 1
            for (source, target) in edge_path:
                if target != source:
                    schedule.append((source, target), dt)
                    break
                start += 1
                
            for (source, target) in edge_path[start:]:
                if target in schedule.labels[-1]:
                    schedule.lengths[-1] += dt
                    continue
                schedule.append((source, target), dt)
                    
            self._schedules.append(schedule)


def CorrectXYnp(Xp,Yp,Xn,Yn, slope_opt):
    for i in range(1,len(slope_opt)):
        d1 = (Xp[i] - Xp[i-1])**2 + (Yp[i] - Yp[i-1])**2
        d2 = (Xn[i] - Xp[i-1])**2 + (Yn[i] - Yp[i-1])**2
        d3 = (Xn[i] - Xn[i-1])**2 + (Yn[i] - Yn[i-1])**2
        d4 = (Xp[i] - Xn[i-1])**2 + (Yp[i] - Yn[i-1])**2
        if max(d2,d4) < max(d1,d3):
            tmpx, tmpy = [Xp[i].copy(), Yp[i].copy()]
            tmpx1, tmpy1 = [Xn[i].copy(), Yn[i].copy()]
            Xp[i], Yp[i], Xn[i], Yn[i] = [tmpx1, tmpy1, tmpx, tmpy]

def findnorm(fp):
    slope = np.array([[None]*len(fp)],dtype=float).reshape(len(fp),1)
    for i in range(len(fp)-1):
        slope[i] = -(fp[i+1,0] - fp[i,0])/(fp[i+1,1] - fp[i,1])
    slope[-1]=slope[-2]
    return slope

def interpolateCurve(rawdata, pts=200, kind='cubic'):
    data = rawdata[:,1:3]

    pairlist = []
    for i in range(len(data)-1):
        pairlist.append([data[i], data[i+1]])
        
    cum_euc_dist = [0]
    dist = 0
    for i in range(len(pairlist)):
        dist += euclidean(pairlist[i][0],pairlist[i][1])
        cum_euc_dist.append(dist)
    cum_euc_dist = np.array(cum_euc_dist) 
    
    func1 = interp1d(cum_euc_dist, data[:,0], kind=kind)
    func2 = interp1d(cum_euc_dist, data[:,1], kind=kind)
    
    xnew = np.linspace(0, cum_euc_dist[-1], num=pts, endpoint=True)
    
    return np.column_stack((func1(xnew), func2(xnew)))

def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

def OptMileStone(x, y, xp, yp, xn, yn, slopeopt, step=0.05, niter=20):
    pd1, pd2 = [np.array([0.0]*len(slopeopt)*2).reshape(len(slopeopt),2) for i in range(2)]
    for j in range(niter):
        for i in range(1,len(slopeopt)):
            d1 = (xp[i] - xp[i-1])**2 + (yp[i] - yp[i-1])**2
            d2 = (xn[i] - xp[i-1])**2 + (yn[i] - yp[i-1])**2
            if d1>d2:
                tmpx, tmpy = [xp[i], yp[i]]
                xp[i], yp[i] = [xn[i], yn[i]]
                xn[i], yn[i] = [tmpx, tmpy]
        for i in range(1,len(slopeopt)-1):
            ip1 = max(i-1, 0)
            ip2 = min(i+1, len(slopeopt))
            pp11 = Project([xp[i],yp[i]], [xp[ip1],yp[ip1]], [x[ip1],y[ip1]])
            pp12 = Project([xp[i],yp[i]], [xp[ip2],yp[ip2]], [x[ip2],y[ip2]])
            pp21 = Project([xn[i],yn[i]], [xn[ip1],yn[ip1]], [x[ip1],y[ip1]])
            pp22 = Project([xn[i],yn[i]], [xn[ip2],yn[ip2]], [x[ip2],y[ip2]])
            mid1_x, mid1_y = [(pp11[j]+pp12[j])/2 for j in range(2)]
            mid2_x, mid2_y = [(pp21[j]+pp22[j])/2 for j in range(2)]
            pd1[i,0], pd1[i,1] = [(mid1_x-xp[i])*step, (mid1_y-yp[i])*step ]
            pd2[i,0], pd2[i,1] = [(mid2_x-xn[i])*step, (mid2_y-yn[i])*step ]
            
        pd1[0,0], pd1[0,1] = [pd1[1,0], pd1[1,1]]
        pd2[0,0], pd2[0,1] = [pd2[1,0], pd2[1,1]]
        xp += pd1[:,0].reshape(len(xp),1)
        yp += pd1[:,1].reshape(len(xp),1)
        xn += pd2[:,0].reshape(len(xp),1)
        yn += pd2[:,1].reshape(len(xp),1)
    
    # slope array N*1
    #global slope_opt 
    #slope_opt = (yp-y)/(xp-x)
    return (yp-y)/(xp-x)

def PlotTransition(TRANS, PCA1, PCA2, Xn, Xp, Yn, Yp, T, sc=20, trans_id=0, hit_type=0):
    N, ini, fin = TRANS[TRANS[:,-1]==hit_type][trans_id,[0,3,4]] # traj id, initial id, final id
#     print("Transition type:\t%s"%hit_type)
    print("Short MD ID:\t%s"%N)
    print("Initial frame:\t%s"%ini)
    print("Final frame:\t%s"%fin)

    trjx = PCA1[N*T:(N+1)*T]
    trjy = PCA2[N*T:(N+1)*T]

    plt.figure(figsize=(8,8))

    for i in range(len(Xn)):
        if i==0:
            plt.plot([Xn[i,0], Xp[i,0]],[Yn[i,0], Yp[i,0]],'-',
                 color='k', linewidth=1,label='milestones')
        else:
            plt.plot([Xn[i,0], Xp[i,0]],[Yn[i,0], Yp[i,0]],
                     '-',color='k', linewidth=1)

    plt.plot(trjx,trjy,color='gray', alpha=0.5, zorder=0)
    plt.scatter(trjx[ini:fin+2], trjy[ini:fin+2], 
                s=20, 
                c=range(int(fin+2-(ini))), 
                zorder=3
               )
    plt.plot(trjx[np.max([0,ini-2]):fin+2], trjy[np.max([0,ini-2]):fin+2],
             color='k', zorder=1
            )
    plt.scatter([trjx[ini],trjx[fin]], [trjy[ini],trjy[fin]],
                marker='x', s=200,
                c='r', zorder=2
               )

    plt.xlabel('PC1', fontsize=16)
    plt.ylabel('PC2',fontsize=16)
    plt.tick_params(labelsize=12)
#     plt.xlim([min(trjx)-1,max(trjx)+1])
#     plt.ylim([min(trjy)-1,max(trjy)+1])
    plt.show()

def Project(P, M, P1):
    PM_x = P[0]-M[0]
    PM_y = P[1]-M[1]
    P1M_x = P1[0]-M[0]
    P1M_y = P1[1]-M[1]
    L = P1M_x*P1M_x + P1M_y*P1M_y
    p = PM_x*P1M_x + PM_y*P1M_y
    pp_x = M[0] + p/L*P1M_x
    pp_y = M[1] + p/L*P1M_y
    return np.array([pp_x, pp_y])

def RotationMatrix(Xp,Yp,X,Y,r):
    RotM = []
    vec0 = np.array([1,0])
    for i in range(len(X)):
        vec1 = np.array([Xp[i]-X[i], Yp[i]-Y[i]])
        theta = np.arccos( np.dot(vec1.T, vec0)/r )
        if vec1[1]<0:
            theta = 2*np.pi-theta

        c, s = [np.cos(theta), np.sin(theta)]
        rot = np.array([[c, s],[-s, c]]).reshape(2,2)
        RotM.append(rot)
    RotM = np.asarray(RotM)
    
    return(RotM)

def area(x1,y1,x2,y2,x3,y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

def SortByMilestones(Xp, Xn, Yp, Yn, PCA1, PCA2):
    
    x1,y1=[Xp[:-1],Yp[:-1]]
    x2,y2=[Xn[:-1],Yn[:-1]]
    x3,y3=[Xp[1:],Yp[1:]]
    A1 = area(x1,y1,x2,y2,x3,y3)
    x1,y1=[Xn[1:],Yn[1:]]
    x2,y2=[Xn[:-1],Yn[:-1]]
    x3,y3=[Xp[1:],Yp[1:]]
    A2 = area(x1,y1,x2,y2,x3,y3)
    Acell = A1+A2

    midx=np.empty(len(PCA1))
    midx.fill(2*len(Xp))
    
    for i in log_progress(range(len(Xp)-1)):
        x1,y1=[Xp[i],Yp[i]]
        x2,y2=[Xn[i],Yn[i]]
        x3,y3=[Xn[i+1],Yn[i+1]]
        x4,y4=[Xp[i+1],Yp[i+1]]
        x,y=[PCA1[:],PCA2[:]]
        a1 = area(x,y,x1,y1,x2,y2)
        a2 = area(x,y,x2,y2,x3,y3)
        a3 = area(x,y,x3,y3,x4,y4)
        a4 = area(x,y,x4,y4,x1,y1)
        A = a1+a2+a3+a4
        index = abs(A-Acell[i][0])<10**-7
        midx[index]=i
    return midx  

def TransitionKernel(PCA1, PCA2, MIDX, slope_mod, fp_mod, X, Y, sc, T, check_escape=True):
    
    ntraj = int(len(PCA1)/T)
    TRANS = []

    # for touching
    idx = np.array([range(len(fp_mod))]*T)
    time = np.array([[i]*len(fp_mod) for i in range(T)])
    b = Y - slope_mod*X # b of y=m*x+b

    count_crossing, count_touch = 0,0

    for N in log_progress(range(ntraj),every=1000):
        trans_tmp=[]

    ######### hit by crossing #########
        midx2 = MIDX[N*T:(N+1)*T]
        transition = midx2[1:] - midx2[:-1]
        frameid = np.array(range(len(midx2)-1))
        id1 = abs(transition) == 1

        index = id1
        frameid_trans = frameid[index]
        diff= midx2[frameid_trans+1] - midx2[frameid_trans]
        milestone = np.max(np.array([midx2[frameid_trans],midx2[frameid_trans+1]]).T, axis=1)

        if len(milestone)>0:
            m_ini,t_ini = milestone[0],frameid_trans[0]
            for i in range(len(milestone)-1):

                if (abs(milestone[i+1] - milestone[i])==1):
                    m_end, t_end = milestone[i+1], frameid_trans[i+1]
                    if m_ini != milestone[i]:
                        m_ini,t_ini = milestone[i],frameid_trans[i]

                    # check if the transition travel outside the milestone line
                    if check_escape:
                        escape = (np.any(midx2[t_ini:t_end]==2*len(fp_mod)))
                    else:
                        escape = False

                    if escape==False:
                        TRANS.append([N, m_ini, m_end, t_ini, t_end,0])
                        trans_tmp.append([N, m_ini, m_end, t_ini, t_end,0])
                        count_crossing+=1
                        m_ini, t_ini = m_end, t_end
                else:
                    continue

#     ######### hit by touching #########
#         trjx = PCA1[N*T:(N+1)*T]
#         trjx = trjx.reshape(trjx.size,1)
#         trjy = PCA2[N*T:(N+1)*T]
#         trjy = trjy.reshape(trjy.size,1)
#         midx = np.array(MIDX[N*T:(N+1)*T],dtype=int)
#         midx = np.array([midx]*len(fp_mod)).T

#         # distance between the trajectory to the milestone line
#         dist = abs(trjy + (trjx*(-slope_mod).T) - b.T)/np.sqrt(1+slope_mod**2).T
#         index1 = dist < 0.4

#         # discard trajectory out of the cutoff radius
#         index2 = midx != 2*len(fp_mod)

#         # distance between the trajectory to the path
#         dist2 = np.sqrt( (trjy-Y.T)**2 + (trjx-X.T)**2 )
#         index3 = dist2 < sc/2

#         index_hit = np.all([index1,index2,index3],axis=0)

#         idx_ = idx[index_hit]
#         time_ = time[index_hit]
#         data_ = np.concatenate((time_.reshape(time_.size,1), 
#                                 idx_.reshape(idx_.size,1)),axis=1)

#         if len(data_)>0:
#             data_ini = data_[0]
#             for i in range(len(data_)-1):

#                 if (abs(data_[i+1,1] - data_[i,1])==1):
#                     data_end = data_[i+1]
#                     if data_ini[1] != data_[i,1]:
#                         data_ini = data_[i]

#                     # check if the transition travel outside the milestone line
#                     if check_escape:
#                         escape = (np.any(midx2[data_ini[0]:data_end[0]]==2*len(fp_mod)))
#                     else:
#                         escape = False                              

#                     # exclude "hit by corssing"
#                     if len(trans_tmp) != 0:
#                         for j in range(len(trans_tmp)):
#                             if (trans_tmp[j][3]<=data_ini[0]<=trans_tmp[j][4]) or (trans_tmp[j][3]<=data_end[0]<=trans_tmp[j][4]):
#                                 break
#                             elif (data_ini[0]<=trans_tmp[j][3]<=data_end[0]):
#                                 break

#                     elif escape==False:
#                         TRANS.append([N, data_ini[1], data_end[1], data_ini[0], data_end[0],1])
#                         count_touch+=1
#                         data_ini = data_end
#                 else:
#                     continue


    TRANS = np.array(TRANS).astype(int) # (NRUN, initial milestone, final milestone, initial frame, final frame, type)
#     print('crossing: %d, touching: %d'%(count_crossing,count_touch))
    print('total transition: %s'%count_crossing)
    
    return TRANS

# From example at https://matplotlib.org/3.1.1/users/event_handling.html
class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        #print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()


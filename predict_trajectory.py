from copyreg import pickle
import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from matplotlib import pyplot as plt
import time
import pickle

def knn(x:torch.Tensor, q:torch.Tensor, k:int=26, cuda = True):
  '''
  x: all points. (N, 3)
  q: query points. (Q, 3)
  k: number of neighbors
  '''
  device = torch.device('cuda') if cuda else torch.device('cpu')
  x = x.to(device)
  q = q.to(device)
  with torch.no_grad():
    qx = (q @ x.t()) # (Q, N)
    xx = torch.sum(x**2, dim=-1, keepdim=True) #(N, 1)
    qq = torch.sum(q**2, dim=-1, keepdim=True) #(Q, 1)
    neg_pairwise_distance = -(xx.t() - 2*qx + qq) #(Q, N)

    idx = neg_pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx
  
def kd_knn(x:np.array, q:np.array,k:int=26):
  '''
  x: all points. (N, 3)
  q: query points. (Q, 3)
  k: number of neighbors
  '''
  kd = KDTree(x, 10)
  dist, nn = kd.query(q, k)
  return nn, dist


class PathlineInterpolator():
  '''
  pl:
    pathlines (T, N, 3)
    
  pl_len:
    pathline lengths (N,)

  mass:
    particle mass (T, N)

  interp_fn:
    custom interpolation function
    fn(all, query) = interpolant

  knn_fn:
    custom knn function, return (Q, k), Q=# of query points, k=# of nn
  '''
  def __init__(self, pl:np.array, pl_len:np.array, interp_fn='idw', knn_fn=kd_knn):
    # data
    self.pl = pl
    self.pl_len = pl_len

    # functions
    self.interp_fn = interp_fn
    self.knn_fn = knn_fn

    # derived helper attributes
    self.tmax = self.pl.shape[0]


  def interp_nsteps(self, query, t, k, tend, forward=True):
    '''
    interpolation and advect query particles at time t to tend (tend included)
    query is assumed to be reasonabe (near set pathlines)
    '''
    # assert tend < self.tmax

    tlen = abs(tend - t) + 1
    num_query, dim = query.shape
    output_array = np.zeros((tlen,num_query,dim), dtype=np.float32)
    output_array[0] = query
    output_length = np.full((num_query,),tlen,dtype=np.int32)
    query_idx = np.arange(len(query))

    q_interp = query
    for timestep in range(t, tend, 1 if forward else -1):
      if __name__ == '__main__':
        print(timestep)
      q_interp, live_mask = self.interp_one(q_interp, k, timestep,forward=forward)
      query_idx, query_idx_term = query_idx[live_mask], query_idx[~live_mask]
      ts = timestep-t+1 if forward else timestep -1 
      output_length[query_idx_term] = ts
      output_array[ts, query_idx] = q_interp
    return output_array, output_length
  
  def get_data_mask(self, t, forward=True):
    '''
    Return the this and next time step data observations for time t. (Particles)
    rel_mask: true if this time step observation continue to exist in next time step.
    '''
    # assert t < self.tmax-1
    maskt0 = self.pl_len > t
    rel_mask = self.pl_len[maskt0] > (t + 1) if forward else self.pl_len[maskt0] > (t - 1)

    ts = self.pl[t,maskt0]
    ts_nx = self.pl[t+1, maskt0] if forward else self.pl[t-1, maskt0]
    return ts, ts_nx, rel_mask

  def interp_one(self, query, k, t, out_proportion:float = 0.01, return_neighbor = False, forward=True):
    '''
    query:
      (Q, 3) starting positions of query pathlines
      query is already masked

    k:
      number of neighbors to interpolate from

    t:
      starting timestep

    '''
    # assert t < self.tmax-1
    ts, ts_nx, rel_mask = self.get_data_mask(t, forward)
    ts_nx_vector = ts_nx - ts #change to relative 

    ##### check out of bound neighbors
    nn, dist = self.knn_fn(ts, query, k) #(Q,K)
    nn_mask = rel_mask[nn]
    neighbor_nx = ts_nx_vector[nn] #(Q, K)
    live_mask = nn_mask.sum(1) >= k * (1-out_proportion) #exclude the query if out_proportion is outside

    neighbor_nx = neighbor_nx[live_mask] #(Q_new, K)
    dist = dist[live_mask]
    nn_mask = nn_mask[live_mask]
    query = query[live_mask]
    nn = nn[live_mask]

    if self.interp_fn  == 'idw':
      #IDW
      eps = 1e-10
      weights = 1/(dist + eps) #* mass[ts_nn_idx] #(Q_new, K)
      weights[~nn_mask] = 0
      weights /= np.sum(weights,axis=1,keepdims=True) #(Q_new, K)
      interp_vector = np.sum(weights[:,:,None] * neighbor_nx, axis=1).astype(np.float32) #(Q_new)
      interp = query + interp_vector

    elif self.interp_fn == 'rbf':
      rbf = RBFInterpolator(ts,ts_nx,k, kernel = 'thin_plate_spline')
      interp = rbf(query).astype(np.float32).squeeze()

    elif self.interp_fn == 'lin':
      raise NotImplementedError
      lin = LinearNDInterpolator(ts, ts_nx)
      interp = lin(query).astype(np.float32)
      nan_mask = np.isnan(interp.sum(1))
      live_mask = live_mask & ~nan_mask
      interp = interp[~nan_mask]
      nn = nn[~nan_mask]
      neighbor_nx = neighbor_nx[~nan_mask]
    
    if not  return_neighbor:
        return interp, live_mask
    else: 
        return interp, live_mask, ts[nn], neighbor_nx






if __name__ == '__main__':
  # interp pathline
  # interplation method
  interp_method = 'idw'
  tra = np.load('data/tra_hcart.npy').transpose(1,0,2)
  tra_len = np.load('data/tra_len.npy')
  print(tra.shape,tra_len.shape)

  rand_num = np.random.rand(len(tra_len))
  train = tra[:,rand_num<0.75]
  train_len = tra_len[rand_num<0.75]
  test = tra[:,rand_num>=0.75]
  test_len = tra_len[rand_num>=0.75]

  interper = PathlineInterpolator(train, train_len, interp_method)
  start_time = 0
  end_time = 2000
  k = 26
  query_mask = test_len > start_time
  query = test[start_time, query_mask]
  
  t1 = time.time()
  interp, interp_len = interper.interp_nsteps(query,start_time,k, end_time, True)
  interp_back, interp_len_back = interper.interp_nsteps(query, start_time, k, 0, False)
  print(time.time()- t1)


  # error masked
  mse_all = 0
  count = 0
  mse_list = []
  for t in range(0,end_time+1):
      mask = (test_len[query_mask] > t) & (interp_len > (t-start_time))
      gt = test[t,query_mask]
      gt = gt[mask]
      if t >=start_time:
        # mask = test_len > t
        inp = interp[t-start_time,mask]
      else:
        inp = interp_back[t, mask]
      mse_all += ((gt - inp) ** 2).mean(1).sum()
      count += len(inp)
      mse = np.sqrt(((gt - inp) ** 2).mean())
      mse_list.append(mse)
  mse_array = np.array(mse_list)
  print('average mse', np.sqrt(mse_all/count))
  np.save('error/interp_tra_%d.npy' % start_time, mse_array)
  plt.plot(np.arange(len(mse_list)),mse_list)
  plt.savefig('fig/interp_tra_%s_%d.jpg' % (interp_method, start_time))
  
  # np.save('data/interp_tra_rbf', interp)
  # np.save('data/interp_tra_len_rbf', interp_len)

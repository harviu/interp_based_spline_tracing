import numpy as np
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator, splev
from matplotlib import pyplot as plt
from utils import load_spline_file, time_mse
from predict_trajectory import kd_knn
import pickle
from numba import jit
import time

@jit(nopython=True)
def identify_u(u_vector:np.array, idx:int) -> float:
    if idx >= len(u_vector):
      return -1
    else:
      return u_vector[idx]

@jit(nopython=True)
def write_result(active_idx:np.array, result:list, nx_interp:np.array) -> None:
  for idx in active_idx:
    result[idx].append(nx_interp[idx])

def splev_fn(t,c,k, u):
    # c(D,N)
    # t(N+k)
    tck = (t,c,k)
    ret = np.array(splev(u, tck, ext=0))
    return ret

@jit(nopython=True)
def find_larger_idx(knot:np.array, search_knot:float) -> int:
    '''
    Function to find the idx of a sorted array (increase) that is just larger to the floating point value
    '''
    # the first and the last two knot should be duplicated, return 0 or len(knot) -1 when the value is the same
    eps = 1e-8
    if search_knot < -eps:
      return -1
    elif search_knot < eps:
      return 0
    elif search_knot > knot[-1]+ eps:
      return -1
    elif search_knot > knot[-1]-eps:
      return len(knot) - 1
    else:
      idx = len(knot) - 1
      for i in range(0, len(knot)):
          if knot[i] - search_knot > eps:
              idx = i
              break
      return idx

def get_masked(mask, knot_idx, knots, cp):
    masked_t = knots[:,mask]
    masked_cp = cp[:,mask]
    masked_idx = knot_idx[mask]
    return masked_idx, masked_t, masked_cp


class ControlPointInterpolator():
  def __init__(self, tcku_dict:dict, interp_fn='idw'):
    '''
    We use the same number of knots every line
    '''
    self.interp_fn = interp_fn
    self.pl = tcku_dict['c'] #control points
    self.k = tcku_dict['k'] #degree
    self.t = tcku_dict['t'] #knots
    self.u = tcku_dict['u'] #u notice u has the N as the first dimension
    self.T, self.N = self.t.shape
    # self.TP = np.array(list(map(lambda u:len(u), self.u)))

  def interp_any_start(self, starting_point, ts, return_neighbor = False):
    # first identify the nearest knot
    assert len(starting_point[0]) == 3
    assert len(starting_point.shape) == 2
    nn_k=26
    
    # time step
    # ts = round((starting_point[0,3] + 6.9068203) / (6.9068203 * 2)* 2000)
    print("time_step:",ts)
    u = float(ts) /2000
    pad_t = self.t.T
    pad_t = np.pad(pad_t, ((0,0),(2,2)), constant_values= ((0,1)))
    C_u = list(map(splev_fn, pad_t, self.pl.transpose(1,2,0), [self.k] * self.N, [u] * self.N))
    knot_idx = np.array(list(map(find_larger_idx, self.t.T, [u] * self.N)))
    mask = ~(knot_idx==-1)
    masked_idx, masked_t, masked_cp = get_masked(mask, knot_idx, self.t, self.pl)
    C_u = np.array(C_u)[mask]

    # interpolate the first knot and control point
    nn, dist = kd_knn(C_u,starting_point,nn_k)
    N_idx = np.arange(masked_t.shape[1])
    knot_nx = masked_t[masked_idx, N_idx][nn]
    cp_nx = masked_cp[masked_idx, N_idx,:][nn]

    eps = 1e-10
    weights = 1/(dist + eps) #* mass[ts_nn_idx] #(Q, K)
    weights /= np.sum(weights,axis=1,keepdims=True) #(Q, K)
    knot_interp = (knot_nx * weights).sum(1).astype(np.float32)
    cp_rel_nx = cp_nx - starting_point[:,None,:]
    cp_interp = (cp_rel_nx* weights[:,:,None]).sum(1).astype(np.float32) + starting_point
    print("Finished Calculating the first knots and cps")

    # interp forward
    forward_idx = knot_idx.copy()
    query_knot_cp = np.concatenate((knot_interp[:,None], cp_interp), axis=-1)
    result = np.zeros((1, query_knot_cp.shape[0], query_knot_cp.shape[1]))
    result[0] = query_knot_cp
    active_idx = np.arange(starting_point.shape[0])
    ite = 0

    while(True):
      print("ite",ite)
      ite += 1
      nx_idx = np.where(forward_idx!=-1, forward_idx +1, -1)
      nx_idx[nx_idx>=self.T] = -1
      mask = ~(nx_idx==-1)
      if mask.sum() == 0:
        break
      masked_idx, masked_t, masked_cp = get_masked(mask, forward_idx, self.t, self.pl)
      masked_nx_idx = nx_idx[mask]
      N_idx = np.arange(masked_t.shape[1])
      knot = masked_t[masked_idx, N_idx]
      cp = masked_cp[masked_idx, N_idx]
      knot_cp = np.concatenate((knot[:,None], cp), axis=-1)
      # need to think more carefully about the end 
      if len(knot_cp) < nn_k:
        break
      nn, dist = kd_knn(knot_cp, query_knot_cp, min(nn_k,len(knot_cp)))
      knot_cp = knot_cp[nn]
      knot_nx = masked_t[masked_nx_idx, N_idx][nn]
      cp_nx = masked_cp[masked_nx_idx, N_idx][nn]
      knot_cp_nx = np.concatenate((knot_nx[:,:,None], cp_nx), axis=-1)
      rel_nx = knot_cp_nx - knot_cp
      nx_interp = self.interp_one_raw(dist, rel_nx, query_knot_cp)
      nx_result = np.zeros((1,starting_point.shape[0], starting_point.shape[1]+1))
      nx_result[:,active_idx,:] = nx_interp
      # mask = nx_interp[:,3] <= 6.9068203
      mask = nx_interp[:,0] <= 1
      result = np.concatenate((result,nx_result),axis=0)

      active_idx = active_idx[mask]
      query_knot_cp = nx_interp[mask].copy()

      forward_idx = nx_idx
      if mask.sum() == 0:
        break

    # interp backward
    # may miss some pathlines
    backward_idx = knot_idx.copy()
    query_knot_cp = np.concatenate((knot_interp[:,None], cp_interp), axis=-1)
    active_idx = np.arange(starting_point.shape[0])

    while(True):
      print("ite",ite)
      ite += 1
      nx_idx = np.where(backward_idx!=-1, backward_idx - 1, -1)
      nx_idx[nx_idx<0] = -1
      mask = ~(nx_idx==-1)
      if mask.sum() == 0:
        break
      masked_idx, masked_t, masked_cp = get_masked(mask, backward_idx, self.t, self.pl)
      masked_nx_idx = nx_idx[mask]
      N_idx = np.arange(masked_t.shape[1])
      knot = masked_t[masked_idx, N_idx]
      cp = masked_cp[masked_idx, N_idx]
      knot_cp = np.concatenate((knot[:,None], cp), axis=-1)
      if len(knot_cp) < nn_k:
        break
      nn, dist = kd_knn(knot_cp, query_knot_cp, min(nn_k,len(knot_cp)))
      knot_cp = knot_cp[nn]
      knot_nx = masked_t[masked_nx_idx, N_idx][nn]
      cp_nx = masked_cp[masked_nx_idx, N_idx][nn]
      knot_cp_nx = np.concatenate((knot_nx[:,:,None], cp_nx), axis=-1)
      rel_nx = knot_cp_nx - knot_cp
      nx_interp = self.interp_one_raw(dist, rel_nx, query_knot_cp)
      nx_result = np.zeros((1,starting_point.shape[0], starting_point.shape[1]+1))
      nx_result[:,active_idx,:] = nx_interp
      # mask = nx_interp[:,3] >= -6.9068203
      mask = nx_interp[:,0] >= 0
      result = np.concatenate((nx_result, result),axis=0)

      active_idx = active_idx[mask]
      query_knot_cp = nx_interp[mask].copy()

      backward_idx = nx_idx
      if mask.sum() == 0:
        break
    return result
    

  def interp_one_raw(self, dist, ts_nx_rel, query):
      if self.interp_fn  == 'idw':
          #IDW
          eps = 1e-10
          weights = 1/(dist + eps) #* mass[ts_nn_idx] #(Q, K)
          weights /= np.sum(weights,axis=1,keepdims=True) #(Q, K)
          nx_vector = np.sum(weights[:,:,None] * ts_nx_rel, axis=1).astype(np.float32) #(Q)
      nx_interp = nx_vector + query
      return nx_interp

  def interp_nsteps(self, query, query_knot, tmin, tmax, k, return_neighbor=False):
    '''
    interpolation and advect query particles at time t to tend (tend included)
    query is assumed to be reasonabe (near set pathlines)
    '''
    assert tmin < self.T
    assert tmax < self.T

    tlen = tmax - tmin + 1
    num_query, dim = query.shape
    output_cp = np.zeros((tlen,num_query,dim), dtype=np.float32)
    output_cp[0] = query
    output_knot = np.zeros((tlen,num_query),np.float32)
    output_knot[0] = query_knot
    output_nn = np.zeros((tlen,num_query,k,dim+1),np.float32)

    q_interp = query.copy()
    knot_interp = query_knot.copy()
    for t in range(tmin, tmax):
      if __name__ == '__main__':
        print(t)
      if return_neighbor:
        q_interp, knot_interp, nn1, nn2 = self.interp_one(q_interp, knot_interp, t, k, return_neighbor=return_neighbor)
        if t == tmin:
          output_nn[t-tmin] = nn1
        output_nn[t-tmin+1] = nn2
      else:
        q_interp, knot_interp = self.interp_one(q_interp, knot_interp, t, k)
      output_cp[t-tmin+1] = q_interp
      output_knot[t-tmin+1] = knot_interp
    if return_neighbor:
      return output_cp, output_knot, output_nn
    else:
      return output_cp, output_knot

  def interp_noacc(self, query, query_knot, tmin, tmax, k, return_neighbor=False):
    '''
    interpolation query particles from time t to tend (tend included)
    query is assumed to be reasonabe (near set pathlines)
    Do not accumulate errors, query: (T,N,D)
    '''
    assert tmin < tmax
    assert tmax < self.tmax

    tlen = tmax - tmin + 1
    T, num_query, dim = query.shape
    assert T == tlen
    output_cp = np.zeros((tlen,num_query,dim), dtype=np.float32)
    output_cp[0] = query[0]
    output_knot = np.zeros((tlen,num_query),np.float32)
    output_knot[0] = query_knot[0]
    output_nn = np.zeros((tlen,num_query,k,dim+1),np.float32)

    for t in range(tmin, tmax):
      if return_neighbor:
        q_interp, knot_interp, nn1, nn2 = self.interp_one(query[t], query_knot[t], t, k, return_neighbor=return_neighbor)
        if t == tmin:
          output_nn[t-tmin] = nn1
        output_nn[t-tmin+1] = nn2
      else:
        q_interp, knot_interp = self.interp_one(query[t], query_knot[t], t, k)
      output_cp[t-tmin+1] = q_interp
      output_knot[t-tmin+1] = knot_interp
    if return_neighbor:
      return output_cp, output_knot, output_nn
    else:
      return output_cp, output_knot

  def interp_one(self, query, query_knot, t, k, out_proportion: float = 0.01, return_neighbor=False):
    '''
    input t is the control point time step 
    '''

    ts = self.pl[t]
    knot = self.t[t]
    ts_nx = self.pl[t+1] - self.pl[t]
    knot_nx = self.t[t+1] - self.t[t]

    ##### check out of bound neighbors
    nn, dist = kd_knn(ts, query, k) #(Q,K)
    neighbor_nx = ts_nx[nn] #(Q, K)
    knot_nx_nn = knot_nx[nn]

    if self.interp_fn  == 'idw':
      #IDW
      eps = 1e-10
      weights = 1/(dist + eps) #* mass[ts_nn_idx] #(Q, K)
      weights /= np.sum(weights,axis=1,keepdims=True) #(Q, K)
      cp_vector = np.sum(weights[:,:,None] * neighbor_nx, axis=1).astype(np.float32) #(Q)
      knot_vector = np.sum(weights * knot_nx_nn, axis=1).astype(np.float32) #(Q)

    elif self.interp_fn == 'rbf':
      rbf = RBFInterpolator(ts, ts_nx, k, kernel='linear')
      cp_vector = rbf(query).astype(np.float32).squeeze()
      rbf_knot = RBFInterpolator(ts, knot_nx, k)
      knot_vector = rbf_knot(query).astype(np.float32).squeeze()

    elif self.interp_fn == 'lin':
      raise NotImplementedError
      # lin = LinearNDInterpolator(ts, ts_nx)
      # interp = lin(query).astype(np.float32)
      # nan_mask = np.isnan(interp.sum(1))
      # interp = interp[~nan_mask]
      # nn = nn[~nan_mask]
      # neighbor_nx = neighbor_nx[~nan_mask]

    q_interp = query + cp_vector
    knot_interp = query_knot + knot_vector

    if not  return_neighbor:
        return q_interp, knot_interp
    else: 
        return (q_interp, knot_interp, 
        np.concatenate([self.pl[t][nn], self.t[t][nn][:,:,None]],axis = -1), 
        np.concatenate([self.pl[t+1][nn], self.t[t+1][nn][:,:,None]],axis = -1), )


if __name__ == '__main__':
    #interp control points
    for n_cp in [10,25,50,100]:
      print(n_cp)
      fname = 'data/spl_3d_cp_%d.npy' % n_cp
      train, test, train_idx, test_idx = load_spline_file(fname,True, False)
      tra = np.load('data/tra_hcart.npy')
      tra_len = np.load('data/tra_len.npy')
      # np.save('train_idx.npy',train_idx)
      # np.save('test_idx.npy',test_idx)
      # with open('data/train.pkl','wb') as f:
      #   pickle.dump(train, f)
      # I remove the truncate method in the utils
      test['t'] = test['t'][2:-2]
      train['t'] = train['t'][2:-2]
      # val['t'] = val['t'][2:-2]

      interper = ControlPointInterpolator(train,'idw')
      start_time = 0

      t1 = time.time()
      result = interper.interp_any_start(tra[test_idx, start_time], start_time, 26)
      print(time.time()- t1)
      gt = tra[test_idx]
      gt_len = tra_len[test_idx]

      all_mse = np.zeros((2001))
      total_e = 0
      count = 0
      acc = np.zeros((2001))
      
      recon_path_3d = []
      gt_path_3d = []
      result = result.transpose(1,0,2)
      for i,spline in enumerate(result):
        mask = spline.sum(1) != 0 
        spline = spline[mask]
        t = spline[:,0]
        t = (t-t.min())/(t.max()-t.min())
        t[1] = 0
        t[-2] = 1
        t = np.pad(t, ((2,2)), constant_values= ((0,1)))
        c = spline[:,1:].T
        k = test['k']
        u = test['u'][i]
        path = np.array(splev(u, (t,c,k), ext=1)).T
        gtline = gt[i,:gt_len[i]]
        time_diff = gtline - path
        time_mse = np.sqrt((time_diff ** 2).mean(1))
        total_e += (time_diff ** 2).mean(1).sum()
        count += len(time_diff)

        recon_path = np.concatenate((path, time_mse[:,None]),axis=-1)
        gt_path = np.concatenate((gtline, time_mse[:,None]),axis=-1)
        recon_path_3d.append(recon_path)
        gt_path_3d.append(gt_path)

        all_mse[:gt_len[i]] += time_mse
        acc[:gt_len[i]] += 1
      all_mse /= acc
      ave_e = total_e / count
      print(np.sqrt(ave_e))

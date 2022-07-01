from cProfile import label
import numpy as np
import math
import h5py
from matplotlib import pyplot as plt
from scipy.interpolate import splev

def mse(a1:np.array, a2:np.array):
  return ((a1-a2) * (a1-a2)).mean()

def max_error(a1, a2):
    diff = a1 - a2
    dist = (diff * diff).sum(-1)
    return dist.max()

def pad_knot(knot, k=3):
    if knot.ndim == 1:
        knot = knot[None,:]
    N, T = knot.shape
    knot_full = np.zeros((N,T+(k-1)*2),np.float32)
    knot_full[:,k-1:-(k-1)] = knot
    knot_full[:,-(k-1):] = knot[:,-1:]
    return knot_full.squeeze()

def time_mse(a1, a2):
  '''
  a1 (T,N,...)
  '''
  T = a1.shape[0]
  a1 = a1.reshape(T,-1)
  a2 = a2.reshape(T,-1)
  return ((a1-a2) * (a1-a2)).mean(1)

def load_spline_file(fname, return_idx = False, use_val=True, seed = 0):
    spline = np.load(fname, allow_pickle=True).item()
    np.random.seed(seed)
    rand_num = np.random.rand(spline['c'].shape[0])
    degree = spline['k']
    u = np.array(spline['u'],dtype=object)
    # t = spline['t'].T[2:-2].astype(np.float32)
    t = spline['t'].T.astype(np.float32)
    c = spline['c'].transpose(1,0,2).astype(np.float32)
    train = {
        'u': u[rand_num<0.75],
        't': t[:,rand_num<0.75],
        'c': c[:,rand_num<0.75],
        'k': degree,
    }
    if use_val:
        test = {
            'u': u[(rand_num>=0.75) & (rand_num<0.875)],
            't': t[:,(rand_num>=0.75) & (rand_num<0.875)],
            'c': c[:,(rand_num>=0.75) & (rand_num<0.875)],
            'k': degree,
        }
        validate = {
            'u': u[rand_num>=0.875],
            't': t[:,rand_num>=0.875],
            'c': c[:,rand_num>=0.875],
            'k': degree,
        }
        if return_idx:
            all_idx = np.arange(len(u))
            train_idx = all_idx[rand_num<0.75]
            test_idx = all_idx[(rand_num>=0.75) & (rand_num<0.875)]
            val_idx = all_idx[rand_num>=0.875]
            return train, test, validate, train_idx, test_idx, val_idx
        else:
            return train, test, validate
    else:
        test = {
            'u': u[rand_num>=0.75],
            't': t[:,rand_num>=0.75],
            'c': c[:,rand_num>=0.75],
            'k': degree,
        }
        if return_idx:
            all_idx = np.arange(len(u))
            train_idx = all_idx[rand_num<0.75]
            test_idx = all_idx[rand_num>=0.75]
            return train, test, train_idx, test_idx
        else:
            return train, test


def draw_tra(trace, l, draw_points = False, ax = None, label = None):
    '''
    trace for one time step
    '''
    if ax is None:
        fig = plt.figure(figsize=(16, 10), dpi=80)
        ax = fig.add_subplot(1,1,1,projection='3d')
    l = min(l,len(trace))
    rgb = get_rgb_cycle(l)
    if draw_points:
        ax.scatter(trace[:l,0], trace[:l,1], trace[:l,2], 'o',c=rgb)
    ax.plot(trace[:l,0], trace[:l,1], trace[:l,2],label=label)


def get_rgb_cycle(length):
    phi = np.linspace(0, 2*np.pi, length)
    rgb_cycle = np.vstack((            # Three sinusoids
        .5*(1.+np.cos(phi          )), # scaled to [0,1]
        .5*(1.+np.cos(phi+2*np.pi/3)), # 120° phase shifted.
        .5*(1.+np.cos(phi-2*np.pi/3)))).T # Shape = (60,3)
    return rgb_cycle

def draw_spline(tck,u, max_time, draw_points = False, ax = None):
    '''
    tck, u is for one ts
    '''
    if ax is None:
        fig = plt.figure(figsize=(16, 16), dpi=80)
        ax = fig.add_subplot(1,1,1,projection='3d')
    new_points = splev(u, tck)
    max_time = min(len(new_points[0]),max_time)
    rgb = get_rgb_cycle(max_time)
    ax.plot(new_points[0][:max_time],new_points[1][:max_time],new_points[2][:max_time])
    if draw_points:
        ax.scatter(new_points[0][:max_time], new_points[1][:max_time], new_points[2][:max_time], '.',c=rgb)
    return new_points


def save_tcku(fname,tck,u):
    t,c,k = zip(*tck)
    t = np.array(t)
    c = np.array(c).transpose(0,2,1)
    save_dict = {
        't': t,
        'c': c,
        'k': k[0],
        'u': u,
    }
    np.save(fname,save_dict)

def dict_to_tcku(save_dict, transposed=False):
    t = save_dict['t']
    if transposed:
        t = t.T
        c = save_dict['c'].transpose(1,2,0)
    else:
        c = save_dict['c'].transpose(0,2,1)
    if t.shape[1] == c.shape[2]:
        t = pad_knot(t)
    k = [save_dict['k']] * len(t)
    u = save_dict['u']
    tck = tuple(zip(t,c,k))
    return tck, u

def array_to_tcku(array, num_u=5000, k=3):
    T, N, D = array.shape
    knot = array[:,:,-1].T
    knot_full = pad_knot(knot)
    cp = array[:,:,:D-1].transpose(1,2,0)
    k = np.array([k] * N)
    tck = tuple(zip(knot_full, cp, k))
    u = np.linspace(0, 1, num_u)
    u = np.array([u] * N)
    return tck, u


def add_time_dim(tra):
    time_index = np.arange(0,tra.shape[1],1).astype(np.float32)
    time_index = np.tile(time_index[None,:,None], (tra.shape[0],1,1))
    time_index = (time_index/(tra.shape[1]-1) - 0.5) * tra.max() * 2
    tra = np.concatenate([tra,time_index],axis=-1)
    return tra

def harm_to_xyz(tra:np.array):
    '''
    Directly convert harm coord to xyz (treat as spherical)
    '''
    r = tra[:,:,0]
    cos_theta = (tra[:,:,1] * 2) -1
    theta = np.arccos(cos_theta)
    # theta = (-tra[:,:,1] + 1) * math.pi
    # cos_theta = np.cos(theta)
    phi = tra[:,:,2]
    z = r * cos_theta
    x = r * np.cos(phi) * cos_theta
    y = r * np.sin(phi) * np.sin(theta)
    return np.stack([x,y,z],-1)

def thJ_of_X(Xharm):
    poly_alpha = 14
    poly_xt = 0.82
    poly_norm = 0.5 * math.pi * 1. / (1. + 1. / (poly_alpha + 1.) * 1. / poly_xt ** poly_alpha)
    y = 2 * Xharm[1] - 1
    thJ = poly_norm * y * (1 + (y / poly_xt) ** poly_alpha / (poly_alpha + 1.)) + 0.5 * math.pi
    return thJ

def thG_of_X(Xharm:np.array, hslope):
    return math.pi * Xharm[1] + ((1-hslope)/2) * math.sin(2 * math.pi * Xharm[1])

def harm2bl(Xharm:np.array, hslope=0.3):
    Xbl = np.zeros_like(Xharm)
    Xbl[2] = Xharm[2]
    Xbl[0] = math.exp(Xharm[0])

    thg = thG_of_X(Xharm,hslope)
    thj = thJ_of_X(Xharm)
    mks_smooth = 0.3
    Rout = 20.
    N1TOT = 256
    Reh = 0.9
    Rin = math.exp((N1TOT * math.log(Reh) / 5.5 - math.log(Rout)) / (-1. + N1TOT / 5.5));
    startx = [math.log(Rin)]
    th = thg + math.exp(mks_smooth * (startx[0] - Xharm[0])) * (thj-thg)
    Xbl[1] = th

    return Xbl

def bl2cart(Xbl:np.array):
    Xcart = np.zeros_like(Xbl)
    r = Xbl[0]
    th = Xbl[1]
    ph = Xbl[2]
    Xcart[0] = r * math.sin(th) * math.cos(ph)
    Xcart[1] = r * math.sin(th) * math.sin(ph)
    Xcart[2] = r * math.cos(th)
    return Xcart


def harm2cart(Xharm:np.array):
    Xbl = harm2bl(Xharm)
    Xcart = bl2cart(Xbl)
    return Xcart

def get_raw_coords_mass(f):
    with h5py.File(f,'r') as f:
        if 'Step#0' in f.keys():
            Xharm = f['Step#0/Xharm'][()]
            Xcart = f['Step#0/Xcart'][()]
            id = f['Step#0/id'][()]
            mass = f['Step#0/mass'][()]
        else:
            Xharm = f['Xharm'][()]
            Xcart = f['Xcart'][()]
            id = f['id'][()]
            mass = f['Step#0/mass'][()]
    return np.concatenate([Xharm,Xcart],axis=-1), id, mass

def harm_to_ijk(h,X1,X2,X3):
    '''
    convert the harm coordinates to grid position
    '''
    i = next(x for x, val in enumerate(X1)
                                  if val > h[0])
    j = next(x for x, val in enumerate(X2)
                                  if val > h[1])
    k = next(x for x, val in enumerate(X3)
                                  if val > h[2])
    return (i,j,k)





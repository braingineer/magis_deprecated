import pickle
import pprint
from math import *
import collections
import numpy as np
from scipy.stats import norm, beta
import os
import scipy.special as scispec
import time
import magis


## TODO: profile this against the itertools.tee version
def unzip(xys):
    return [[x[i] for x in xys] for i in range(len(xys[0]))]


def make_kth_slice(data,k,k_size):
    """
    Input: data matrix, the current slice index and the size of each slice,
    Outpt: Data minus kth slice, kth slice of data
    """
    indices = np.ones(len(data),dtype=bool)
    indices[k*k_size:(k+1)*k_size]=np.zeros(k_size,dtype=bool)
    kth_slice = np.zeros(len(data),dtype=bool)
    kth_slice[k*k_size:(k+1)*k_size] = np.ones(k_size,dtype=bool)
    #print kth_slice,data.shape,indices
    return data[indices],data[kth_slice]
    

def prettyprint(items):
    """
    Prefers lists or dicts
    """
    pp = pprint.PrettyPrinter(indent=3)
    print(pp.pformat(items))

def prettyformat(items):
    """
    Prefers lists or dicts
    """
    pp = pprint.PrettyPrinter(indent=3)


def binx(x, nb_bins, x_max):
        """
        One of my oldest utility functions. I kind of have a soft spot for it.
        Input: x, the current value
               i, the max bin number
               t, the upper bound on x's support set (or max(data))
        """
        if x == x_max:
            return i-1
        elif x == 0.0:
            return 0
        return int(floor(1.0*(x)*nb_bins/x_max))


def progress_bar(*args, **kwargs):
    ### TODO make these accept same arguments
    try:
        from tqdm import tqdm
        return tqdm(*args, **kwargs)
    except:
        return SimplePreogress(*args, **kwargs)

class SimpleProgress:
    def __init__(self, total):
        self.total = total
    
    def start_progress(self):
        self.start_time = time.time()
        
    def update(self, x):
        if x>0:
            elapsed = time.time()-self.start_time
            percDone = x*100.0/self.total
            estimatedTimeInSec=(elapsed*1.0/x)*self.total
            return "%s %s percent\n%s Processed\nElapsed time: %s\nEstimated time: %s\n--------" % (self.bar(percDone), round(percDone, 2), x, self.form(elapsed), self.form(estimatedTimeInSec))
        return ""
    
    def expiring(self):
        elapsed = time.time()-self.start_time
        return elapsed/(60.0**2) > 71.
    
    def form(self, t):
        hour = int(t/(60.0*60.0))
        minute = int(t/60.0 - hour*60)
        sec = int(t-minute*60-hour*3600)
        return "%s Hours, %s Minutes, %s Seconds" % (hour, minute, sec)
        
    def bar(self, perc):
        done = int(round(30*(perc/100.0)))
        left = 30-done
        return "[%s%s]" % ('|'*done, ':'*left)

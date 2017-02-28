from scipy.special import gdtrc 
from math import atan2, sin, cos, pi
import numpy as np
cimport numpy as np

cdef class Bound:
    cdef double rate
    cdef double shape
    cdef double origin
    
    def __init__(self, double rate, double shape, double origin):
        self.shape = shape
        self.rate = rate
        self.origin = origin 
        
    @classmethod
    def from_scale(cls, double scale, double shape, double origin):
        return cls(rate=1.0/scale, shape=shape, origin=origin)
                
    cpdef phi(self, double x):
        return gdtrc(self.rate, self.shape, x)

cdef class LeftBound(Bound):
    '''
    Compute the gamma survival function for a bound on the left side. 
         
  value of x __________ in boundary
         |  /
         | /
         |/
         |
        /|
       / |
    -----x----|origin|

    so this is testing for x > origin.
        aka. true/1.0 when larger than origin.  
    '''
    def __call__(self, double x):
        if x >= self.origin:
            return 1.0
        else:
            return self.phi(self.origin - x)


cdef class RightBound(Bound):
    '''
    Compute the gamma survival function for a bound on the right side. 

    1.0|_________  Value of x in boundary
       |         \ |
       |          \| 
       |           |\
       |           | \
    0.0|-|origin|--x----------

    so this is testing for x < origin. 
    '''


    def __call__(self, x):
        if x <= self.origin:
            return 1.0
        else:
            return self.phi(x - self.origin)
    
cdef class DualBoundaries:
    '''
    Two boundaries, one right and one left. 
    '''
    cdef Bound left
    cdef Bound right
    cdef double left_origin, right_origin
    cdef bint vectorized

    def __init__(self, LeftBound left, RightBound right): 
        self.left = left
        self.left_origin = left.origin
        self.right = right
        self.right_origin = right.origin
        self.vectorized = 1
        
    @classmethod
    def from_parameters(cls, double left_scale, double left_shape, double left_origin, 
                             double right_scale, double right_shape, double right_origin):
        return cls(LeftBound.from_scale(left_scale, left_shape, left_origin), 
                   RightBound.from_scale(right_scale, right_shape, right_origin))
    
    def __call__(self, *args):
        try:
            if self.vectorized:
                return self.__ndarray_call__(*args)
            else:
                return self.__scalar_call__(*args)
        except TypeError:
            self.toggle()
            if self.vectorized:
                self.toggle()
                return self.__ndarray_call__(*args)
            else:
                self.toggle()
                return self.__scalar_call__(*args)
            
    def toggle(self):
        if self.vectorized:
            self.vectorized = 0
        else:
            self.vectorized = 1
    
    def __scalar_call__(self, double x):
        if x < self.left_origin:
            return self.left.phi(self.left_origin - x)
        elif x > self.right_origin:
            return self.right.phi(x - self.right_origin)
        else:
            return 1.0
    
    def __ndarray_call__(self, np.ndarray[double, ndim=1] x):
        cdef: 
            Py_ssize_t i = 0
            Py_ssize_t n = x.shape[0]
            double x_i = 0
            double l_origin = self.left_origin
            double r_origin = self.right_origin
            np.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
        for i in range(n):
            x_i = x[i]
            if x_i < l_origin:
                out[i] = self.left.phi(l_origin - x_i)
            elif x_i > r_origin:
                out[i] = self.right.phi(x_i - r_origin)
            else:
                out[i] = 1.0
        return out 
    
cdef class CircularBoundaries(DualBoundaries):
    cdef bint adjust_coords
    
    def __init__(self, *args, adjust_coords=0): 
        super(CircularBoundaries, self).__init__(*args)
        self.adjust_coords = adjust_coords
        
    @classmethod
    def from_parameters(cls, double left_scale, double left_shape, double left_origin, 
                             double right_scale, double right_shape, double right_origin, 
                             bint adjust_coords):
        return cls(LeftBound(left_scale, left_shape, left_origin), 
                   RightBound(right_scale, right_shape, right_origin), adjust_coords=adjust_coords)

    def __scalar_call__(self, double x):
        if self.adjust_coords:
            x = atan2(sin(x*pi/180), cos(x*pi/180))*180/pi
        if x < self.left_origin:
            return self.left.phi(self.left_origin - x)
        elif x > self.right_origin:
            return self.right.phi(x - self.right_origin)
        else:
            return 1.0
    
    def __ndarray_call__(self, np.ndarray[double, ndim=1] x):
        cdef: 
            Py_ssize_t i = 0
            Py_ssize_t n = x.shape[0]
            double x_i = 0
            double l_origin = self.left_origin
            double r_origin = self.right_origin
            np.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    
        if self.adjust_coords:
            x = np.arctan2(np.sin(x*np.pi/180.0), np.cos(x*np.pi/180.0))*180.0/np.pi
            
        for i in range(n):
            x_i = x[i]
            if x_i < l_origin:
                out[i] = self.left.phi(l_origin - x_i)
            elif x_i > r_origin:
                out[i] = self.right.phi(x_i - r_origin)
            else:
                out[i] = 1.0
        return out 

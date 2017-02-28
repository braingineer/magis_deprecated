"""
Defines the standard functions and properties of

"""
import logging, os, operator

from math import sin, cos, atan2, pi
from collections import OrderedDict
import json
import numpy as np

class Model(object):
    '''The abstract class for a grounded semantics model
    
    Model's Contract: 
        - constructor accepts name and already made components
        - classmethods accept the class of their components to instantiate
        - each component is a category/class/hypothesis
        - the model computes P(category | input)
    '''
    def __init__(self, name, components=[], graceful_failure=False):
        ''' instantiate this model with the specified name and components '''
        self.name = name
        self.components = components
        self._lookup = {c.name:c for c in components}
        self.graceful_failure = graceful_failure

    @classmethod
    def from_json(cls, filename, ComponentClass):
        ''' accept filename and component class for insantiating 
            
            json should be in the form of:
                {
                'name': name, 
                'components': {
                              'name': component_name, 
                              'parameters': parameters
                              }
                }
        '''
        with open(filename) as fp:
            info = json.load(fp)
            components = list(map(ComponentClass.from_dict, info['components']))
            for i, c in enumerate(components):
                c.index = i
            return cls(info['name'], components)
    
    def __contains__(self, k):
        ''' test for membership '''
        return k in self._lookup

    def __getitem__(self, k):
        if k in self:
            return self._lookup[k]
        if self.graceful_failure:
            self.logger.warning("[-][OOV][{} not in {}]".format(k, self.name))
            return None
        raise OutOfVocabularyException

    def __len__(self):
        return len(self.components)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if len(self) == 0:
            err_string = "Load with Model.from_json(filename) or .pretrained() instead"
        else:
            err_string = ""
        return "<Model>{}; {} components; {}".format(self.name, len(self), err_string)

    def predict(self, *datum):
        '''
        return component with highest probability
            e.g. argmax_component P(component, datum)
        '''
        if len(datum) == 1 and isinstance(datum[0], (list,tuple)):
            datum = datum[0]
        p_vec = np.array([component(datum) for component in self.components])
        try:
            return [self.components[i] for i in p_vec.argmax(axis=0)]
        except TypeError as e:  
            return self.components[p_vec.argmax()]
    
    def likelihood(self, datum, component_name):
        '''
        return probability of the component given the datum 
            e.g. P(component | datum)
        '''
        assert self[component_name]
        p_vec = self.posterior(datum)
        return p_vec[self._lookup[component_name].index]

    def posterior(self, *datum): 
        if len(datum) == 1 and isinstance(datum[0], (list,tuple)):
            datum = datum[0]
        p_vec = np.array([component(datum) for component in self.components])
        try:
            p_vec /= p_vec.sum(axis=0, keepdims=True)
        except TypeError as e:
            p_vec /= p_vec.sum()
        return Distribution([c.name for c in self.components], p_vec)
        # if neat:
        #     return {c.name:p for c,p in zip(self.components, p_vec)}
        # else:
        #     return p_vec
    
class Component(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        ## this plays well with mixins
        try:
            super(Component, self).__init__(*args, **kwargs)
        except TypeError as te:
            ''
        
    @classmethod
    def from_dict(cls, info):
        return cls(info['name'], info)
        
    @property
    def prior(self):
        raise NotImplementedError 
        
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __call__(self, *args):
        return self.pdf(*args)*self.prior

    def pdf(self, x):
        raise NotImplementedError
        
class Distribution(object):
    def __init__(self, names, numbers):
        self.names = set(names)
        self.numbers = numbers
        self.lookup = dict(zip(names, numbers))
        self.sortd = sorted(self.lookup.items(), key=lambda x: x[1], reverse=True)
        
    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self.names:
            return self.lookup[key]
        elif 'top' == key[:3] and key[3:].isdigit():
            return dict(self.sortd[:int(key[3:])])
        else:
            return self.lookup
                
                
    def __getitem__(self, key):
        return self.lookup[key]

class OutOfVocabularyException(Exception):
    pass


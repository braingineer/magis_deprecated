from nose.tools import assert_equal, assert_not_equal, assert_raises, raises

import magis
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))

class TestModel(magis.models.abstract.Model):
    pass

class TestComponent(magis.models.abstract.Component):
    @property
    def prior(self):
        return 1.0

    def pdf(self, x):
        return 1.0

def test():
        mod = TestModel.from_json(os.path.join(HERE,'abstract_model.json'), TestComponent)
        assert_equal(mod.name, 'TestModel')
        assert_equal(len(mod.components), 1)
        assert_equal(mod.predict(0), mod.components[0])
        assert_equal(mod.likelihood(0, 'TestComponent'), 1.0)
        assert_equal(mod.posterior(0), np.array([1.0]))

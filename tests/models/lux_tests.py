from magis.models import Lux

import numpy as np

from nose.tools import assert_equal, assert_not_equal, assert_raises, raises

def test():
    lux = Lux.pretrained()
    d = (200,100,100)
    lux.predict(d)
    lux.likelihood(d, 'blue')
    lux.posterior(d)

    d2 = np.empty((50,3))
    d2[:,0] = np.arange(150,200)
    d2[:,1] = 70
    d2[:,2] = 70

    lux.predict(d2)
    lux.likelihood(d2, 'blue')
    lux.posterior(d2)

test()
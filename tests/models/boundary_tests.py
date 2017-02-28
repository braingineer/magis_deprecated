from nose.tools import assert_equal, assert_not_equal, assert_raises, raises

import numpy as np

import magis
from magis.models.abstract import LeftBound, RightBound, DualBoundaries, CircularBoundaries

def test():
    lb = LeftBound.from_scale(1.0, 1.0, 2.0)
    rb = RightBound.from_scale(1.0, 1.0, 2.0)
    assert_equal(lb(3.0), 1.0)
    assert_equal(rb(1.0), 1.0)
    assert_equal(lb(1.0), rb(3.0))

    db = DualBoundaries.from_parameters(1.0, 1.0, 2.0, 1.0, 1.0, 3.0)
    assert_equal(db(2.5), 1.0)
    assert_equal(db(1.0), rb(3.0))
    assert_equal(db(1.0), lb(1.0))
    assert_equal(db(4.0), rb(3.0))
    assert_equal(db(4.0), lb(1.0))
    assert_not_equal(db(100.0), 0.0)

    cb = CircularBoundaries.from_parameters(1.0, 1.0, -10, 1.0, 1.0, 5.0, True)
    # left in negative
    assert_equal(np.allclose(cb(-11), lb(1.0)), True)
    assert_equal(np.allclose(cb(-11), rb(3.0)), True)
    assert_equal(np.allclose(cb(-11), db(1.0)), True)
    assert_equal(np.allclose(cb(-11), db(4.0)), True)
    # right side 
    assert_equal(np.allclose(cb(6.0), lb(1.0)), True)
    assert_equal(np.allclose(cb(6.0), rb(3.0)), True)
    assert_equal(np.allclose(cb(6.0), db(1.0)), True)
    assert_equal(np.allclose(cb(6.0), db(4.0)), True)

    # now for the circular part
    assert_equal(np.allclose(cb(349), lb(1.0)), True)
    assert_equal(np.allclose(cb(349), rb(3.0)), True)
    assert_equal(np.allclose(cb(349), db(1.0)), True)
    assert_equal(np.allclose(cb(349), db(4.0)), True)



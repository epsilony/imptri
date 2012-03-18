import numpy.random as rd
from nose.tools import ok_
import imptri.tools


def feqok_(a,b,tol,msg=None):
    if msg is None:
        msg="%r != %r by tolerance %r" % (a,b,tol)
    return ok_(feq(a,b,tol),msg)


def feq(a,b,tol):
    return abs(a-b)<abs(tol)


def random_float(mag_range):
    return rd.random()*cmp(rd.randint(0,2),0.5)*mag_range


def random_vec(mag_range, dim=3):
    return rd.random(dim)*random_float(mag_range)

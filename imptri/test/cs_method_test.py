#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rd
from imptri.cs_method import rotation_array, norm_vec, vec_len, intersec_angle, projection_to_plan
from nose.tools import ok_, eq_
from math import pi,sqrt,acos

def test_vec_len():
    vec=np.array((1,3,2))
    expected=sqrt(1+9+4)
    result=vec_len(vec)
    eq_(result,expected)

def test_norm_vector():
    while True:
        vec=_random_vec(100)
        if vec_len(vec) !=1:
            break
    normed_vec=norm_vec(vec)
    feqok_(np.dot(normed_vec,normed_vec),1,1e-6)

def intersec_angle_test():
    a=np.array((1,3,4))
    b=np.array((2,6,5))
    
    tol=1e-3
    exp=0.2328
    print intersec_angle(a,b),'test intersec_angle'
    feqok_(exp,intersec_angle(a,b),tol)

def test_rotation_matrix():
    sample_count=10
    mag_range=10000
    for i in xrange(10):
        axis_dir=_random_vec(mag_range)
        vec_to_rotate=_random_vec(mag_range)
        theta=_random_float(pi)
        mat=rotation_array(theta,axis_dir)
        rotated_vec=mat.dot(vec_to_rotate)
        
        act_theta=intersec_angle(projection_to_plan(vec_to_rotate,axis_dir),projection_to_plan(rotated_vec,axis_dir))
        act_theta*=cmp(np.dot(np.cross(vec_to_rotate,rotated_vec),axis_dir),0)
        
        feqok_(act_theta,theta,1e-10)
    

def _random_vec(mag_range):
    return rd.random(3)*_random_float(mag_range)

def _random_float(mag_range):
    return rd.random()*cmp(rd.randint(0,2),0.5)*mag_range

def _feq(a,b,tol):
    return abs(a-b)<abs(tol)

def feqok_(a,b,tol,msg=None):
    if msg is None:
        msg="%r != %r by tolerance %r" % (a,b,tol)
    return ok_(_feq(a,b,tol),msg)


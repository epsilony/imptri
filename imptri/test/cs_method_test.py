#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rd
from nose.tools import ok_, eq_
from math import pi,sqrt,acos

from nose.tools import set_trace

from imptri.tools import vec_len, norm_vec, intersec_angle, circum, \
    rotation_array, projection_to_plan, smallest_triangle_sphere
from imptri.test_tools import feqok_, random_vec, random_float

def test_vec_len():
    vec=np.array((1,3,2))
    expected=sqrt(1+9+4)
    result=vec_len(vec)
    eq_(result,expected)

def test_norm_vec():
    while True:
        vec=random_vec(100)
        if vec_len(vec) !=1:
            break
    normed_vec=norm_vec(vec)
    feqok_(np.dot(normed_vec,normed_vec),1,1e-6)

def test_intersec_angle():
    a=np.array((1,3,4))
    b=np.array((2,6,5))
    
    tol=1e-3
    exp=0.2328
    feqok_(exp,intersec_angle(a,b),tol)


def test_circum():
    vec_len_range=10000
    sample_count=1000
    tol=1e-6
    for i in xrange(sample_count):
        p1=random_vec(vec_len_range)
        p2=random_vec(vec_len_range)
        p3=random_vec(vec_len_range)
        (r,center)=circum(p1,p2,p3)
        err=max(vec_len(p1-p2),vec_len(p2-p3),vec_len(p3-p1))*tol
        feqok_(vec_len(p1-center),r,err)
        feqok_(vec_len(p2-center),r,err)
        feqok_(vec_len(p3-center),r,err)
        feqok_((p1-center).dot(np.cross(p2-center,p3-center))
                                 ,0,(tol*vec_len(np.cross(p2-center,p3-center)) 
                                     * vec_len(p1-center)))


def test_smallest_triangle_sphere():
    
    #this test is based on test_circum
    p1=np.array((0,0,0))
    p2=np.array((3,0,0))
    p3=np.array((0,4,0))
    exp_r=2.5
    exp_center=(1.5,2,0)
    tol=1e-10
    (r,center)=smallest_triangle_sphere(p1,p2,p3)
    feqok_(exp_r,r,tol)
    iter_=iter(exp_center)
    for item in center:
        feqok_(next(iter_),item,tol)
    
    p1=np.array((0,0,0))
    p2=np.array((3,-1,0))
    p3=np.array((0,4,0))
    exp_r=sqrt(34)/2
    exp_center=(1.5,1.5,0)
    tol=1e-10
    (r,center)=smallest_triangle_sphere(p1,p2,p3)
    feqok_(exp_r,r,tol)
    iter_=iter(exp_center)
    for item in center:
        feqok_(next(iter_),item,tol)
    
    

def test_rotation_matrix():
    sample_count=10
    vec_len_range=10000
    theta_range=pi    # if > pi, may cause test fail, but don't worry.
    tol = 1e-4
    for i in xrange(sample_count):
        axis_dir=random_vec(vec_len_range)
        vec_to_rotate=random_vec(vec_len_range)
        theta=random_float(theta_range)
        mat=rotation_array(theta,axis_dir)
        rotated_vec=mat.dot(vec_to_rotate)
        rotated_vec2=mat.dot(rotated_vec)
        
        act_theta=intersec_angle(
            projection_to_plan(vec_to_rotate,axis_dir),
            projection_to_plan(rotated_vec,axis_dir))
        act_theta*=cmp(np.dot(np.cross(vec_to_rotate,rotated_vec),axis_dir),0)
        
        feqok_(act_theta,theta,tol)
        if theta < pi*tol:
            continue
        feqok_(vec_len(vec_to_rotate),vec_len(rotated_vec),tol)
        (r,center)=circum(vec_to_rotate,rotated_vec,rotated_vec2)
        feqok_(vec_len(np.cross(center,axis_dir)),0,tol*max(vec_len(axis_dir),vec_len(center)))
        ok_(cmp(np.dot(center,axis_dir),0) == \
                cmp(np.dot(vec_to_rotate,axis_dir),0))

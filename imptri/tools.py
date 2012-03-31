#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import sqrt, cos, sin, acos, pi
import numpy as np


def is_circle_cross_segment(
    center,
    r,
    p1,
    p2,
    ):
    v12 = p2 - p1
    vc1 = p1 - center
    if vec_len(vc1) <= r or vec_len(p2 - center) <= r:
        return True

   # parameter for the center's neareat point on line (p2-p1)*t+p1

    t = v12.dot(vc1) / v12.dot(v12)
    if t < 0 or t > 1:
        return False
    elif vec_len(p1 * (1 - t) + p2 * t - center) <= r:
        return True
    else:
        return False

def rb_acos(v):
    if v > 1.000001 or v < -1.000001:
        raise ValueError()
    if v>1:
        return 0
    elif v<1:
        return pi
    else:
        return acos(v)

def intersec_angle(vec1, vec2,normed=False):
    if normed:
        return acos(np.dot(vec1,vec2))
    else:
        return acos(np.dot(vec1, vec2) / vec_len(vec1) / vec_len(vec2))


def rotation_array(theta, u):
    """ 
    Returns an array type rotation matrix
    
    Rotation matrix from an axis and angle, the matrix for a rotation by an 
    angle of theta about an axis in the direction u

    How to use:
    >>>m=rotation_array(theta,u)
    >>>v_rotated=m.dot(v_to_be_rotated)
    """

    # SEE: http://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions

    u = norm_vec(u)
    cosvl = cos(theta)
    sinvl = sin(theta)
    mat = np.diag((cosvl, cosvl, cosvl)) + sinvl * np.array(((0, -u[2],
            u[1]), (u[2], 0, -u[0]), (-u[1], u[0], 0))) + (1 - cosvl) \
        * np.outer(u, u)
    return mat


def projection_to_plan(vec, plan_normal):
    plan_normal=norm_vec(plan_normal)
    return vec - np.dot(vec, plan_normal) * plan_normal


def vec_len(vec):
    """ Get the length of vec (Euclid norm of real vectors) """

    return sqrt(np.dot(vec, vec))


def norm_vec(vec):
    """ Get the normalized vector of vec  """

    return vec / sqrt(np.dot(vec, vec))


def circum(pa, pb, pc):
    va = pb - pc
    a = vec_len(va)
    vb = pc - pa
    b = vec_len(vb)
    vc = pa - pb
    c = vec_len(vc)
    area_4_times = sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a
                        + b - c))
    r = a * b * c / area_4_times

    # barycentric coordinates

    bc = np.array((a * a * np.dot(vc, -vb), b * b * np.dot(-vc, va), c
                  * c * np.dot(vb, -va))) / (area_4_times
            * area_4_times * 0.5)
    center = bc[0] * pa + bc[1] * pb + bc[2] * pc
    return (r, center)


def smallest_triangle_sphere(pa, pb, pc):

   # if find bugs here must examine the circum

    va = pb - pc
    a = vec_len(va)
    vb = pc - pa
    b = vec_len(vb)
    vc = pa - pb
    c = vec_len(vc)
    tt = [a, b, c]
    max_abc = max(tt)
    tt.remove(max_abc)
    if max_abc * max_abc >= np.dot(tt, tt):
        r = max_abc / 2.0
        tt = (a, b, c)
        i = tt.index(max_abc)
        center = (pb + pc, pa + pc, pa + pb)[i] / 2.0
        return (r, center)
    area_4_times = sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a
                        + b - c))
    r = a * b * c / area_4_times

    # barycentric coordinates

    bc = np.array((a * a * np.dot(vc, -vb), b * b * np.dot(-vc, va), c
                  * c * np.dot(vb, -va))) / (area_4_times
            * area_4_times * 0.5)
    center = bc[0] * pa + bc[1] * pb + bc[2] * pc
    return (r, center)



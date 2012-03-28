from imptri.tools import vec_len,norm_vec
from imptri.test_tools import feqok_
from imptri.cs_method import CSTri
import numpy as np
from numpy import array
from math import pi
from nose.tools import set_trace,ok_,eq_
from mayavi.mlab import triangular_mesh

def test_edge_spining():
    tol=1e-10
    center=array((0,0,1))
    r=1.1
    samp=sphere_CSTri(center=center,r=r)
    v1=array((1,2,3))
    v2=array((1.1,2.3,2.9))
    v3=array((0.7,2.2,3))
    v1=norm_vec(v1)*r
    v2=norm_vec(v2)*r
    v3=norm_vec(v3)*r
    p1=v1+center
    p2=v2+center
    pold=v3+center
    c=0.1
    pnew=samp.edge_spining(p1,p2,np.cross(p1-p2,pold-p2),c)

    feqok_(vec_len(pnew-p1),vec_len(pnew-p2),tol)
    feqok_(np.dot(pnew-(p1+p2)/2,p1-p2),0,tol*vec_len(p1-p2))
    
    samp=sphere_CSTri(center=center,r=r,is_hole=True)
    pnew=samp.edge_spining(p1,p2,np.cross(p1-p2,pold-p2),c)
    feqok_(vec_len(pnew-p1),vec_len(pnew-p2),tol)
    feqok_(np.dot(pnew-(p1+p2)/2,p1-p2),0,tol*vec_len(p1-p2))
    
    c_fail=r*100
    pnew=samp.edge_spining(p1,p2,np.cross(p1-p2,pold-p2),c_fail)


def illustrate():
    cstri=sphere_CSTri()
    cstri.first_trangle(array([0,0,1]),0.1)
    tris=cstri.get_triangles()
    triangular_mesh(*tris)

def elipse_CSTri(a=0.3,b=0.5,c=1,
                 rmax=None,
                 alpha_err=pi/6,
                 delt_alpha=pi/10,
                 line_search_dlt=0.1,
                 is_hole=False):
    if rmax is None:
        rmax=alpha_err*max(a,b,c)*1.2
    return CSTri(lambda x:np.sum(x**2/np.array((a,b,c))**2)-1,
                 lambda x:2*x/np.array((a,b,c)),
                 alpha_err,
                 rmax,
                 delt_alpha,
                 line_search_dlt) 

def sphere_CSTri(
    center=array((0,0,0)),
    r=1,
    rmax=None,
    alpha_err=pi/6,
    delt_alpha=pi/10,
    line_search_dlt=0.1,
    is_hole=False):
    if rmax is None:
        rmax=alpha_err*r*1.2
    return CSTri(sphere_func(center,r,is_hole),
                 sphere_func_gd(center,r,is_hole),
                 alpha_err,
                 rmax,
                 delt_alpha,
                 line_search_dlt) 


def sphere_func(center,r,is_hole=False):
    if is_hole:
        return lambda p:r-vec_len(p-center)
    else:
        return lambda p:vec_len(p-center)-r


def sphere_func_gd(center,r,is_hole=False):
    if is_hole:
        return lambda p:center-p/vec_len(center-p)
    else:
        return lambda p:p-center/vec_len(p-center)

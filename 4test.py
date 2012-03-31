import numpy as np
from numpy import array as ar
from imptri.test.cs_method_test import sphere_CSTri, elipse_CSTri
from collections import deque
from mayavi import mlab
from math import pi


def sphere_wire(center,r):
    pi=np.pi
    dphi=pi/100
    phi=np.arange(0.0,2*pi+0.5*dphi,dphi)
    result=deque()
    #X,Y,Z direction circles
    x=np.zeros_like(phi)
    y=np.cos(phi)*r
    z=np.sin(phi)*r
    result.append((x,y,z))
    
    x=np.cos(phi)*r
    y=np.zeros_like(phi)
    z=np.sin(phi)*r

    result.append((x,y,z))
    x=np.sin(phi)*r
    y=np.cos(phi)*r
    z=np.zeros_like(phi)

    result.append((x,y,z))
    return result
    
def plot_sphere(center,r,color=(1,1,1),representation='wireframe'):
    sw=sphere_wire(center,r)
    for item in sw:
        mlab.plot3d(*item,color=color,representation=representation)

def plotc(cs,before=None,representation='wireframe',color=(0.5,1,1)):
    mlab.clf()
    if None is not before:
        before()
    mlab.triangular_mesh(*cs.get_triangles(),color=color,representation=representation)
    plot_ale(cs)
    if cs.ale:
        plot_edge(cs.ale[-1])
        plot_edge(cs.ale[-1].succ,color=(1,0,1))

def plot_edge(e,color=(0,1,0),representation='wireframe'):
    mlab.plot3d([e.start[0],e.end[0]],[e.start[1],e.end[1]],[e.start[2],e.end[2]],color=color,tube_radius=None)


def plot_eclipse(a,b,c,color=(1,1,1)):
    zs=np.linspace(-c*0.95,c*0.95,10)
    for z0 in zs:
        theta=np.linspace(0,pi*2,num=30)
        r=(1-z0**2/c**2)/(np.cos(theta)**2/a**2+np.sin(theta)**2/b**2)
        r=r**0.5
        mlab.plot3d(r*np.cos(theta),r*np.sin(theta),np.ones_like(r)*z0,color=color,tube_radius=None)

 
def plot_ale(cs):
    for item in cs.ale:
        color=(1,0,0)
        plot_edge(item,color=color)


def sphere_demo():
    center=np.array((0,0,0))
    r=1
    cs=sphere_CSTri(center=center,r=r)
    cs.center=center
    cs.r=r
    start_pt=ar([0,0,1])
    start_c=0.1
    cs.alpha_err=pi/18
    cs.gen_triangles(start_pt,start_c)
    plotc(cs,before=lambda :plot_sphere(center,r))

plotes=lambda :plotc(es,lambda :plot_eclipse(a,b,c))   
   

if __name__ == '__main__':
    (a,b,c)=(0.3,0.5,1)
    es=elipse_CSTri()
    es.alpha_err=pi/6
    es.rmin=0.03
    es.rmax=0.5
    es.first_triangle(np.array([0,0,1]),0.1)
    for i in range(0):
        es.edge_polygon()
    plotes()
    for i in range(0):
        es.edge_polygon()
        plotc(es,lambda :plot_eclipse(a,b,c))

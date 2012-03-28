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

def plot_edge(e,color=(0,1,0)):
    mlab.plot3d([e.start[0],e.end[0]],[e.start[1],e.end[1]],[e.start[2],e.end[2]],color=color)

    
def plot_ale(cs):
    for item in cs.ale:
      s=item.start
      e=item.end
      mlab.plot3d([s[0],e[0]],[s[1],e[1]],[s[2],e[2]],color=(1,0,0))

def shpere_demo():
    center=np.array((0,0,0))
    r=1
    cs=sphere_CSTri(center=center,r=r)
    cs.center=center
    cs.r=r
    start_pt=ar([0,0,1])
    start_c=0.1
    cs.alpha_err=pi/18
    cs.get_polygon(start_pt,start_c)
    plotc(cs,before=lambda :plot_sphere(center,r))

if __name__ == '__main__':
    es=elipse_CSTri()
    es.first_triangle(np.array([0,0,1]),0.1)
    for i in range(195):
        if i==193:
            plotc(es)
            plot_edge(es.ale[-1])
            import ipdb
            ipdb.set_trace()
        es.edge_polygon()
   
   


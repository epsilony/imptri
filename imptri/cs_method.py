#!/usr/bin/python
# -*- coding: utf-8 -*-

"""This module realized the algorithm in reference:
    "Cermak M & Skala V.
    Polygonization of implicit surfaces with sharp features by edge-spinning.
    VISUAL COMPUTER (2005) 21: pp. 252-264." """

from collections import deque
from math import cos, acos, sqrt, pi

import numpy as np
from scipy.optimize import brentq

from imptri.tools import vec_len, norm_vec, intersec_angle, \
    rotation_array, projection_to_plan, smallest_triangle_sphere


class _WingedEdge(object):

    def __init__(
        self,
        start,
        end,
        start_index=None,
        end_index=None,
        tri_pt=None,
        pred=None,
        succ=None,
        ):
        """ not classical winged-edge start & end point, 
        triangle on the left """

        self.start = start
        self.end = end
        self.start_index
        self.end_index
        self.tri_pt
        self.pred = None
        self.succ = None


class CSTri(object):

    def __init__(
        self,
        func,
        func_gd,
        start_pt,
        max_len,
        alpha_err,
        rmax,
        delt_alpha,
        sharp_search_delta,
        k=0.8,
        alpha_lim=8 / 18.0 * pi,
        ):
        """ Must give

        Parameters
        __________
        func : callable function, func(x,y,z) returns a real
        func_gd: func_gd(x,y,z) returns a grad(x',y',z')
        
        give function of implicit surface func=0, func's gradient func_gd,
        start point, max length of polygon's edges, error limit meausred
        by curvature     """

        self.func = func
        self.func_gd = func_gd
        self.start_pt = start_pt
        self.max_len = max_len
        self.alpha_err = alpha_err
        self.ale = deque()
        self.rmax = rmax
        self.rmin = rmax * 0.1
        self.surf_curv_coef = k * sqrt(2 * (1 - cos(alpha_err)))
        self.delt_alpha = delt_alpha
        self.alpha_lim = abs(alpha_lim)
        self.sharp_search_delta = sharp_search_delta
        self.alpha_shape_min = pi / 6
        self.alpha_shape_big = 2 * pi / 3
        self.distance_test_c = 1.3
        self.points = deque()
        self.triangles = deque()

    def get_polygon(self, p0, c0):
        """ another test
       
        returns  a Winged-edge model"""

        self.first_triangle(p0, c0)
        while self.ale:
            self.edge_polygon()
        return (self.points, self.triangles)

    def edge_polygon(self):
        e = self.ale.popleft()
        p1 = e.start
        p2 = e.end
        p_s=(p1+p2)/2.0
        grd_s=norm_vec(self.func_gd(p_s))
        pold = e.tri_pt
        v12 = p2 - p1
        grd_old = np.cross(v12, pold - p2)

        # not like Cermak2005!!!
        c = vec_len(grd_old) / vec_len(v12)

        pnew = self.edge_spining(p1, p2, grd_old, c)
        if pnew is not None and  not self.check_alpha_lim(self.func(p_s),pnew,False):
            pnew = self.sharp_feature_seek(p_s,grd_s,v12,pnew)
        v_succ = e.succ.end - e.end
        v_pred = e.start - e.pred.start
        alpha2 = intersec_angle(v_succ, -v12)
        if np.cross(v12, v_succ).dot(grd_old) > 0:
            alpha2 = 2 * pi - alpha2
        alpha1 = intersec_angle(v12, -v_pred)
        if np.cross(v_pred, v12).dot(grd_old) > 0:
            alpha1 = 2 * pi - alpha1
        alpha = min(alpha1, alpha2)

        # MARK: alpha_n_assuming the pnew made by edge_spining is on the plan P:
        #   P across the mid point of e
        #   P is perpendicular to e

        if pnew is not None:
            alpha_n = intersec_angle(pnew - p1, v12)
            alpha_shape = alpha_n + self.alpha_shape_min
        else:
            alpha_shape = self.alpha_shape_big

        # neibourhood test

        if alpha < alpha_shape:

            # situation a) in Fig8 & Fig9

            if alpha == alpha1:
                case = 1
                (r, center) = smallest_triangle_sphere(p1, p2,
                        e.pred.start)
            else:
                case = 2
                (r, center) = smallest_triangle_sphere(p1, p2,
                        e.succ.end)
        elif pnew is None:
            r = vec_len(v12)
            center = (p1 + p2) * 0.5
            case = 3
        else:
            (r, center) = smallest_triangle_sphere(p1, p2, pnew)
            center = r * self.distance_test_c
            case = 4

        # filet out a NAEL,distance test

        emin = self._distance_test(e, r, center)

        if emin is None:
            if 1 == case:
                self._add_tri_ale_A(e.pred, e)
            elif 2 == case:
                self._add_tri_ale_A(e, e.succ)
            elif 4 == case:
                pnew_index = len(self.points)
                self.points.append(pnew)
                self.triangles.append((pnew_index, e.end_index,
                        e.start_index))
                e_new1 = _WingedEdge(e.start, pnew, e.start_index,
                        pnew_index, e.end)
                e_new2 = _WingedEdge(pnew, e.end, pnew_index,
                        e.end_index, e.start)
                e_new1.succ = e_new2
                e_new1.pred = e.pred
                e_new2.pred = e_new1
                e_new2.succ = e.succ
                e.pred.succ = e_new1
                e.succ.pred = e_new2
                self.ale.append(e_new1)
                self.ale.append(e_new2)
        else:
            emin1 = emin
            emin2 = emin.succ
            alpha_m1 = intersec_angle(emin2.end - emin2.start, e.start
                    - emin2.start)
            alpha_m2 = intersec_angle(emin1.start - emin1.end, e.end
                    - emin1.end)
            if alpha_m1 < alpha_m2:
                te = _WingedEdge(e.start, emin1.end)
                exceptions = (emin1, emin2, e.pred)
                (r, center) = smallest_triangle_sphere(e.start,
                        emin2.start, emin2.end)
            else:
                te = _WingedEdge(emin1.start, e.end)
                exceptions = (emin1, emin1.pred)
                (r, center) = smallest_triangle_sphere(e.end,
                        emin1.start, emin1.end)
            t_emin = self._distance_test(te, r, center, exceptions)

            if t_emin is not None:
                self.ale.append(e)
                return

            if alpha_m1 < alpha_m2:
                self.ale.remove(emin2)
                enew1 = _WingedEdge(
                    e.start,
                    emin2.end,
                    e.start_index,
                    emin2.end_index,
                    emin2.start,
                    e.pred,
                    emin2.succ,
                    )
                enew2 = _WingedEdge(
                    emin2.start,
                    e.end,
                    emin2.start_index,
                    e.end_index,
                    e.start,
                    emin1,
                    e.succ,
                    )
                e.pred.succ = enew1
                emin2.succ.pred = enew1
                emin1.succ = enew2
                e.succ.pred = enew2
                self.triangles.append(emin2.end_index,
                        emin2.start_index, e.start_index)
            else:
                self.ale.remove(emin1)
                enew1 = _WingedEdge(
                    emin1.start,
                    e.end,
                    emin1.start_index,
                    e.end_index,
                    emin1.end,
                    emin1.pred,
                    e.succ,
                    )
                enew2 = _WingedEdge(
                    e.start,
                    emin1.end,
                    e.start_index,
                    emin1.end_index,
                    e.end,
                    e.pred,
                    emin2,
                    )
                emin1.pred.succ = enew1
                e.succ.pred = enew1
                e.pred.succ = enew2
                emin2.pred = enew2
                self.triangles.append(emin1.end_index,
                        emin1.start_index, e.end_index)
            self.triangles.append(emin2.start_index, e.end_index,
                                  e.start_index)
            self.ale.append(enew1)
            self.ale.append(enew2)

            pass

    def _distance_test(
        self,
        e,
        r,
        center,
        exceptions=None,
        ):

        e_grd = (self.func_gd(e.start) + self.func_gd(e.end)) / 2.0
        (min_dis, emin) = (None, None)
        if exceptions is None:
            exceptions = (e.pred, e.pred.pred, e.succ)
        for item in self.ale:
            pt = item.end
            dis = vec_len(pt - center)
            if item not in exceptions and dis < center and (min_dis
                    is None or dis < min_dis) and np.dot(e_grd,
                    self.func(pt)) > 0:
                min_dis = dis
                emin = item
        return emin

    def _add_tri_ale_A(self, e1, e2):
        if self._is_cross_angle_ok(e1.start, e2.end):
            triangle = (e2.end_index, e2.start_index, e1.start_index)
            self.triangles.append(triangle)
            e_new = _WingedEdge(
                e1.start,
                e2.end,
                e1.start_index,
                e2.end_index,
                e1.end,
                e1.pred,
                e2.succ,
                )
            e2.succ.pred = e_new
            e1.pred.succ = e_new
            self.ale.append(e_new)
        else:
            pmid = (e1.start + e2.end) / 2.0

            # MARK: pmid can be other situations

            pmid = self._search_root_on_plane(pmid, e2.end - e1.start)
            pmid_index = len(self.points)
            self.points.append(pmid)
            self.triangles.append((e1.end_index, e1.start_index,
                                  pmid_index))
            self.triangles.append((e2.end_index, e2.start_index,
                                  pmid_index))
            e_new_1 = _WingedEdge(pmid, e2.end, pmid_index,
                                  e2.end_index, e1.end)
            e_new_2 = _WingedEdge(e1.start, pmid, e1.start_index,
                                  pmid_index, e1.end)

            self.ale.append(e_new_1)
            self.ale.append(e_new_2)
            e_new_1.pred = e1.pred
            e1.pred.succ = e_new_1
            e_new_1.succ = e_new_2
            e_new_2.pred = e_new_1
            e_new_2.succ = e2.succ
            e2.succ.pred = e_new_2

    def _is_cross_angle_ok(self, p1, p2):
        grd1 = norm_vec(self.func_gd(p1))
        grd2 = norm_vec(self.func_gd(p2))
        angle = np.dot(grd1, grd2)
        if angle > self.alpha_err:
            return False
        else:
            return True

    def edge_spining(
        self,
        p1,
        p2,
        grd_old,
        c,
        ):
        """ edge spinning method
        
        p1->p2 is the activate edge,
        grd_old is the triangle normal vector, no need with length 1
        c the circle radius for estimation spining circle radius"""

        # MARK: alpha_n_assuming see self.edge_polygon
        # Estimation of the circle radius(S6.1), r2

        p_s = (p1 + p2) / 2.0
        d12 = norm_vec(p2 - p1)
        vinit = norm_vec(np.cross(d12, grd_old))
        pinit = p_s + vinit * c
        grd1 = norm_vec(self.func_gd(p1))
        grd2 = norm_vec(self.func_gd(p2))
        grd_s = norm_vec(self.func_gd(p_s))
        grdinit = norm_vec(self.func_gd(pinit))
        rc1 = vec_len(p1 - pinit) / acos(grd1.dot(grdinit))
        rc2 = vec_len(p2 - pinit) / acos(grd2.dot(grdinit))
        rcs = vec_len(p_s - pinit) / acos(grd_s.dot(grdinit))
        rc = min(rc1, rc2, rcs)
        r2 = rc * self.surf_curv_coef
        if r2 < self.rmin:
            r2 = self.rmin
        elif r2 > self.rmax:
            r2 = self.rmax

        # edge spin root finding (S6.2)
        #   determine init point pnew, init search direction of rotation

        vnew = vinit * r2
        pnew = p_s + vnew
        rot_mat = rotation_array(self.delt_alpha, d12)
        vnew2 = rot_mat.dot(vnew)
        pnew2 = vnew2 + p_s
        func_val2 = self.func(pnew2)
        func_val = self.func(pnew)
        delta = self.delt_alpha
        if abs(func_val2) > abs(func_val):
            delta = -delta
            rot_mat = rotation_array(-self.delt_alpha, d12)
            vnew2 = rot_mat * vnew.reshape(3, 1)
            pnew2 = vnew2 + p_s
            func_val2 = self.func(pnew2)
        alpha = delta

        #   find the sign different interval
        while func_val * func_val2 > 0:
            func_val = func_val2
            vnew2 = rot_mat.dot(vnew2)
            pnew = pnew2
            pnew2 = vnew2 + p_s
            func_val2 = self.func(pnew2)
            alpha += delta
            if abs(alpha) > self.alpha_lim:
                return None

        #   interval search( not bisection search!!!
            # that's different to Cermak2005)

        rslt = brentq(lambda t: self.func(pnew * (1 - t) + pnew2 * t),
                      0, 1)
        pnew = pnew * (1 - rslt) + pnew2 * rslt
        return pnew

    def sharp_feature_seek(self,p_s,grd_s,plan_norm,p_failed):

        # sharp edge situation (S6.3)
        #   algorithm 4,
        #       p_failed: pnew reture by self.edge_spining but failed to pass
        #                 _check_alpha_lim
        pinit=self._sharp_feature_init(p_s,grd_s,plan_norm,p_failed)
        pnew=self._search_root_on_plane(pinit, plan_norm)
        
        # p_failed maybe not satisfy line alpha limit, that's seams reasonable,
        # near a sharp edge the surface normal vector shifts very quickly.
        
        return pnew

    def _sharp_feature_init(self,p_s,grd_s,plan_norm,p_failed):
        
        #       v1:norm vector, points to p_failed', starts from p_s,
        #           alongs the cross line of t1 and circle plan,
        #       v2:norm vector, points to p_failed', starts from p_failed,
        #           alongs the cross line of t2 and circle plan
        #       pinit algorithm is not mentioned in Cermak2005,
        #       the realization below is numerical robust, c.f. the detail
        #       in http://epsilony.net/wiki/ImpTri/SCmethod#p_failed
        grd_failed=self.func_gd(p_failed)
        v1 = norm_vec(np.cross(plan_norm, grd_s))
        v2 = norm_vec(np.cross(grd_failed, plan_norm))
        vns = p_s - p_failed
        c12 = np.dot(v1, v2)
        t = 1.0 - c12 * c12
        tx = np.dot(vns, v1 - c12 * v2) / t
        ty = np.dot(vns, v1 * c12 - v2) / t
        pinit = (p_s + v1 * tx + p_failed + v2 * ty) / 2.0
        return pinit

    def _search_root_on_plane(self, pnew, plane_norm,normed=True):

        # MARK: alpha_n_assuming see also self.edge_polygon
        # in Cermak2005 it is not directly presented
        # that grdnew should in the circle plan,
        # but the Neighborhood test in S7.1 assumed that.
        if not normed:
            plane_norm=norm_vec(plane_norm)
        
        grdnew = self.func_gd(pnew)
        grdnew = projection_to_plan(grdnew,plan_norm)
        func_val = self.func_gd(pnew)
        delta = self.sharp_search_delta * -cmp(func_val, 0)
        pnew2 = pnew + delta * grdnew
        func_val2 = self.func(pnew2)

        while func_val * func_val2 > 0:
            pnew = pnew2
            pnew2 = pnew + delta
            func_val = func_val2
            func_val2 = self.func_gd(pnew2)

        rslt = brentq(lambda t: self.func(pnew * (1 - t) + pnew2 * t),
                      0, 1)
        pnew = pnew * (1 - rslt) + pnew2 * rslt
        return pnew

    def check_alpha_lim(self,grd,pnew,normed=True):
        if normed:
            grdnew = norm_vec(self.func_gd(pnew))
            if intersec_angle(grdnew,grd,True) < self.alpha_lim_edge:
                return True
        else:
            grdnew= self.func_gd(pnew)
            if intersec_angle(grdnew,grd) < self.alpha_lim_edge:
                return True
        return False


    def first_triangle(self, p0, c):
        self.ale.clear()
        self.triangles.clear()
        self.points.clear()

        n = norm_vec(self.func_gd(p0))
        if n[1] > 0.5 or n[0] > 0.5:
            t = np.array((n[1], -n[0], 0))
        else:
            t = np.array((-n[2], 0, n[0]))
        p1 = self.edge_spining(p0, p0 + t, n, c)
        p2 = self.edge_spining(p0, p1, np.cross(p1 - p0, t), c)
        self.points.extend((p0, p1, p2))
        self.triangles.append((0, 2, 1))
        e0 = _WingedEdge(p0, p1, 0, 1, p2)
        e1 = _WingedEdge(p1, p2, 1, 2, p0)
        e2 = _WingedEdge(p2, p0, 2, 0, p1)
        e0.pred = e2
        e0.succ = e1
        e1.pred = e0
        e1.succ = e2
        e2.pred = e1
        e2.succ = e0
        self.ale.extend((e0, e1, e2))

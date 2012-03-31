#!/usr/bin/python
# -*- coding: utf-8 -*-

"""This module realized the algorithm in reference:
    "Cermak M & Skala V.
    Polygonization of implicit surfaces with sharp features by edge-spinning.
    VISUAL COMPUTER (2005) 21: pp. 252-264." """

from collections import deque
from imptri.tools import vec_len, norm_vec, intersec_angle, rotation_array, \
    projection_to_plan, smallest_triangle_sphere, is_circle_cross_segment, \
    rb_acos
from math import cos, sqrt, pi
from scipy.optimize import brentq
import numpy as np




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
        self.start_index = start_index
        self.end_index = end_index
        self.tri_pt = tri_pt
        self.pred = pred
        self.succ = succ


class CSTri(object):

    def __init__(
        self,
        func,
        func_gd,
        alpha_err,
        rmax,
        delt_alpha,
        line_search_dlt,
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
        self._alpha_err = abs(alpha_err)
        self.ale = deque()
        self.rmax = abs(rmax)
        self.rmin = rmax * 0.1
        self.k = k
        self._surf_curv_coef = k * sqrt(2 * (1 - cos(alpha_err)))
        self.dlt_alpha = abs(delt_alpha)
        self.alpha_lim = abs(alpha_lim)
        self.line_search_dlt = line_search_dlt
        self.alpha_shape_min = pi / 6
        self.alpha_shape_big = 2 * pi / 3
        self.dist_test_cf = 1.3
        self.points = deque()
        self.triangles = deque()
        self.max_try = 5
    
    @property
    def alpha_err(self):
        return self._alpha_err
    
    @alpha_err.setter
    def alpha_err(self, value):
        value = abs(value)
        self._alpha_err = value
        self._surf_curv_coef = self.k * sqrt(2 * (1 - cos(value)))

    def gen_triangles(self, p0, c0):
        """ another test
       
        returns  a Winged-edge model"""

        self.first_triangle(p0, c0)
        while self.ale:
            self.edge_polygon()

    def search_on_plane(self, center, radius, plane_norm, vinit):
        pnew = self.spining_search(center, radius, plane_norm, vinit)
        if None is pnew:
            return None
        center_gd = self.func_gd(center)
        for i in xrange(self.max_try):
            pnew_gd = self.func_gd(pnew)
            angle = intersec_angle(center_gd, pnew_gd)
            if angle < self._alpha_err:
                return pnew
            pinit = self.estimate_direct_search_init(center, plane_norm, pnew, center_gd, pnew_gd, angle)
            pnew = self._search_root_on_plane(pinit, plane_norm, False)


    def edge_polygon(self):
        e = self.ale.pop()
        p1 = e.start
        p2 = e.end
        p_s = (p1 + p2) / 2.0
        #grd_s = norm_vec(self.func_gd(p_s))
        pold = e.tri_pt
        v12 = p2 - p1
        grd_old = np.cross(v12, pold - p2)

        # not like Cermak2005!!!
        c0 = vec_len(grd_old) / vec_len(v12)
        vinit = np.cross(v12, grd_old)
        radius = self.estimate_spining_radius(p1, p2, vinit, c0)
        pnew = self.search_on_plane(p_s, radius, v12, vinit)
        
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
            if e.succ.succ is e.pred:
                self.ale.remove(e.succ)
                self.ale.remove(e.pred)
                self.triangles.append((e.end_index, e.start_index, e.pred.start_index))
                return
            # situation a) in Fig8 & Fig9
            if alpha == alpha1:
                case = 1
                (r, center) = smallest_triangle_sphere(p1, p2,
                        e.pred.start)
                except_pts = (p1, p2, e.pred.start)
                check_dir = p2 - e.pred.start
                radius_check_pts = (e.pred.pred.start, e.succ.end)
            else:
                case = 2
                (r, center) = smallest_triangle_sphere(p1, p2,
                        e.succ.end)
                except_pts = (p1, p2, e.succ.end)
                check_dir = e.succ.end - p1
                radius_check_pts = (e.pred.start, e.succ.succ.end)
        elif pnew is None:
            r = vec_len(v12) * 0.5
            center = (p1 + p2) * 0.5
            except_pts = (p1, p2)
            check_dir = e
            radius_check_pts = (e.pred.start, e.succ.end)
            case = 3
        else:
            (r, center) = smallest_triangle_sphere(p1, p2, pnew)
            r = r * self.dist_test_cf
            except_pts = (p1, p2)
            check_dir = e
            radius_check_pts = (e.pred.start, e.succ.end)
            case = 4
            
        # filet out a NAEL,distance test
        emin = self._distance_test(e, r, center, check_dir, except_pts, radius_check_pts)
        if emin is e.pred.pred:
            emin = None
            case = 1
        elif emin is e.succ:
            emin = None
            case = 2
        
        if emin is None:
            if 1 == case:
                self.ale.remove(e.pred)
                self._add_tri_ale_A(e.pred, e)
            elif 2 == case:
                self.ale.remove(e.succ)
                self._add_tri_ale_A(e, e.succ)
            elif 3 == case:
                self.ale.appendleft(e)
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
                te = _WingedEdge(e.start, emin2.end)
                (r, center) = smallest_triangle_sphere(e.start,
                        emin2.start, emin2.end)
                except_pts = (emin2.start, emin2.end, e.start, e.end)
                radius_check_pts = (emin2.succ.end, emin1.start, e.pred.start)
            else:
                te = _WingedEdge(emin1.start, e.end)
                (r, center) = smallest_triangle_sphere(e.end,
                        emin1.start, emin1.end)
                except_pts = (emin1.start, emin1.end, e.start, e.end)
                radius_check_pts = (emin2.end, emin1.pred.start, e.succ.end)
            check_dir = te.end - te.start
            t_emin = self._distance_test(te, r, center, check_dir, except_pts, radius_check_pts)
            
            if t_emin is not None:
                self.ale.appendleft(e)
                return
            
            if alpha_m1 < alpha_m2:
                self.ale.remove(emin2)
                if emin2.succ is e.pred:
                    self.ale.remove(e.pred)
                else:
                    enew1 = _WingedEdge(
                        e.start,
                        emin2.end,
                        e.start_index,
                        emin2.end_index,
                        emin2.start,
                        e.pred,
                        emin2.succ,
                        )
                    e.pred.succ = enew1
                    emin2.succ.pred = enew1
                    self.ale.append(enew1)
                enew2 = _WingedEdge(
                    emin2.start,
                    e.end,
                    emin2.start_index,
                    e.end_index,
                    e.start,
                    emin1,
                    e.succ,
                    )
                emin1.succ = enew2
                e.succ.pred = enew2
                self.ale.append(enew2)
                self.triangles.append((emin2.end_index,
                                      emin2.start_index, e.start_index))
            else:
                self.ale.remove(emin1)
                if e.succ is emin1.pred:
                    self.ale.remove(emin1.pred)
                else:
                    enew1 = _WingedEdge(
                        emin1.start,
                        e.end,
                        emin1.start_index,
                        e.end_index,
                        emin1.end,
                        emin1.pred,
                        e.succ,
                        )
                    emin1.pred.succ = enew1
                    e.succ.pred = enew1
                    self.ale.append(enew1)
                enew2 = _WingedEdge(
                    e.start,
                    emin1.end,
                    e.start_index,
                    emin1.end_index,
                    e.end,
                    e.pred,
                    emin2,
                    )     
                e.pred.succ = enew2
                emin2.pred = enew2
                self.ale.append(enew2)
                self.triangles.append((emin1.end_index,
                        emin1.start_index, e.end_index))
            self.triangles.append((emin2.start_index, e.end_index,
                                  e.start_index))
            pass

    def _distance_test(
        self,
        e,
        r,
        center,
        check_dir,
        except_pts,
        radius_check_pts,
        ):
        ##new code:
        t_ale = deque()
        (min_dis, emin) = (None, None)
        e_mid = (e.start + e.end) * 0.5
        e_grd = 0.5 * (self.func_gd(e.start) + self.func_gd(e.end))
        e_v = e.end - e.start
        for item in self.ale:
            if is_circle_cross_segment(center, r, item.end, item.start):
                t_ale.append(item)
        for item in t_ale:
            for pt in (item.start, item.end):
                tv = pt - e_mid
                if np.cross(e_v, tv).dot(e_grd) < 0 or self.func_gd(pt).dot(e_grd) < 0:
                    continue
                dis = vec_len(tv)
                tb = False
                for i in except_pts:
                    if i is pt:
                        tb = True
                        break
                if tb:
                    continue
                tb = False
                for i in radius_check_pts:
                    if pt is i:
                        tb = True
                        break
                if not tb:
                    if (min_dis is None or dis < min_dis):
                        min_dis = dis
                        if pt is item.start:
                            emin = item.pred
                        else:
                            emin = item
                else:
                    if vec_len(pt - center) < r and (min_dis is None or dis < min_dis):
                        min_dis = dis
                        if pt is item.start:
                            emin = item.pred
                        else:
                            emin = item
        return emin
    
    def _add_tri_ale_A(self, e1, e2):
        if self._is_intersect_angle_ok(e1.start, e2.end):
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
            e_new_2 = _WingedEdge(pmid, e2.end, pmid_index,
                                  e2.end_index, e1.end)
            e_new_1 = _WingedEdge(e1.start, pmid, e1.start_index,
                                  pmid_index, e1.end)

            self.ale.append(e_new_2)
            self.ale.append(e_new_1)
            e_new_1.pred = e1.pred
            e1.pred.succ = e_new_1
            e_new_1.succ = e_new_2
            e_new_2.pred = e_new_1
            e_new_2.succ = e2.succ
            e2.succ.pred = e_new_2

    def _is_intersect_angle_ok(self, p1, p2):
        angle = intersec_angle(self.func_gd(p1), self.func_gd(p2))
        if angle > self.alpha_err:
            return False
        else:
            return True

    def spining_search(
        self,
        center,
        radius,
        plane_norm,
        vinit
        ):
        """ edge spinning method
        
        p1->p2 is the activate edge,
        grd_old is the triangle normal vector, no need with length 1
        c the circle radius for estimation spining circle radius"""

        # MARK: alpha_n_assuming see self.edge_polygon
        # Estimation of the circle radius(S6.1), radius

        # edge spin root finding (S6.2)
        #   determine init point pnew, init search direction of rotation

        vinit = norm_vec(vinit)
        vnew = vinit * radius
        pnew = center + vnew
        rot_mat = rotation_array(self.dlt_alpha, plane_norm)
        vnew2 = rot_mat.dot(vnew)
        pnew2 = vnew2 + center
        func_val2 = self.func(pnew2)
        func_val = self.func(pnew)
        delta = self.dlt_alpha
        if abs(func_val2) > abs(func_val) and func_val2 * func_val > 0 :
            delta = -delta
            rot_mat = rotation_array(-self.dlt_alpha, plane_norm)
            vnew2 = rot_mat.dot(vnew)
            pnew2 = vnew2 + center
            func_val2 = self.func(pnew2)
        alpha = delta

        #   find the sign different interval
        while func_val * func_val2 > 0:
            func_val = func_val2
            vnew2 = rot_mat.dot(vnew2)
            pnew = pnew2
            pnew2 = vnew2 + center
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

    def estimate_spining_radius(self, p2, p1, vinit, c0):
        p_s = (p1 + p2) / 2.0
        vinit = norm_vec(vinit)
        pinit = p_s + vinit * c0
        grdinit = norm_vec(self.func_gd(pinit))
        rcmin = None
        for item in (p1, p2, p_s):
            item_gd = norm_vec(self.func_gd(item))
            t = rb_acos(item_gd.dot(grdinit))
            if t == 0:
                rc = self.rmax
            else:
                rc = vec_len(item - pinit) / t
            if None is rcmin or rc < rcmin:
                rc_min = rc
        rc = rc_min
        r2 = rc * self._surf_curv_coef
        if r2 < self.rmin:
            r2 = self.rmin
        elif r2 > self.rmax:
            r2 = self.rmax
        return r2

    def sharp_feature_seek(self, p_s, grd_s, plan_norm, p_failed):

        # sharp edge situation (S6.3)
        #   algorithm 4,
        #       p_failed: pnew reture by self.spining_search but failed to pass
        #                 _check_alpha_err
        pinit = self.estimate_direct_search_init(p_s, grd_s, plan_norm, p_failed)
        if pinit is not None:
            pnew = self._search_root_on_plane(pinit, plan_norm)
        else:
            return None
        # p_failed maybe not satisfy line alpha_err, that's seams reasonable,
        # near a sharp edge the surface normal vector shifts very quickly.
        
        return pnew

    def estimate_direct_search_init(self, center, plane_norm, p_failed, center_gd, p_failed_gd, failed_angle):
        
        # algorithm from:
        # (x - center) dot center_gd = 0
        # (x - p_failed) dot grd_failed = 0
        # (x - center) dot plane_norm = 0
        a = np.array((center_gd, p_failed_gd, plane_norm))
        b = np.array((np.dot(center, center_gd), np.dot(p_failed, p_failed_gd), np.dot(center, plane_norm)))
        pinit = np.linalg.solve(a, b)
        if np.cross(pinit - center, center_gd).dot(np.cross(p_failed - center, center_gd)) <= 0:
            pinit = None
        
        t = self.k * self._alpha_err / failed_angle
        pinit2 = center * (1 - t) + p_failed * t
        if pinit is None or vec_len(pinit2 - center) > vec_len(pinit - center):
            return pinit2
        else:
            return pinit

    def _search_root_on_plane(self, p_init, plane_norm, normed=True):

        # MARK: alpha_n_assuming see also self.edge_polygon
        # in Cermak2005 it is not directly presented
        # that grd should in the circle plan,
        # but the Neighborhood test in S7.1 assumed that.
        if not normed:
            plane_norm = norm_vec(plane_norm)
        
        grd = self.func_gd(p_init)
        grd = projection_to_plan(grd, plane_norm)
        func_val = self.func(p_init)
        delta = self.line_search_dlt * -cmp(func_val, 0)
        delta = delta * grd
        pnew2 = p_init + delta
        func_val2 = self.func(pnew2)
        pnew = p_init

        while func_val * func_val2 > 0:
            pnew = pnew2
            pnew2 = pnew + delta
            func_val = func_val2
            func_val2 = self.func(pnew2)

        rslt = brentq(lambda t: self.func(pnew * (1 - t) + pnew2 * t),
                      0, 1)
        pnew = pnew * (1 - rslt) + pnew2 * rslt
        return pnew

    def check_alpha_err(self, grd, pnew, normed=True):
        if normed:
            grdnew = norm_vec(self.func_gd(pnew))
            if intersec_angle(grdnew, grd, True) < self._alpha_err:
                return True
        else:
            grdnew = self.func_gd(pnew)
            if intersec_angle(grdnew, grd) < self._alpha_err:
                return True
        return False

    def first_triangle(self, p0, c0):
        self.ale.clear()
        self.triangles.clear()
        self.points.clear()

        n = norm_vec(self.func_gd(p0))
        if n[1] > 0.5 or n[0] > 0.5:
            tg_vec = np.array((n[1], -n[0], 0))
        else:
            tg_vec = np.array((-n[2], 0, n[0]))
        tg_vec = norm_vec(tg_vec)
        vinit = np.cross(tg_vec, n)
        radius = self.estimate_spining_radius(p0 - tg_vec * c0 * 0.5, p0 + tg_vec * c0 * 0.5, vinit, c0)
        p1 = self.search_on_plane(p0, radius, tg_vec, vinit)
        plane_norm = p1 - p0
        vinit = np.cross(plane_norm, n)
        radius = self.estimate_spining_radius(p0, p1, vinit, vec_len(plane_norm))
        p2 = self.search_on_plane((p0 + p1) * 0.5, radius, plane_norm, vinit)
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
    
    def get_triangles(self, form='None'):
        if form is None:
            form = 'mayavi'
        x = deque()
        y = deque()
        z = deque()
        for point in self.points:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        return (x, y, z, self.triangles)

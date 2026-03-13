from cartopy import crs, feature
import matplotlib.pyplot as plt
from nansat import Nansat, Domain, NSR
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import distance_transform_edt


def get_dst_rows_cols(dst_dom):
    """ Create grids with row, column coordinates of the destination domain """
    rows2, cols2 = np.meshgrid(
        np.arange(0, dst_dom.shape()[0]),
        np.arange(0, dst_dom.shape()[1]),
        indexing='ij',
    )
    return rows2, cols2

def warp_with_rowcol(src_dom, src_img, c1, r1, c2, r2, dst_dom):
    """ Train interpolators of coordinates and apply to full resolution coordinates to computed a warped image """
    interp_r1 = LinearNDInterpolator(list(zip(r2, c2)), r1)
    interp_c1 = LinearNDInterpolator(list(zip(r2, c2)), c1)
    rows2, cols2 = get_dst_rows_cols(dst_dom)
    r1a = np.clip(interp_r1((rows2, cols2)), 0, src_dom.shape()[0])
    c1a = np.clip(interp_c1((rows2, cols2)), 0, src_dom.shape()[1])
    dst_img = map_coordinates(src_img, (r1a, c1a), order=0)
    return dst_img

def warp_distance(dst_dom, lon1, lat1, mask):
    """ Create a matrix with distance to the nearest valid drift and warp it onto the destination domain """
    c2_dist, r2_dist = dst_dom.transform_points(lon1.flatten(), lat1.flatten(), DstToSrc=1)
    mask_dist = distance_transform_edt(mask)
    interp_dist = LinearNDInterpolator(list(zip(r2_dist, c2_dist)), mask_dist.flatten())
    rows2, cols2 = get_dst_rows_cols(dst_dom)
    dst_dist = interp_dist((rows2, cols2))
    return dst_dist

def warp_with_lonlat(src_dom, src_img, lon1, lat1, lon2, lat2, dst_dom):
    """ Warp input image on destination domain if vectors of lon,lat source and destination points are knwown """
    c1, r1 = src_dom.transform_points(lon1.flatten(), lat1.flatten(), DstToSrc=1)
    c2, r2 = dst_dom.transform_points(lon2.flatten(), lat2.flatten(), DstToSrc=1)
    dst_img = warp_with_rowcol(src_dom, src_img, c1, r1, c2, r2, dst_dom)
    return dst_img

def warp(src_dom, src_img, dst_dom, step=None):
    """ Warp input image on destination domain (without drift compensation) """
    if step is None:
        step = int(src_dom.shape()[0]/100)
    src_lon, src_lat = src_dom.get_geolocation_grids(step)
    dst_img = warp_with_lonlat(src_dom, src_img, src_lon, src_lat, src_lon, src_lat, dst_dom)
    return dst_img

def warp_and_mask_with_lonlat(src_dom, src_img, lon1, lat1, lon2, lat2, mask, dst_dom, max_dist=2, fill_value=0):
    """ Warp input image on destination domain with drift compensation and masking if lon,lat,mask matrices are given """
    lon1v, lat1v, lon2v, lat2v = [i[~mask] for i in [lon1, lat1, lon2, lat2]]
    dst_img = warp_with_lonlat(src_dom, src_img, lon1v, lat1v, lon2v, lat2v, dst_dom)
    dst_dist = warp_distance(dst_dom, lon1, lat1, mask)
    dst_img[(dst_dist > max_dist) + np.isnan(dst_dist)] = fill_value
    return dst_img

def warp_with_uv(src_dom, src_img, uv_dom, u, v, mask, dst_dom):
    """ Warp input image on destination domain with drift compensation and masking if U,V,mask matrices are given """
    uv_srs = NSR(uv_dom.vrt.get_projection()[0])
    lon1uv, lat1uv = uv_dom.get_geolocation_grids()
    x1, y1, _ = uv_dom.vrt.transform_coordinates(NSR(), (lon1uv[~mask], lat1uv[~mask]), uv_srs)
    x2 = x1 + u[~mask]
    y2 = y1 + v[~mask]
    lon2uv, lat2uv, _ = uv_dom.vrt.transform_coordinates(uv_srs, (x2, y2), NSR())
    inp_img = np.array(src_img)
    inp_img[0] = 0
    inp_img[-1] = 0
    inp_img[:, 0] = 0
    inp_img[:, -1] = 0
    dst_img = warp_with_lonlat(src_dom, inp_img, lon1uv[~mask], lat1uv[~mask], lon2uv, lat2uv, dst_dom)
    return dst_img
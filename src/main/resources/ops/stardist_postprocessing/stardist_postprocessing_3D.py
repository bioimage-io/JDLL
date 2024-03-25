###
# #%L
# Use deep learning frameworks from Java in an agnostic and isolated way.
# %%
# Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
# %%
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# #L%
###
from __future__ import print_function, unicode_literals, absolute_import, division
from time import time
t = time()
import numpy as np
print("impot numpy star " + str(time() - t))
t = time()
from csbdeep.utils import axes_check_and_normalize, _raise
print("impot csbd.utils star " + str(time() - t))
t = time()
import math
print("impot math star NEW " + str(time() - t))
t = time()
import sys
import re
import os
ENV_DIR = os.path.dirname(sys.executable if os.name == 'nt' else os.path.dirname(sys.executable))
SITE_PACKAGES_DIR = os.path.join(ENV_DIR, "lib")
if os.name != 'nt':
	if os.path.isdir(SITE_PACKAGES_DIR + "/python" + str(sys.version_info[0]) + "." + str(sys.version_info[1])):
		SITE_PACKAGES_DIR = os.path.join(SITE_PACKAGES_DIR, "python" + str(sys.version_info[0]) + "." + str(sys.version_info[1]))
	else:
		PATTERN = r"python3\.\d{1,2}$"
		matching_files = [file for file in os.listdir(SITE_PACKAGES_DIR) if re.match(PATTERN, file)]
		SITE_PACKAGES_DIR = os.path.join(SITE_PACKAGES_DIR, matching_files[0])
STARDIST_DIR = os.path.join(SITE_PACKAGES_DIR, "site-packages", "stardist", "lib")
sys.path.append(os.path.join(STARDIST_DIR))
from stardist3d import c_non_max_suppression_inds, c_polyhedron_to_label
print("impot nms star " + str(time() - t))
t = time()

from scipy.spatial import ConvexHull
import copy
import warnings



def stardist_postprocessing(raw_output, prob_thresh, nms_thresh, n_classes=None,
                             grid=(1,1,1), b=2, channel=3, n_rays=96, n_dim=3, axes_net='ZYXC'):
    prob = raw_output[0, :, :, :, 0:1]
    dist = raw_output[0, :, :, :, 1:]
    prob, dist = _prep(prob, dist)
    inds = _ind_prob_thresh(prob, prob_thresh, b=b)
    proba = prob[inds].copy()
    dista = dist[inds].copy()
    _points = np.stack(np.where(inds), axis=1)
    pointsa = (_points * np.array(grid).reshape((1,len(grid))))

    if n_classes is not None:
        p = np.moveaxis(raw_output[2], channel, -1)
        prob_classa = p[inds].copy()

    proba = np.asarray(proba)
    dista = np.asarray(dista).reshape((-1, n_rays))
    pointsa = np.asarray(pointsa).reshape((-1, n_dim))

    grid_dict = dict(zip(axes_net.replace('C', ''), grid))
    resizer = StarDistPadAndCropResizer(grid=grid_dict, pad={"X": (0, 0), "Y": (0, 0), "Z": (0, 0), "C": (0, 0)},
                                        padded_shape={"X": prob.shape[2], "Z": prob.shape[0], "Y": prob.shape[1]})

    idx = resizer.filter_points(raw_output[0].ndim, pointsa, axes_net)
    prob = proba[idx]
    dist = dista[idx]
    points = pointsa[idx]

    # last "yield" is the actual output that would have been "return"ed if this was a regular function
    if n_classes is not None:
        prob_classa = np.asarray(prob_classa).reshape((-1, n_classes + 1))
        prob_class = prob_classa[idx]
    else:
        prob_classa = None
        prob_class = None

    nms_kwargs = {'verbose': False}
    res_instances = instances_from_prediction(raw_output[0, :, :, :, 0].shape, prob, dist,
                                              prob_thresh, nms_thresh, grid,
                                              points=points,
                                              prob_class=prob_class,
                                              scale=None,
                                              return_labels=True,
                                              overlap_label=None,
                                              **nms_kwargs)
    return res_instances[0].astype("float32")


def _prep(prob, dist, channel=3):
    prob = np.take(prob, 0, axis=channel)
    dist = np.moveaxis(dist, channel, -1)
    dist = np.maximum(1e-3, dist)
    return prob, dist


def _ind_prob_thresh(prob, prob_thresh, b=2):
    if b is not None and np.isscalar(b):
        b = ((b, b),) * prob.ndim

    ind_thresh = prob > prob_thresh
    if b is not None:
        _ind_thresh = np.zeros_like(ind_thresh)
        ss = tuple(slice(_bs[0] if _bs[0] > 0 else None,
                         -_bs[1] if _bs[1] > 0 else None) for _bs in b)
        _ind_thresh[ss] = True
        ind_thresh &= _ind_thresh
    return ind_thresh


def instances_from_prediction(img_shape, prob, dist, prob_thresh, nms_thresh, grid, points=None, prob_class=None,
                              overlap_label=None, return_labels=True, scale=None, **nms_kwargs):
    """
    if points is None     -> dense prediction
    if points is not None -> sparse prediction

    if prob_class is None     -> single class prediction
    if prob_class is not None -> multi  class prediction
    """
    rays = rays_from_json({'name': 'Rays_GoldenSpiral', 'kwargs': {'n': 96, 'anisotropy': (2, 1, 1)}})
    
    if overlap_label is not None: raise NotImplementedError("overlap_label not supported for 2D yet!")

    # sparse prediction
    if points is not None:
        points, probi, disti, indsi = non_maximum_suppression_3D_sparse(dist, prob, points, rays, nms_thresh=nms_thresh,
                                                                     **nms_kwargs)
        if prob_class is not None:
            prob_class = prob_class[indsi]

    # dense prediction
    else:
        points, probi, disti = non_maximum_suppression(dist, prob, grid=grid,
                                                       prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs)
        if prob_class is not None:
            inds = tuple(p // g for p, g in zip(points.T, grid))
            prob_class = prob_class[inds]

    if scale is not None:
        # need to undo the scaling given by the scale dict, e.g. scale = dict(X=0.5,Y=0.5):
        #   1. re-scale points (origins of polygons)
        #   2. re-scale coordinates (computed from distances) of (zero-origin) polygons
        if not (isinstance(scale, dict) and 'X' in scale and 'Y' in scale):
            raise ValueError("scale must be a dictionary with entries for 'X' and 'Y'")
        rescale = (1 / scale['Y'], 1 / scale['X'])
        points = points * np.array(rescale).reshape(1, 2)
    else:
        rescale = (1, 1)

    if return_labels:
        labels = polyhedron_to_label(disti, points, rays=rays, prob=probi, shape=img_shape, overlap_label=overlap_label)
        if overlap_label is not None and overlap_label<0 and (overlap_label in labels):
                overlap_mask = (labels == overlap_label)
                overlap_label2 = max(set(np.unique(labels))-{overlap_label})+1
                labels[overlap_mask] = overlap_label2
                labels, fwd, bwd = relabel_sequential(labels)
                labels[labels == fwd[overlap_label2]] = overlap_label
        else:
		# TODO relabel_sequential necessary?
		# print(np.unique(labels))
            labels, _,_ = relabel_sequential(labels)
    else:
        labels = None

    res_dict = dict(dist=disti, points=points, prob=probi, rays=rays, rays_vertices=rays.vertices, rays_faces=rays.faces)


    # multi class prediction
    if prob_class is not None:
        prob_class = np.asarray(prob_class)
        class_id = np.argmax(prob_class, axis=-1)
        res_dict.update(dict(class_prob=prob_class, class_id=class_id))

    return labels, res_dict
    
def relabel_sequential(label_field, offset=1):
    """Relabel arbitrary labels to {`offset`, ... `offset` + number_of_labels}.

    This function also returns the forward map (mapping the original labels to
    the reduced labels) and the inverse map (mapping the reduced labels back
    to the original ones).

    Parameters
    ----------
    label_field : numpy array of int, arbitrary shape
        An array of labels, which must be non-negative integers.
    offset : int, optional
        The return labels will start at `offset`, which should be
        strictly positive.

    Returns
    -------
    relabeled : numpy array of int, same shape as `label_field`
        The input label field with labels mapped to
        {offset, ..., number_of_labels + offset - 1}.
        The data type will be the same as `label_field`, except when
        offset + number_of_labels causes overflow of the current data type.
    forward_map : numpy array of int, shape ``(label_field.max() + 1,)``
        The map from the original label space to the returned label
        space. Can be used to re-apply the same mapping. See examples
        for usage. The data type will be the same as `relabeled`.
    inverse_map : 1D numpy array of int, of length offset + number of labels
        The map from the new label space to the original space. This
        can be used to reconstruct the original label field from the
        relabeled one. The data type will be the same as `relabeled`.

    Notes
    -----
    The label 0 is assumed to denote the background and is never remapped.

    The forward map can be extremely big for some inputs, since its
    length is given by the maximum of the label field. However, in most
    situations, ``label_field.max()`` is much smaller than
    ``label_field.size``, and in these cases the forward map is
    guaranteed to be smaller than either the input or output images.

    Examples
    --------
    >>> from skimage.segmentation import relabel_sequential
    >>> label_field = np.array([1, 1, 5, 5, 8, 99, 42])
    >>> relab, fw, inv = relabel_sequential(label_field)
    >>> relab
    array([1, 1, 2, 2, 3, 5, 4])
    >>> fw
    array([0, 1, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5])
    >>> inv
    array([ 0,  1,  5,  8, 42, 99])
    >>> (fw[label_field] == relab).all()
    True
    >>> (inv[relab] == label_field).all()
    True
    >>> relab, fw, inv = relabel_sequential(label_field, offset=5)
    >>> relab
    array([5, 5, 6, 6, 7, 9, 8])
    """
    offset = int(offset)
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    if np.min(label_field) < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    max_label = int(label_field.max()) # Ensure max_label is an integer
    if not np.issubdtype(label_field.dtype, np.integer):
        new_type = np.min_scalar_type(max_label)
        label_field = label_field.astype(new_type)
    labels = np.unique(label_field)
    labels0 = labels[labels != 0]
    new_max_label = offset - 1 + len(labels0)
    new_labels0 = np.arange(offset, new_max_label + 1)
    output_type = label_field.dtype
    required_type = np.min_scalar_type(new_max_label)
    if np.dtype(required_type).itemsize > np.dtype(label_field.dtype).itemsize:
        output_type = required_type
    forward_map = np.zeros(max_label + 1, dtype=output_type)
    forward_map[labels0] = new_labels0
    inverse_map = np.zeros(new_max_label + 1, dtype=output_type)
    inverse_map[offset:] = labels0
    relabeled = forward_map[label_field]
    return relabeled, forward_map, inverse_map


def non_maximum_suppression(dist, prob, grid=(1,1), b=2, nms_thresh=0.5, prob_thresh=0.5,
                            use_bbox=True, use_kdtree=True, verbose=False):
    """Non-Maximum-Supression of 2D polygons

    Retains only polygons whose overlap is smaller than nms_thresh

    dist.shape = (Ny,Nx, n_rays)
    prob.shape = (Ny,Nx)

    returns the retained points, probabilities, and distances:

    points, prob, dist = non_maximum_suppression(dist, prob, ....

    """

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    assert prob.ndim == 2 and dist.ndim == 3  and prob.shape == dist.shape[:2]
    dist = np.asarray(dist)
    prob = np.asarray(prob)
    n_rays = dist.shape[-1]

    grid = _normalize_grid(grid,2)

    # mask = prob > prob_thresh
    # if b is not None and b > 0:
    #     _mask = np.zeros_like(mask)
    #     _mask[b:-b,b:-b] = True
    #     mask &= _mask

    mask = _ind_prob_thresh(prob, prob_thresh, b)
    points = np.stack(np.where(mask), axis=1)

    dist   = dist[mask]
    scores = prob[mask]

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    dist   = dist[ind]
    scores = scores[ind]
    points = points[ind]

    points = (points * np.array(grid).reshape((1,2)))

    if verbose:
        t = time()

    inds = non_maximum_suppression_inds(dist, points.astype(np.int32, copy=False), scores=scores,
                                        use_bbox=use_bbox, use_kdtree=use_kdtree,
                                        thresh=nms_thresh, verbose=verbose)

    if verbose:
        print("keeping %s/%s polygons" % (np.count_nonzero(inds), len(inds)))
        print("NMS took %.4f s" % (time() - t))

    return points[inds], scores[inds], dist[inds]


def non_maximum_suppression_3D_sparse(dist, prob, points, rays, b=2, nms_thresh=0.5,
                                   use_bbox=True, use_kdtree = True, verbose=False):
    """Non-Maximum-Supression of 2D polygons from a list of dists, probs (scores), and points

    Retains only polyhedra whose overlap is smaller than nms_thresh

    dist.shape = (n_polys, n_rays)
    prob.shape = (n_polys,)
    points.shape = (n_polys,2)

    returns the retained instances

    (pointsi, probi, disti, indsi)

    with
    pointsi = points[indsi] ...

    """

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    dist = np.asarray(dist)
    prob = np.asarray(prob)
    points = np.asarray(points)
    n_rays = dist.shape[-1]

    assert dist.ndim == 2 and prob.ndim == 1 and points.ndim == 2 and \
        points.shape[-1]==3 and len(prob) == len(dist) == len(points)

    verbose and print("predicting instances with nms_thresh = {nms_thresh}".format(nms_thresh=nms_thresh), flush=True)

    inds_original = np.arange(len(prob))
    _sorted = np.argsort(prob)[::-1]
    probi = prob[_sorted]
    disti = dist[_sorted]
    pointsi = points[_sorted]
    inds_original = inds_original[_sorted]

    if verbose:
        print("non-maximum suppression...")
        t = time()

    inds = non_maximum_suppression_3d_inds(disti, pointsi, rays=rays, scores=probi, thresh=nms_thresh, use_kdtree = use_kdtree, verbose=verbose)

    if verbose:
        print("keeping %s/%s polyhedra" % (np.count_nonzero(inds), len(inds)))
        print("NMS took %.4f s" % (time() - t))

    return pointsi[inds], probi[inds], disti[inds], inds_original[inds]


def non_maximum_suppression_3d_inds(dist, points, rays, scores, thresh=0.5, use_bbox=True, use_kdtree = True, verbose=1):
    """
    Applies non maximum supression to ray-convex polygons given by dists and points
    sorted by scores and IoU threshold

    P1 will suppress P2, if IoU(P1,P2) > thresh

    with IoU(P1,P2) = Ainter(P1,P2) / min(A(P1),A(P2))

    i.e. the smaller thresh, the more polygons will be supressed

    dist.shape = (n_poly, n_rays)
    point.shape = (n_poly, 2)
    score.shape = (n_poly,)

    returns indices of selected polygons
    """


    assert dist.ndim == 2
    assert points.ndim == 2

    n_poly = dist.shape[0]

    if scores is None:
        scores = np.ones(n_poly)

    assert len(scores) == n_poly
    assert points.shape[0] == n_poly

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    survivors = np.ones(n_poly, bool)
    dist = dist[ind]
    points = points[ind]
    scores = scores[ind]

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    survivors[ind] = c_non_max_suppression_inds(_prep(dist,  np.float32),
                                      	_prep(points, np.float32),
		                        _prep(rays.vertices, np.float32),
		                        _prep(rays.faces, np.int32),
		                        _prep(scores, np.float32),
		                        int(use_bbox),
		                        int(use_kdtree),
		                        int(verbose),
		                        np.float32(thresh))

    return survivors


def _ind_prob_thresh(prob, prob_thresh, b=2):
    if b is not None and np.isscalar(b):
        b = ((b,b),)*prob.ndim

    ind_thresh = prob > prob_thresh
    if b is not None:
        _ind_thresh = np.zeros_like(ind_thresh)
        ss = tuple(slice(_bs[0] if _bs[0]>0 else None,
                         -_bs[1] if _bs[1]>0 else None)  for _bs in b)
        _ind_thresh[ss] = True
        ind_thresh &= _ind_thresh
    return ind_thresh


class StarDistPadAndCropResizer():

    # TODO: check correctness
    def __init__(self, grid, mode='reflect', pad=None, padded_shape=None, **kwargs):
        assert isinstance(grid, dict)
        self.mode = mode
        self.grid = grid
        self.pad = pad
        self.padded_shape = padded_shape
        self.kwargs = kwargs

    def before(self, x, axes, axes_div_by):
        assert all(a%g==0 for g,a in zip((self.grid.get(a,1) for a in axes), axes_div_by))
        axes = axes_check_and_normalize(axes,x.ndim)
        def _split(v):
            return 0, v # only pad at the end
        self.pad = {
            a : _split((div_n-s%div_n)%div_n)
            for a, div_n, s in zip(axes, axes_div_by, x.shape)
        }
        x_pad = np.pad(x, tuple(self.pad[a] for a in axes), mode=self.mode, **self.kwargs)
        self.padded_shape = dict(zip(axes,x_pad.shape))
        if 'C' in self.padded_shape: del self.padded_shape['C']
        return x_pad


    def after(self, x, axes):
        # axes can include 'C', which may not have been present in before()
        axes = axes_check_and_normalize(axes,x.ndim)
        assert all(s_pad == s * g for s,s_pad,g in zip(x.shape,
                                                       (self.padded_shape.get(a,_s) for a,_s in zip(axes,x.shape)),
                                                       (self.grid.get(a,1) for a in axes)))
        # print(self.padded_shape)
        # print(self.pad)
        # print(self.grid)
        crop = tuple (
            slice(0, -(math.floor(p[1]/g)) if p[1]>=g else None)
            for p,g in zip((self.pad.get(a,(0,0)) for a in axes),(self.grid.get(a,1) for a in axes))
        )
        # print(crop)
        return x[crop]

    def filter_points(self, ndim, points, axes):
        """ returns indices of points inside crop region """
        assert points.ndim==2
        axes = axes_check_and_normalize(axes,ndim)

        bounds = np.array(tuple(self.padded_shape[a]-self.pad[a][1] for a in axes if a.lower() in ('z','y','x')))
        idx = np.where(np.all(points< bounds, 1))
        return idx


def dist_to_coord(dist, points, scale_dist=(1,1)):
    """convert from polar to cartesian coordinates for a list of distances and center points
    dist.shape   = (n_polys, n_rays)
    points.shape = (n_polys, 2)
    len(scale_dist) = 2
    return coord.shape = (n_polys,2,n_rays)
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    assert dist.ndim==2 and points.ndim==2 and len(dist)==len(points) \
        and points.shape[1]==2 and len(scale_dist)==2
    n_rays = dist.shape[1]
    phis = ray_angles(n_rays)
    coord = (dist[:,np.newaxis]*np.array([np.sin(phis),np.cos(phis)])).astype(np.float32)
    coord *= np.asarray(scale_dist).reshape(1,2,1)    
    coord += points[...,np.newaxis] 
    return coord


def ray_angles(n_rays=32):
    return np.linspace(0,2*np.pi,n_rays,endpoint=False)


def polyhedron_to_label(dist, points, rays, shape, prob=None, thr=-np.inf, labels=None, mode="full", verbose=True, overlap_label=None):
    """
    creates labeled image from stardist representations

    :param dist: array of shape (n_points,n_rays)
        the list of distances for each point and ray
    :param points: array of shape (n_points, 3)
        the list of center points
    :param rays: Rays object
        Ray object (e.g. `stardist.Rays_GoldenSpiral`) defining
        vertices and faces
    :param shape: (nz,ny,nx)
        output shape of the image
    :param prob: array of length/shape (n_points,) or None
        probability per polyhedron
    :param thr: scalar
        probability threshold (only polyhedra with prob>thr are labeled)
    :param labels: array of length/shape (n_points,) or None
        labels to use
    :param mode: str
        labeling mode, can be "full", "kernel", "hull", "bbox" or  "debug"
    :param verbose: bool
        enable to print some debug messages
    :param overlap_label: scalar or None
        if given, will label each pixel that belongs ot more than one polyhedron with that label
    :return: array of given shape
        labeled image
    """
    if len(points) == 0:
        if verbose:
            print("warning: empty list of points (returning background-only image)")
        return np.zeros(shape, np.uint16)

    dist = np.asanyarray(dist)
    points = np.asanyarray(points)

    if dist.ndim == 1:
        dist = dist.reshape(1, -1)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if labels is None:
        labels = np.arange(1, len(points) + 1)

    if np.amin(dist) <= 0:
        raise ValueError("distance array should be positive!")

    prob = np.ones(len(points)) if prob is None else np.asanyarray(prob)

    if dist.ndim != 2:
        raise ValueError("dist should be 2 dimensional but has shape %s" % str(dist.shape))

    if dist.shape[1] != len(rays):
        raise ValueError("inconsistent number of rays!")

    if len(prob) != len(points):
        raise ValueError("len(prob) != len(points)")

    if len(labels) != len(points):
        raise ValueError("len(labels) != len(points)")

    modes = {"full": 0, "kernel": 1, "hull": 2, "bbox": 3, "debug": 4}

    if not mode in modes:
        raise KeyError("Unknown render mode '%s' , allowed:  %s" % (mode, tuple(modes.keys())))

    lbl = np.zeros(shape, np.uint16)

    # filter points
    ind = np.where(prob >= thr)[0]
    if len(ind) == 0:
        if verbose:
            print("warning: no points found with probability>= {thr:.4f} (returning background-only image)".format(thr=thr))
        return lbl

    prob = prob[ind]
    points = points[ind]
    dist = dist[ind]
    labels = labels[ind]

    # sort points with decreasing probability
    ind = np.argsort(prob)[::-1]
    points = points[ind]
    dist = dist[ind]
    labels = labels[ind]

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    return c_polyhedron_to_label(_prep(dist, np.float32),
                                 _prep(points, np.float32),
                                 _prep(rays.vertices, np.float32),
                                 _prep(rays.faces, np.int32),
                                 _prep(labels, np.int32),
                                 np.int32(modes[mode]),
                                 np.int32(verbose),
                                 np.int32(overlap_label is not None),
                                 np.int32(0 if overlap_label is None else overlap_label),
                                 shape
                                 )

def _check_label_array(y, name=None, check_sequential=False):
    err = ValueError("{label} must be an array of {integers}.".format(
        label = 'labels' if name is None else name,
        integers = ('sequential ' if check_sequential else '') + 'non-negative integers',
    ))
    is_array_of_integers(y) or _raise(err)
    if len(y) == 0:
        return True
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True


def is_array_of_integers(y):
    return isinstance(y,np.ndarray) and np.issubdtype(y.dtype, np.integer)


def label_are_sequential(y):
    """ returns true if y has only sequential labels from 1... """
    labels = np.unique(y)
    return (set(labels)-{0}) == set(range(1,1+labels.max()))


def _normalize_grid(grid,n):
    try:
        grid = tuple(grid)
        (len(grid) == n and
         all(map(np.isscalar,grid)) and
         all(map(_is_power_of_2,grid))) or _raise(TypeError())
        return tuple(int(g) for g in grid)
    except (TypeError, AssertionError):
        raise ValueError("grid = {grid} must be a list/tuple of length {n} with values that are power of 2".format(grid=grid, n=n))
        
        
"""
Ray factory

classes that provide vertex and triangle information for rays on spheres

Example:

    rays = Rays_Tetra(n_level = 4)

    print(rays.vertices)
    print(rays.faces)

"""

class Rays_Base(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._vertices, self._faces = self.setup_vertices_faces()
        self._vertices = np.asarray(self._vertices, np.float32)
        self._faces = np.asarray(self._faces, int)
        self._faces = np.asanyarray(self._faces)

    def setup_vertices_faces(self):
        """has to return

         verts , faces

         verts = ( (z_1,y_1,x_1), ... )
         faces ( (0,1,2), (2,3,4), ... )

         """
        raise NotImplementedError()

    @property
    def vertices(self):
        """read-only property"""
        return self._vertices.copy()

    @property
    def faces(self):
        """read-only property"""
        return self._faces.copy()

    def __getitem__(self, i):
        return self.vertices[i]

    def __len__(self):
        return len(self._vertices)

    def __repr__(self):
        def _conv(x):
            if isinstance(x,(tuple, list, np.ndarray)):
                return "_".join(_conv(_x) for _x in x)
            if isinstance(x,float):
                return "%.2f"%x
            return str(x)
        return "%s_%s" % (self.__class__.__name__, "_".join("%s_%s" % (k, _conv(v)) for k, v in sorted(self.kwargs.items())))
    
    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "kwargs": self.kwargs
        }

    def dist_loss_weights(self, anisotropy = (1,1,1)):
        """returns the anisotropy corrected weights for each ray"""
        anisotropy = np.array(anisotropy)
        assert anisotropy.shape == (3,)
        return np.linalg.norm(self.vertices*anisotropy, axis = -1)

    def volume(self, dist=None):
        """volume of the starconvex polyhedron spanned by dist (if None, uses dist=1)
        dist can be a nD array, but the last dimension has to be of length n_rays
        """
        if dist is None: dist = np.ones_like(self.vertices)

        dist = np.asarray(dist)
        
        if not dist.shape[-1]==len(self.vertices):
            raise ValueError("last dimension of dist should have length len(rays.vertices)")
        # all the shuffling below is to allow dist to be an arbitrary sized array (with last dim n_rays)
        # self.vertices -> (n_rays,3)
        # dist -> (m,n,..., n_rays)
        
        # dist  -> (m,n,..., n_rays, 3)
        dist = np.repeat(np.expand_dims(dist,-1), 3, axis = -1)
        # verts  -> (m,n,..., n_rays, 3)
        verts = np.broadcast_to(self.vertices, dist.shape)

        # dist, verts  -> (n_rays, m,n, ..., 3)        
        dist = np.moveaxis(dist,-2,0)
        verts = np.moveaxis(verts,-2,0)

        # vs -> (n_faces, 3, m, n, ..., 3)
        vs = (dist*verts)[self.faces]
        # vs -> (n_faces, m, n, ..., 3, 3)
        vs = np.moveaxis(vs, 1,-2)
        # vs -> (n_faces * m * n, 3, 3)        
        vs = vs.reshape((len(self.faces)*int(np.prod(dist.shape[1:-1])),3,3))
        d = np.linalg.det(list(vs)).reshape((len(self.faces),)+dist.shape[1:-1])
        
        return -1./6*np.sum(d, axis = 0)
    
    def surface(self, dist=None):
        """surface area of the starconvex polyhedron spanned by dist (if None, uses dist=1)"""
        dist = np.asarray(dist)
        
        if not dist.shape[-1]==len(self.vertices):
            raise ValueError("last dimension of dist should have length len(rays.vertices)")

        # self.vertices -> (n_rays,3)
        # dist -> (m,n,..., n_rays)
        
        # all the shuffling below is to allow dist to be an arbitrary sized array (with last dim n_rays)
        
        # dist  -> (m,n,..., n_rays, 3)
        dist = np.repeat(np.expand_dims(dist,-1), 3, axis = -1)
        # verts  -> (m,n,..., n_rays, 3)
        verts = np.broadcast_to(self.vertices, dist.shape)

        # dist, verts  -> (n_rays, m,n, ..., 3)        
        dist = np.moveaxis(dist,-2,0)
        verts = np.moveaxis(verts,-2,0)

        # vs -> (n_faces, 3, m, n, ..., 3)
        vs = (dist*verts)[self.faces]
        # vs -> (n_faces, m, n, ..., 3, 3)
        vs = np.moveaxis(vs, 1,-2)
        # vs -> (n_faces * m * n, 3, 3)        
        vs = vs.reshape((len(self.faces)*int(np.prod(dist.shape[1:-1])),3,3))
       
        pa = vs[...,1,:]-vs[...,0,:]
        pb = vs[...,2,:]-vs[...,0,:]

        d = .5*np.linalg.norm(np.cross(list(pa), list(pb)), axis = -1)
        d = d.reshape((len(self.faces),)+dist.shape[1:-1])
        return np.sum(d, axis = 0)

    
    def copy(self, scale=(1,1,1)):
        """ returns a copy whose vertices are scaled by given factor"""
        scale = np.asarray(scale)
        assert scale.shape == (3,)
        res = copy.deepcopy(self)
        res._vertices *= scale[np.newaxis]
        return res 



    
def rays_from_json(d):
    return eval(d["name"])(**d["kwargs"])


################################################################

class Rays_Explicit(Rays_Base):
    def __init__(self, vertices0, faces0):
        self.vertices0, self.faces0 = vertices0, faces0
        super().__init__(vertices0=list(vertices0), faces0=list(faces0))
        
    def setup_vertices_faces(self):
        return self.vertices0, self.faces0
    

class Rays_Cartesian(Rays_Base):
    def __init__(self, n_rays_x=11, n_rays_z=5):
        super().__init__(n_rays_x=n_rays_x, n_rays_z=n_rays_z)

    def setup_vertices_faces(self):
        """has to return list of ( (z_1,y_1,x_1), ... )  _"""
        n_rays_x, n_rays_z = self.kwargs["n_rays_x"], self.kwargs["n_rays_z"]
        dphi = np.float32(2. * np.pi / n_rays_x)
        dtheta = np.float32(np.pi / n_rays_z)

        verts = []
        for mz in range(n_rays_z):
            for mx in range(n_rays_x):
                phi = mx * dphi
                theta = mz * dtheta
                if mz == 0:
                    theta = 1e-12
                if mz == n_rays_z - 1:
                    theta = np.pi - 1e-12
                dx = np.cos(phi) * np.sin(theta)
                dy = np.sin(phi) * np.sin(theta)
                dz = np.cos(theta)
                if mz == 0 or mz == n_rays_z - 1:
                    dx += 1e-12
                    dy += 1e-12
                verts.append([dz, dy, dx])

        verts = np.array(verts)

        def _ind(mz, mx):
            return mz * n_rays_x + mx

        faces = []

        for mz in range(n_rays_z - 1):
            for mx in range(n_rays_x):
                faces.append([_ind(mz, mx), _ind(mz + 1, (mx + 1) % n_rays_x), _ind(mz, (mx + 1) % n_rays_x)])
                faces.append([_ind(mz, mx), _ind(mz + 1, mx), _ind(mz + 1, (mx + 1) % n_rays_x)])

        faces = np.array(faces)

        return verts, faces


class Rays_SubDivide(Rays_Base):
    """
    Subdivision polyehdra

    n_level = 1 -> base polyhedra
    n_level = 2 -> 1x subdivision
    n_level = 3 -> 2x subdivision
                ...
    """

    def __init__(self, n_level=4):
        super().__init__(n_level=n_level)

    def base_polyhedron(self):
        raise NotImplementedError()

    def setup_vertices_faces(self):
        n_level = self.kwargs["n_level"]
        verts0, faces0 = self.base_polyhedron()
        return self._recursive_split(verts0, faces0, n_level)

    def _recursive_split(self, verts, faces, n_level):
        if n_level <= 1:
            return verts, faces
        else:
            verts, faces = Rays_SubDivide.split(verts, faces)
            return self._recursive_split(verts, faces, n_level - 1)

    @classmethod
    def split(self, verts0, faces0):
        """split a level"""

        split_edges = dict()
        verts = list(verts0[:])
        faces = []

        def _add(a, b):
            """ returns index of middle point and adds vertex if not already added"""
            edge = tuple(sorted((a, b)))
            if not edge in split_edges:
                v = .5 * (verts[a] + verts[b])
                v *= 1. / np.linalg.norm(v)
                verts.append(v)
                split_edges[edge] = len(verts) - 1
            return split_edges[edge]

        for v1, v2, v3 in faces0:
            ind1 = _add(v1, v2)
            ind2 = _add(v2, v3)
            ind3 = _add(v3, v1)
            faces.append([v1, ind1, ind3])
            faces.append([v2, ind2, ind1])
            faces.append([v3, ind3, ind2])
            faces.append([ind1, ind2, ind3])

        return verts, faces


class Rays_Tetra(Rays_SubDivide):
    """
    Subdivision of a tetrahedron

    n_level = 1 -> normal tetrahedron (4 vertices)
    n_level = 2 -> 1x subdivision (10 vertices)
    n_level = 3 -> 2x subdivision (34 vertices)
                ...
    """

    def base_polyhedron(self):
        verts = np.array([
            [np.sqrt(8. / 9), 0., -1. / 3],
            [-np.sqrt(2. / 9), np.sqrt(2. / 3), -1. / 3],
            [-np.sqrt(2. / 9), -np.sqrt(2. / 3), -1. / 3],
            [0., 0., 1.]
        ])
        faces = [[0, 1, 2],
                 [0, 3, 1],
                 [0, 2, 3],
                 [1, 3, 2]]

        return verts, faces


class Rays_Octo(Rays_SubDivide):
    """
    Subdivision of a tetrahedron

    n_level = 1 -> normal Octahedron (6 vertices)
    n_level = 2 -> 1x subdivision (18 vertices)
    n_level = 3 -> 2x subdivision (66 vertices)

    """

    def base_polyhedron(self):
        verts = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, -1],
            [0, -1, 0],
            [1, 0, 0],
            [-1, 0, 0]])

        faces = [[0, 1, 4],
                 [0, 5, 1],
                 [1, 2, 4],
                 [1, 5, 2],
                 [2, 3, 4],
                 [2, 5, 3],
                 [3, 0, 4],
                 [3, 5, 0],
                 ]

        return verts, faces


def reorder_faces(verts, faces):
    """reorder faces such that their orientation points outward"""
    def _single(face):
        return face[::-1] if np.linalg.det(verts[face])>0 else face
    return tuple(map(_single, faces))


class Rays_GoldenSpiral(Rays_Base):
    def __init__(self, n=70, anisotropy = None):
        if n<4:
            raise ValueError("At least 4 points have to be given!")
        super().__init__(n=n, anisotropy = anisotropy if anisotropy is None else tuple(anisotropy))

    def setup_vertices_faces(self):
        n = self.kwargs["n"]
        anisotropy = self.kwargs["anisotropy"]
        if anisotropy is None:
            anisotropy = np.ones(3)
        else:
            anisotropy = np.array(anisotropy)

        # the smaller golden angle = 2pi * 0.3819...
        g = (3. - np.sqrt(5.)) * np.pi
        phi = g * np.arange(n)
        # z = np.linspace(-1, 1, n + 2)[1:-1]
        # rho = np.sqrt(1. - z ** 2)
        # verts = np.stack([rho*np.cos(phi), rho*np.sin(phi),z]).T
        #
        z = np.linspace(-1, 1, n)
        rho = np.sqrt(1. - z ** 2)
        verts = np.stack([z, rho * np.sin(phi), rho * np.cos(phi)]).T

        # warnings.warn("ray definition has changed! Old results are invalid!")

        # correct for anisotropy
        verts = verts/anisotropy
        #verts /= np.linalg.norm(verts, axis=-1, keepdims=True)

        hull = ConvexHull(verts)
        faces = reorder_faces(verts,hull.simplices)

        verts /= np.linalg.norm(verts, axis=-1, keepdims=True)

        return verts, faces


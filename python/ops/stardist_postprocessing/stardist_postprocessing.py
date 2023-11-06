import numpy as np
from stardist.nms import non_maximum_suppression_sparse, non_maximum_suppression
from stardist.geometry import dist_to_coord, polygons_to_label
from xarray import DataArray
from csbdeep.data import Resizer
from csbdeep.utils import axes_check_and_normalize
import math


def stardist_postprocessing(raw_output, prob_thresh, nms_thresh, n_classes=None,
                             grid=(1,1), b=2, channel=2, n_rays=32, n_dim=2, axes_net='YXC'):
    raw_output = raw_output.values
    prob = raw_output[0, :, :, 0:1]
    dist = raw_output[0, :, :, 1:]
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
    resizer = StarDistPadAndCropResizer(grid=grid_dict, pad={"X": (0, 0), "Y": (0, 0), "C": (0, 0)},
                                        padded_shape={"X": prob.shape[1], "Y": prob.shape[0]})

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
    res_instances = instances_from_prediction(raw_output[0, :, :, 0].shape, prob, dist,
                                              prob_thresh, nms_thresh, grid,
                                              points=points,
                                              prob_class=prob_class,
                                              scale=None,
                                              return_labels=True,
                                              overlap_label=None,
                                              **nms_kwargs)
    return DataArray(res_instances[0].astype("float32"), dims=['y', 'x'], name='output')


def _prep(prob, dist, channel=2):
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
    if overlap_label is not None: raise NotImplementedError("overlap_label not supported for 2D yet!")

    # sparse prediction
    if points is not None:
        points, probi, disti, indsi = non_maximum_suppression_sparse(dist, prob, points, nms_thresh=nms_thresh,
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
        labels = polygons_to_label(disti, points, prob=probi, shape=img_shape, scale_dist=rescale)
    else:
        labels = None

    coord = dist_to_coord(disti, points, scale_dist=rescale)
    res_dict = dict(coord=coord, points=points, prob=probi)

    # multi class prediction
    if prob_class is not None:
        prob_class = np.asarray(prob_class)
        class_id = np.argmax(prob_class, axis=-1)
        res_dict.update(dict(class_prob=prob_class, class_id=class_id))

    return labels, res_dict


class StarDistPadAndCropResizer(Resizer):

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
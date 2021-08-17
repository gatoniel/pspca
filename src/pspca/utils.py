import warnings
import numpy as np


def norm(x, keepdims=True):
    return np.linalg.norm(x, ord=2, axis=-1, keepdims=keepdims)


def normalize(x):
    return x / norm(x, True)


def frechet_similarity_distance(x, points):
    x = normalize(x)
    return np.sum(np.inner(x, points) ** 2, axis=-1)


def prospace_pca(points):
    points = normalize(points)
    A = np.matmul(points.T, points)
    w, v = np.linalg.eigh(A)

    inds = np.argsort(w)[::-1]
    w = w[inds]
    v = v[:, inds]

    if np.all(v[:, 0] < 0):
        v[:, 0] *= -1
    if not np.all(v[:, 0] > 0):
        warnings.warn("The first component is not strictly positive!")

    test = test_surroundings(v[:, 0], points)
    if not test[0] and test[1]:
        warnings.warn("The first component is not a maximum!")
    test = test_surroundings(v[:, -1], points)
    if test[0] and not test[1]:
        warnings.warn("The last component is not a minimum!")

    for i in range(1, v.shape[0] - 1):
        test = test_surroundings(v[:, -1], points)
        if not test[0] and not test[1]:
            warnings.warn(f"Component {i} is not a saddle point!")

    for i in range(v.shape[0]):
        v_ = v[:, i]
        mi = v_.min()
        ma = v_.max()
        if np.abs(mi) > ma:
            v[:, i] *= -1

    return w, v, A


def transform(points, v):
    inv_v = np.linalg.inv(v)
    return np.matmul(inv_v, points[:, :, np.newaxis]).squeeze()


def reduce_dimensions(points, v, num):
    return normalize(transform(points, v)[:, :num])


def mean(points):
    _, v, _ = prospace_pca(points)
    return v[:, 0]


def other_points(v, num=10000):
    return normalize(v + np.random.rand(num, v.shape[0]) / 1000)


def test_surroundings(v, points):
    o_points = other_points(v)
    diff_loss = frechet_similarity_distance(
        v, points
    ) - frechet_similarity_distance(o_points, points)
    return (np.all(diff_loss >= 0), np.all(diff_loss <= 0))

"""Tests for utils submodule."""
import numpy as np

import pspca.utils as utils


def test_normalize():
    """Correct normalization."""
    x = utils.normalize(np.random.randn(100, 4))
    np.testing.assert_allclose(utils.norm(x), 1.0)


def test_frechet_similarity_distance():
    """Test Frechet similarity distance with equally distributed vectors."""
    points = np.eye(3)
    np.testing.assert_allclose(utils.frechet_similarity_distance(points, points), 1.0)

    x = np.ones(3) / np.sqrt(3)
    np.testing.assert_allclose(utils.frechet_similarity_distance(x, points), 1.0)


def test_transform_prospace_pca_reduce_dimensions():
    """Test complete pipeline."""
    dim = 5
    points = utils.normalize(np.exp(np.random.randn(100, dim)))
    w, v, matrix_a = utils.pspca(points)

    assert len(w) == dim
    assert v.shape[0] == v.shape[1] == dim
    assert matrix_a.shape[0] == matrix_a.shape[1] == dim

    t_points = utils.transform(points, v)
    assert points.shape == t_points.shape

    np.testing.assert_allclose(
        np.matmul(v, t_points[:, :, np.newaxis]).squeeze(), points
    )

    num = 3
    red_points = utils.reduce_dimensions(points, v, num)
    assert red_points.shape[1] == num


def test_mean():
    """Test correct dimension output of mean."""
    dim = 5
    points = utils.normalize(np.exp(np.random.randn(100, dim)))
    assert len(utils.mean(points)) == dim


def test_warnings():
    """Do warnings pop up?"""
    points = np.eye(3)
    w, v, matrix_a = utils.pspca(points)
    np.testing.assert_allclose(v[::-1, :], points)

"""Tests for wrapper submodule."""
import numpy as np

from pspca import PSPCA


def test_interface():
    """Test scikit-learn interface."""
    points = np.array(
        [
            [np.sqrt(0.5), np.sqrt(0.5), 0],
            [-np.sqrt(0.5), np.sqrt(0.5), 0],
        ]
    )

    dim = 3
    reducer = PSPCA(dim)
    reducer.fit(points)

    np.testing.assert_allclose(reducer.v, np.eye(3))

    t_points = reducer.transform(points)
    assert t_points.shape[0] == points.shape[0]
    assert t_points.shape[1] == dim

    np.testing.assert_allclose(reducer.fit_transform(points), t_points)

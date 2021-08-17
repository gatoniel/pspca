import numpy as np

from prospace_pca import ProspacePCA


def test_ProspacePCA():
    points = np.array(
        [
            [np.sqrt(0.5), np.sqrt(0.5), 0],
            [-np.sqrt(0.5), np.sqrt(0.5), 0],
        ]
    )

    dim = 3
    pspca = ProspacePCA(dim)
    pspca.fit(points)

    np.testing.assert_allclose(pspca.v, np.eye(3))

    t_points = pspca.transform(points)
    assert t_points.shape[0] == points.shape[0]
    assert t_points.shape[1] == dim

    np.testing.assert_allclose(pspca.fit_transform(points), t_points)

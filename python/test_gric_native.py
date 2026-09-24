"""
Test suite for GRIC Python native wrapper.
"""

import os
import sys
import numpy as np

# Ensure local python package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gric


def test_python_bindings():
    print(f"Testing GRIC Python wrapper with libgric version {gric.version()}...")

    # 1. Initialize Clusterer
    clusterer = gric.Clusterer(ndim=4, rlim=1.0, max_clusters=32)
    assert clusterer.ndim == 4
    assert clusterer.num_clusters == 0

    # 2. Test single frame feed
    f0 = np.array([0.0, 0.0, 0.0, 0.0])
    c0 = clusterer.feed(f0)
    assert c0 == 0
    assert clusterer.num_clusters == 1

    # Frame 1: close to origin -> matches cluster 0
    f1 = np.array([0.1, 0.1, 0.0, 0.0])
    c1 = clusterer.feed(f1)
    assert c1 == 0
    assert clusterer.num_clusters == 1

    # Frame 2: far from origin -> creates cluster 1
    f2 = np.array([5.0, 0.0, 0.0, 0.0])
    c2 = clusterer.feed(f2)
    assert c2 == 1
    assert clusterer.num_clusters == 2

    # 3. Test batch feed (scikit-learn style fit_predict)
    batch = np.array([
        [0.05, 0.05, 0.0, 0.0],  # near cluster 0
        [5.10, 0.00, 0.0, 0.0],  # near cluster 1
    ])
    labels = clusterer.fit_predict(batch)
    assert len(labels) == 2
    assert labels[0] == 0
    assert labels[1] == 1

    # 4. Test anchor and member count export
    anchors = clusterer.anchors
    assert anchors.shape == (2, 4)
    np.testing.assert_allclose(anchors[0], [0.0, 0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(anchors[1], [5.0, 0.0, 0.0, 0.0], atol=1e-5)

    members = clusterer.member_counts
    assert len(members) == 2
    assert members[0] >= 2
    assert members[1] >= 2

    # 5. Test context manager
    with gric.Clusterer(ndim=2, rlim=0.5) as cl:
        data = np.random.randn(50, 2)
        lbls = cl.fit_predict(data)
        assert len(lbls) == 50
        assert cl.num_clusters > 0

    clusterer.close()
    print("GRIC Python native wrapper tests passed successfully!")


if __name__ == "__main__":
    test_python_bindings()

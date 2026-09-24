"""
GRIC Python Native Library Interface.

Provides fast, zero-copy, in-memory clustering using the native libgric C engine.
Supports both single-frame real-time streaming and scikit-learn style batch processing.
"""

import ctypes
import os
import sys
from typing import Optional, Union, Tuple, List
import numpy as np


def _find_libgric() -> str:
    """Locate the compiled libgric shared library."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # In-tree build directories
        os.path.join(pkg_dir, "..", "build", "libgric.so"),
        os.path.join(pkg_dir, "libgric.so"),
        os.path.join(pkg_dir, "..", "libgric.so"),
        # System library locations
        "libgric.so",
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return "libgric.so"


class _GricClusterConfig(ctypes.Structure):
    _fields_ = [
        ("rlim", ctypes.c_double),
        ("maxnbclust", ctypes.c_int),
        ("maxnbfr", ctypes.c_long),
        ("tm_mixing_coeff", ctypes.c_double),
        ("use_double", ctypes.c_int),
        ("use_sq16", ctypes.c_int),
        ("use_eq16", ctypes.c_int),
        ("te4_mode", ctypes.c_int),
        ("te5_mode", ctypes.c_int),
        ("entropy_mode", ctypes.c_int),
        ("entropy_gate_bits", ctypes.c_double),
        ("pred_mode", ctypes.c_int),
        ("gprob_mode", ctypes.c_int),
        ("soft_bayesian_mode", ctypes.c_int),
        ("sparse_dcc_mode", ctypes.c_int),
        ("sparse_dcc_extra_evals", ctypes.c_int),
        ("maxcl_strategy", ctypes.c_int),
        ("discard_fraction", ctypes.c_double),
        ("ncpu", ctypes.c_int),
    ]


class _LibGric:
    _instance = None

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = _find_libgric()
        self.lib = ctypes.CDLL(lib_path)

        # Function signatures
        self.lib.gric_version.restype = ctypes.c_char_p
        self.lib.gric_version.argtypes = []

        self.lib.gric_cluster_config_default.restype = ctypes.c_int
        self.lib.gric_cluster_config_default.argtypes = [ctypes.POINTER(_GricClusterConfig)]

        self.lib.gric_cluster_create.restype = ctypes.c_void_p
        self.lib.gric_cluster_create.argtypes = [
            ctypes.POINTER(_GricClusterConfig),
            ctypes.c_size_t,
        ]

        self.lib.gric_cluster_create_simple.restype = ctypes.c_void_p
        self.lib.gric_cluster_create_simple.argtypes = [ctypes.c_size_t, ctypes.c_double]

        self.lib.gric_cluster_feed_frame.restype = ctypes.c_int
        self.lib.gric_cluster_feed_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
        ]

        self.lib.gric_cluster_feed_batch.restype = ctypes.c_int
        self.lib.gric_cluster_feed_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
        ]

        self.lib.gric_cluster_get_num_clusters.restype = ctypes.c_int64
        self.lib.gric_cluster_get_num_clusters.argtypes = [ctypes.c_void_p]

        self.lib.gric_cluster_get_anchors.restype = ctypes.c_int64
        self.lib.gric_cluster_get_anchors.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
        ]

        self.lib.gric_cluster_get_dcc.restype = ctypes.c_int
        self.lib.gric_cluster_get_dcc.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]

        self.lib.gric_cluster_reset.restype = ctypes.c_int
        self.lib.gric_cluster_reset.argtypes = [ctypes.c_void_p]

        self.lib.gric_cluster_destroy.restype = None
        self.lib.gric_cluster_destroy.argtypes = [ctypes.c_void_p]

    @classmethod
    def get(cls, lib_path: Optional[str] = None) -> "_LibGric":
        if cls._instance is None:
            cls._instance = cls(lib_path)
        return cls._instance


def version() -> str:
    """Return the GRIC C library version string."""
    return _LibGric.get().lib.gric_version().decode("utf-8")


class Clusterer:
    """
    High-performance clustering engine backed by native libgric.

    Parameters
    ----------
    ndim : int
        Dimensionality of input coordinate vectors.
    rlim : float, default=0.5
        Clustering radius threshold.
    max_clusters : int, default=256
        Maximum cluster capacity. Set to 0 for unlimited dynamic capacity.
    use_sq16 : bool, default=False
        Enable 16-bit scalar quantization pruning.
    use_eq16 : bool, default=False
        Enable 16-bit E8 lattice quantization pruning.
    te4_mode : bool, default=True
        Enable 4-point triangle inequality pruning.
    entropy_mode : bool, default=False
        Enable entropy-guided target search.
    ncpu : int, default=1
        Number of worker threads.
    lib_path : str, optional
        Path to custom libgric.so.
    """

    def __init__(
        self,
        ndim: int,
        rlim: float = 0.5,
        max_clusters: int = 256,
        use_sq16: bool = False,
        use_eq16: bool = False,
        te4_mode: bool = True,
        entropy_mode: bool = False,
        ncpu: int = 1,
        lib_path: Optional[str] = None,
    ):
        if ndim <= 0:
            raise ValueError(f"ndim must be positive, got {ndim}")

        self._ndim = int(ndim)
        self._gric = _LibGric.get(lib_path)

        cfg = _GricClusterConfig()
        self._gric.lib.gric_cluster_config_default(ctypes.byref(cfg))
        cfg.rlim = float(rlim)
        cfg.maxnbclust = int(max_clusters)
        cfg.use_sq16 = 1 if use_sq16 else 0
        cfg.use_eq16 = 1 if use_eq16 else 0
        cfg.te4_mode = 1 if te4_mode else 0
        cfg.entropy_mode = 1 if entropy_mode else 0
        cfg.ncpu = int(ncpu)

        self._handle = self._gric.lib.gric_cluster_create(ctypes.byref(cfg), self._ndim)
        if not self._handle:
            raise MemoryError("Failed to allocate native GRIC clusterer context")

    def __del__(self):
        self.close()

    def close(self):
        """Release all native C memory allocations."""
        if hasattr(self, "_handle") and self._handle:
            self._gric.lib.gric_cluster_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def ndim(self) -> int:
        """Dimensionality of the feature vector space."""
        return self._ndim

    @property
    def num_clusters(self) -> int:
        """Number of clusters currently discovered."""
        self._check_alive()
        return int(self._gric.lib.gric_cluster_get_num_clusters(self._handle))

    def feed(self, frame: Union[np.ndarray, List[float]]) -> int:
        """
        Process a single incoming frame in real-time.

        Parameters
        ----------
        frame : array-like of shape (ndim,)
            Coordinate vector of incoming sample.

        Returns
        -------
        int
            Assigned cluster ID.
        """
        self._check_alive()
        arr = np.ascontiguousarray(frame, dtype=np.float64)
        if arr.size != self._ndim:
            raise ValueError(f"Expected frame with {self._ndim} dimensions, got {arr.size}")

        out_cid = ctypes.c_int64(-1)
        data_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        status = self._gric.lib.gric_cluster_feed_frame(self._handle, data_ptr, ctypes.byref(out_cid))
        if status != 0:
            raise RuntimeError(f"gric_cluster_feed_frame failed with status code {status}")
        return int(out_cid.value)

    def feed_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Process a 2D batch of frames with zero-copy buffer passing.

        Parameters
        ----------
        X : array-like of shape (n_samples, ndim)
            2D matrix of input samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Assigned cluster indices.
        """
        self._check_alive()
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self._ndim:
            raise ValueError(f"Expected 2D array of shape (N, {self._ndim}), got {X_arr.shape}")

        num_frames = X_arr.shape[0]
        out_labels = np.empty(num_frames, dtype=np.int64)

        data_ptr = X_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_ptr = out_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        status = self._gric.lib.gric_cluster_feed_batch(self._handle, data_ptr, num_frames, out_ptr)
        if status != 0:
            raise RuntimeError(f"gric_cluster_feed_batch failed with status code {status}")
        return out_labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Scikit-learn compatible alias for feed_batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, ndim)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return self.feed_batch(X)

    @property
    def anchors(self) -> np.ndarray:
        """
        Array of shape (K, ndim) containing all discovered cluster centroids.
        """
        self._check_alive()
        K = self.num_clusters
        if K == 0:
            return np.empty((0, self._ndim), dtype=np.float64)

        coords = np.empty((K, self._ndim), dtype=np.float64)
        coords_ptr = coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        n_exported = self._gric.lib.gric_cluster_get_anchors(
            self._handle, coords_ptr, None, K
        )
        if n_exported < 0:
            raise RuntimeError("Failed to export cluster anchors")
        return coords[:n_exported]

    @property
    def member_counts(self) -> np.ndarray:
        """
        Array of shape (K,) containing member count per cluster.
        """
        self._check_alive()
        K = self.num_clusters
        if K == 0:
            return np.empty(0, dtype=np.int32)

        coords = np.empty((K, self._ndim), dtype=np.float64)
        members = np.empty(K, dtype=np.int32)
        coords_ptr = coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        members_ptr = members.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        n_exported = self._gric.lib.gric_cluster_get_anchors(
            self._handle, coords_ptr, members_ptr, K
        )
        if n_exported < 0:
            raise RuntimeError("Failed to export cluster member counts")
        return members[:n_exported]

    @property
    def dcc(self) -> np.ndarray:
        """
        Distance-between-cluster-centers (DCC) square matrix of shape (K, K).
        """
        self._check_alive()
        K = self.num_clusters
        if K == 0:
            return np.empty((0, 0), dtype=np.float64)

        dcc_matrix = np.empty((K, K), dtype=np.float64)
        dcc_ptr = dcc_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        status = self._gric.lib.gric_cluster_get_dcc(self._handle, dcc_ptr, K)
        if status != 0:
            raise RuntimeError("Failed to export DCC matrix")
        return dcc_matrix

    def reset(self):
        """Reset internal clustering state while retaining memory buffers."""
        self._check_alive()
        status = self._gric.lib.gric_cluster_reset(self._handle)
        if status != 0:
            raise RuntimeError("Failed to reset clusterer")

    def _check_alive(self):
        if not hasattr(self, "_handle") or not self._handle:
            raise RuntimeError("Clusterer session has been closed or was not initialized")

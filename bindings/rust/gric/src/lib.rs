//! # GRIC
//!
//! High-performance, streaming distance-based clustering and geometry engine.
//!
//! ## Example
//! ```no_run
//! use gric::{Clusterer, Config};
//!
//! fn main() -> Result<(), gric::Error> {
//!     let mut clusterer = Clusterer::new(4, Config::default().rlim(1.0))?;
//!     let f0 = [0.0, 0.0, 0.0, 0.0];
//!     let id = clusterer.feed(&f0)?;
//!     assert_eq!(id, 0);
//!     Ok(())
//! }
//! ```

use std::ffi::CStr;
use std::fmt;

pub use gric_sys::gric_cluster_config_t;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidParam,
    OutOfMemory,
    DimMismatch,
    CapacityExceeded,
    Generic(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidParam => write!(f, "Invalid parameter passed to GRIC"),
            Error::OutOfMemory => write!(f, "Out of memory in GRIC allocation"),
            Error::DimMismatch => write!(f, "Vector dimensionality mismatch"),
            Error::CapacityExceeded => write!(f, "Cluster capacity limit reached"),
            Error::Generic(code) => write!(f, "GRIC operation failed with error code {}", code),
        }
    }
}

impl std::error::Error for Error {}

fn check_status(code: i32) -> Result<(), Error> {
    match code {
        gric_sys::GRIC_SUCCESS => Ok(()),
        gric_sys::GRIC_ERR_INVALID_PARAM => Err(Error::InvalidParam),
        gric_sys::GRIC_ERR_OUT_OF_MEMORY => Err(Error::OutOfMemory),
        gric_sys::GRIC_ERR_DIM_MISMATCH => Err(Error::DimMismatch),
        gric_sys::GRIC_ERR_CAPACITY => Err(Error::CapacityExceeded),
        other => Err(Error::Generic(other)),
    }
}

/// Retrieve the version string of the underlying libgric library.
pub fn version() -> &'static str {
    unsafe {
        let ptr = gric_sys::gric_version();
        CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
    }
}

/// Configuration builder for clustering sessions.
#[derive(Debug, Clone)]
pub struct Config(pub gric_cluster_config_t);

impl Default for Config {
    fn default() -> Self {
        let mut raw = std::mem::MaybeUninit::uninit();
        unsafe {
            gric_sys::gric_cluster_config_default(raw.as_mut_ptr());
            Config(raw.assume_init())
        }
    }
}

impl Config {
    pub fn rlim(mut self, r: f64) -> Self {
        self.0.rlim = r;
        self
    }

    pub fn max_clusters(mut self, n: i32) -> Self {
        self.0.maxnbclust = n;
        self
    }

    pub fn sq16(mut self, enabled: bool) -> Self {
        self.0.use_sq16 = if enabled { 1 } else { 0 };
        self
    }

    pub fn eq16(mut self, enabled: bool) -> Self {
        self.0.use_eq16 = if enabled { 1 } else { 0 };
        self
    }

    pub fn ncpu(mut self, threads: i32) -> Self {
        self.0.ncpu = threads;
        self
    }
}

/// Safe RAII handle to an active clustering session.
pub struct Clusterer {
    ndim: usize,
    raw: *mut gric_sys::gric_cluster_t,
}

// Clusterer can be sent across threads
unsafe impl Send for Clusterer {}

impl Drop for Clusterer {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                gric_sys::gric_cluster_destroy(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

impl Clusterer {
    /// Create a new Clusterer with specific dimension and configuration.
    pub fn new(ndim: usize, config: Config) -> Result<Self, Error> {
        let raw = unsafe { gric_sys::gric_cluster_create(&config.0, ndim) };
        if raw.is_null() {
            return Err(Error::OutOfMemory);
        }
        Ok(Self { ndim, raw })
    }

    /// Simplified constructor with default settings and a specified radius.
    pub fn simple(ndim: usize, rlim: f64) -> Result<Self, Error> {
        let raw = unsafe { gric_sys::gric_cluster_create_simple(ndim, rlim) };
        if raw.is_null() {
            return Err(Error::OutOfMemory);
        }
        Ok(Self { ndim, raw })
    }

    /// Dimensionality of the vector space.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Number of clusters discovered so far.
    pub fn num_clusters(&self) -> usize {
        let k = unsafe { gric_sys::gric_cluster_get_num_clusters(self.raw) };
        if k < 0 {
            0
        } else {
            k as usize
        }
    }

    /// Process a single incoming coordinate frame.
    pub fn feed(&mut self, coords: &[f64]) -> Result<i64, Error> {
        if coords.len() != self.ndim {
            return Err(Error::DimMismatch);
        }
        let mut out_cid: i64 = -1;
        let st = unsafe {
            gric_sys::gric_cluster_feed_frame(self.raw, coords.as_ptr(), &mut out_cid)
        };
        check_status(st)?;
        Ok(out_cid)
    }

    /// Process a batch of contiguous frames.
    pub fn feed_batch(&mut self, flat_coords: &[f64], num_frames: usize) -> Result<Vec<i64>, Error> {
        if flat_coords.len() != num_frames * self.ndim {
            return Err(Error::DimMismatch);
        }
        let mut out = vec![-1i64; num_frames];
        let st = unsafe {
            gric_sys::gric_cluster_feed_batch(
                self.raw,
                flat_coords.as_ptr(),
                num_frames,
                out.as_mut_ptr(),
            )
        };
        check_status(st)?;
        Ok(out)
    }

    /// Export discovered cluster anchor vectors.
    pub fn anchors(&self) -> Vec<Vec<f64>> {
        let k = self.num_clusters();
        if k == 0 {
            return Vec::new();
        }
        let mut flat = vec![0.0f64; k * self.ndim];
        let n_exported = unsafe {
            gric_sys::gric_cluster_get_anchors(
                self.raw,
                flat.as_mut_ptr(),
                std::ptr::null_mut(),
                k,
            )
        };
        if n_exported <= 0 {
            return Vec::new();
        }

        flat.chunks(self.ndim).map(|chunk| chunk.to_vec()).collect()
    }

    /// Reset internal state while retaining buffer allocations.
    pub fn reset(&mut self) -> Result<(), Error> {
        let st = unsafe { gric_sys::gric_cluster_reset(self.raw) };
        check_status(st)
    }
}

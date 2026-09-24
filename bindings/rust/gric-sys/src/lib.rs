//! Raw FFI bindings to libgric.

use std::os::raw::{c_char, c_double, c_int, c_long};

pub const GRIC_SUCCESS: c_int = 0;
pub const GRIC_ERR_INVALID_PARAM: c_int = -1;
pub const GRIC_ERR_OUT_OF_MEMORY: c_int = -2;
pub const GRIC_ERR_DIM_MISMATCH: c_int = -3;
pub const GRIC_ERR_CAPACITY: c_int = -4;
pub const GRIC_ERR_GENERIC: c_int = -5;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gric_cluster_config_t {
    pub rlim: c_double,
    pub maxnbclust: c_int,
    pub maxnbfr: c_long,
    pub tm_mixing_coeff: c_double,
    pub use_double: c_int,
    pub use_sq16: c_int,
    pub use_eq16: c_int,
    pub te4_mode: c_int,
    pub te5_mode: c_int,
    pub entropy_mode: c_int,
    pub entropy_gate_bits: c_double,
    pub pred_mode: c_int,
    pub gprob_mode: c_int,
    pub soft_bayesian_mode: c_int,
    pub sparse_dcc_mode: c_int,
    pub sparse_dcc_extra_evals: c_int,
    pub maxcl_strategy: c_int,
    pub discard_fraction: c_double,
    pub ncpu: c_int,
}

#[repr(C)]
pub struct gric_cluster_ctx {
    _unused: [u8; 0],
}

pub type gric_cluster_t = gric_cluster_ctx;

extern "C" {
    pub fn gric_version() -> *const c_char;

    pub fn gric_cluster_config_default(cfg: *mut gric_cluster_config_t) -> c_int;

    pub fn gric_cluster_create(
        cfg: *const gric_cluster_config_t,
        ndim: usize,
    ) -> *mut gric_cluster_t;

    pub fn gric_cluster_create_simple(ndim: usize, rlim: c_double) -> *mut gric_cluster_t;

    pub fn gric_cluster_feed_frame(
        ctx: *mut gric_cluster_t,
        coords: *const c_double,
        out_cluster_id: *mut i64,
    ) -> c_int;

    pub fn gric_cluster_feed_batch(
        ctx: *mut gric_cluster_t,
        coords_flat: *const c_double,
        num_frames: usize,
        out_cluster_ids: *mut i64,
    ) -> c_int;

    pub fn gric_cluster_get_num_clusters(ctx: *const gric_cluster_t) -> i64;

    pub fn gric_cluster_get_anchors(
        ctx: *const gric_cluster_t,
        out_coords: *mut c_double,
        out_members: *mut c_int,
        max_anchors: usize,
    ) -> i64;

    pub fn gric_cluster_get_dcc(
        ctx: *const gric_cluster_t,
        out_dcc: *mut c_double,
        K: usize,
    ) -> c_int;

    pub fn gric_cluster_reset(ctx: *mut gric_cluster_t) -> c_int;

    pub fn gric_cluster_destroy(ctx: *mut gric_cluster_t);
}

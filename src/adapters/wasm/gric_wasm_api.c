/**
 * @file gric_wasm_api.c
 * @brief Legacy umbrella compilation unit for the GRIC WebAssembly API.
 *
 * This file formerly contained the single-tile, multi-tile, and k-NN WASM bridges.
 * They have been modularized into:
 *   - src/wasm/wasm_cluster.c
 *   - src/wasm/wasm_multitile.c
 *   - src/wasm/wasm_knn.c
 */

#include "wasm_internal.h"

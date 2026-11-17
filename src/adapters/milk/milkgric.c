/**
 * @file milkgric.c
 * @brief MILK_MODULE_RUNNABLES plugin registry for libmilkgric.so.
 */

#define MODULE_SHORTNAME_DEFAULT "gric"
#define MODULE_DESCRIPTION       "GRIC clustering module"

#include "milk_runnable.h"
#include "gric_fps_common.h"
#include "gric_knn_fps_common.h"
#include "gric_recon_fps_common.h"

MODULE_DEPS("milkCOREMODmemory");

static const MILK_RUNNABLE *const runnables[] = {
    &runnable_gric_cluster,
    &runnable_gric_knn,
    &runnable_gric_reconstruct,
};

MILK_MODULE_RUNNABLES(milkgric, _module_deps, runnables);

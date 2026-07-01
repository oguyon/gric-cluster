/**
 * @file cluster_scandist.h
 * @brief Declarations for pre-clustering distance scanning.
 *
 * Exposes the main routine to scan consecutive frame distances and
 * print statistical results to help select threshold limits.
 */

#ifndef CLUSTER_SCANDIST_H
#define CLUSTER_SCANDIST_H

#include "cluster_defs.h"

void run_scandist(
    ClusterConfig *config,
    char          *out_dir);

#endif // CLUSTER_SCANDIST_H

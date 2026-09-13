/**
 * @file cluster_cli.h
 * @brief Command-line interface parser and configuration initialization for gric-cluster.
 */

#ifndef CLUSTER_CLI_H
#define CLUSTER_CLI_H

#include "cluster_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * cluster_cli_init_config_defaults() - Populate ClusterConfig with default values.
 * @config: Pointer to ClusterConfig to initialize.
 */
void cluster_cli_init_config_defaults(
    ClusterConfig *config);

#include "gric_profile.h"

/**
 * cluster_cli_parse() - Parse CLI options, load profiles, and validate parameters.
 * @argc:        Argument count.
 * @argv:        Argument vector.
 * @config:      Pointer to initialized ClusterConfig.
 * @profile_out: Output pointer to loaded GricProfile (or NULL).
 * @cmdline_out: Output pointer to allocated command line string (or NULL).
 *
 * Return: 0 on success, 1 on error, 2 if help was displayed.
 */
int cluster_cli_parse(
    int            argc,
    char         **argv,
    ClusterConfig *config,
    GricProfile   *profile_out,
    char         **cmdline_out);

#ifdef __cplusplus
}
#endif

#endif // CLUSTER_CLI_H

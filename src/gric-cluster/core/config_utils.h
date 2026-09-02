#ifndef CONFIG_UTILS_H
#define CONFIG_UTILS_H

/**
 * @file config_utils.h
 * @brief Configuration parsing and management for the clustering engine.
 */

#include "cluster_defs.h"

/**
 * @brief Parse a single option key/value pair and apply it to ClusterConfig.
 * @param config Pointer to ClusterConfig to modify.
 * @param key    Option key string (with or without leading dash).
 * @param value  Option value string, or NULL for boolean flags.
 * @return 1 if value was consumed, 0 if flag was consumed, -1 on unknown option.
 */
int apply_option(
    ClusterConfig *config,
    const char    *key,
    const char    *value);

/**
 * @brief Read configuration parameters from a key-value file.
 * @param filename File path to configuration file.
 * @param config   Pointer to ClusterConfig to populate.
 * @return 0 on success, 1 on file open error.
 */
int read_config_file(
    const char    *filename,
    ClusterConfig *config);

/**
 * @brief Serialize the active configuration struct to a reproducible config file.
 * @param filename File path to write.
 * @param config   Pointer to ClusterConfig to serialize.
 * @return 0 on success, 1 on file open error.
 */
int write_config_file(
    const char    *filename,
    ClusterConfig *config);

#endif // CONFIG_UTILS_H

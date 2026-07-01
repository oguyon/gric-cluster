#ifndef CONFIG_UTILS_H
#define CONFIG_UTILS_H

#include "cluster_defs.h"

// Parse a single option key/value pair.
// Returns 1 if value was consumed, 0 if only key was used (flag), -1 on error/unknown.
int apply_option(ClusterConfig *config, const char *key, const char *value);

// Read configuration from file
int read_config_file(const char *filename, ClusterConfig *config);

// Write configuration to file
int write_config_file(const char *filename, ClusterConfig *config);

#endif

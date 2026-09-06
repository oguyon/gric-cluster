#ifndef PROBE_REPORT_H
#define PROBE_REPORT_H

/**
 * @file probe_report.h
 * @brief Terminal reporting and visualization for gric-probe.
 */

#include "probe_engine.h"
#include <stdio.h>

/**
 * @brief Print formatted ANSI terminal report summarizing dataset characteristics.
 */
void probe_report_print_terminal(
    FILE               *out,
    const ProbeResults *results);

/**
 * @brief Print shell environment exports for scripting integration.
 */
void probe_report_print_env(
    FILE               *out,
    const ProbeResults *results);

#endif // PROBE_REPORT_H

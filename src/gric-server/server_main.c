/**
 * @file server_main.c
 * @brief Entry point for gric-server: Native C desktop backend server for GRIC GUI.
 */

#define _GNU_SOURCE
#include "http_server.h"
#include "shared/cli_colors.h"
#include <libgen.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void handle_signal(
    int sig)
{
    (void)sig;
    server_stop();
}

static void print_help(
    const char *progname)
{
    printf("%sNAME%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %sgric-server%s - Native C micro-server for the GRIC Desktop GUI\n\n",
           ansi_bold_green, ansi_reset);

    printf("%sUSAGE%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s%s%s %s[options]%s\n\n",
           ansi_bold_green, progname, ansi_reset,
           ansi_color_grey, ansi_reset);

    printf("%sDESCRIPTION%s\n", ansi_bold_cyan, ansi_reset);
    printf("  Serves the GRIC interactive simulator web UI and provides a native REST API\n"
           "  for workspace file management and high-performance native CLI execution\n"
           "  (gric-cluster, gric-knn) without any external dependencies or Python runtime.\n\n");

    printf("%sOPTIONS%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s-p, --port%s %s<port>%s        HTTP listen port (default: 8080 or $GRIC_SERVER_PORT)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-d, --dir%s %s<path>%s         Workspace working directory (default: current directory)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-w, --docs%s %s<path>%s        HTML/JS documentation directory (default: auto-detected)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-v, --verbose%s            Enable verbose request logging\n",
           ansi_color_green, ansi_reset);
    printf("  %s-h, --help%s               Show this help message and exit\n\n",
           ansi_color_green, ansi_reset);

    cli_print_color_mode();
}

int main(
    int   argc,
    char *argv[])
{
    cli_colors_init();

    ServerConfig config;
    server_config_init(&config);

    /* Detect binary directory from /proc/self/exe */
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0)
    {
        exe_path[len] = '\0';
        char *dir = dirname(exe_path);
        if (dir)
        {
            strncpy(config.server_bin_dir, dir, sizeof(config.server_bin_dir) - 1);
        }
    }

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_help(argv[0]);
            return 0;
        }
        else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0) && i + 1 < argc)
        {
            config.port = atoi(argv[++i]);
        }
        else if ((strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--dir") == 0) && i + 1 < argc)
        {
            char resolved[PATH_MAX];
            if (realpath(argv[++i], resolved))
            {
                snprintf(config.workdir, sizeof(config.workdir), "%s", resolved);
            }
            else
            {
                snprintf(config.workdir, sizeof(config.workdir), "%s", argv[i]);
            }
        }
        else if ((strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--docs") == 0) && i + 1 < argc)
        {
            char resolved[PATH_MAX];
            if (realpath(argv[++i], resolved))
            {
                snprintf(config.docs_dir, sizeof(config.docs_dir), "%s", resolved);
            }
            else
            {
                snprintf(config.docs_dir, sizeof(config.docs_dir), "%s", argv[i]);
            }
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0)
        {
            config.verbose = 1;
        }
        else
        {
            fprintf(stderr, "%sError: Unknown option '%s'%s\n",
                    ansi_color_red, argv[i], ansi_reset);
            print_help(argv[0]);
            return 1;
        }
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    return server_run(&config);
}

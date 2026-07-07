/**
 * @file benchmark_utils.c
 * @brief Utility helper functions for the benchmarking tool.
 *
 * Provides functions to run commands with output redirection, rebuild the
 * project, parse performance metrics, and handle CLI arguments for benchmarks.
 *
 * Main Functions:
 * - run_command_redirect: Safely executes shell commands and redirects stdout/stderr.
 * - rebuild_project: Programmatically runs CMake and Make to rebuild the suite.
 * - split_args: Helper to parse and split command line arguments.
 * - parse_metrics: Extracts and parses clustering performance metrics from log outputs.
 */
#include "benchmark.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static const char *known_patterns[] =
{
    "2Dspiral",
    "2Dcircle-shuffle",
    "2Dspiral-shuffle",
    "2Drand",
    "3Drand",
    "2DcircleP10n",
    "3Dspiral",
    "3Dstar",
    "3Dconcentric",
    "5Dtree",
    "3Dconcentric_dense"
};
#define KNOWN_PATTERNS_COUNT 11

/**
 * @brief Print usage/help information.
 *
 * @param progname The name of the executable.
 */
void print_help(
    const char *progname)
{
    printf("%sNAME%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %sgric-benchmark%s - Run performance benchmarks on gric-cluster\n\n",
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET);

    printf("%sUSAGE%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s%s%s %s[options]%s\n\n",
           ANSI_BOLD_GREEN, progname, ANSI_COLOR_RESET, ANSI_COLOR_GREY, ANSI_COLOR_RESET);

    printf("%sDESCRIPTION%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  Runs performance benchmarks on the gric-cluster algorithm.\n");
    printf("  All input files (txt points files, mp4 videos) are read/generated in 'benchmarks'.\n");
    printf("  All output files (logs, png plots, and the benchmark summary md file) are written\n");
    printf("  to 'benchmarks-out', which is created automatically if it does not exist.\n\n");

    printf("%sOPTIONS%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s-h, --help%s            Show this help message\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    printf("  %s-n, --nsamples%s %s<N>%s    Set number of samples "
           "(%sDefault:%s%s 20000%s)\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET,
           ANSI_COLOR_CYAN, ANSI_COLOR_RESET, ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("  %s-r, --rlim%s %s<R>%s        Set radius limit "
           "(%sDefault:%s%s 0.10%s)\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET,
           ANSI_COLOR_CYAN, ANSI_COLOR_RESET, ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("  %s-p, --pattern%s %s<name>%s  Select pattern (can be used multiple times).\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);
    printf("                        If not set, runs all. Available:\n");
    for (int ii = 0; ii < KNOWN_PATTERNS_COUNT; ii++)
    {
        printf("                        - %s\n", known_patterns[ii]);
    }
    printf("  %s-f, --file%s %s<file>%s     Load test patterns from the specified file.\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);
    printf("  %s-t, --type%s %s<type>%s     Select input type: txt (%sDefault%s), "
           "mp4, or stream\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET,
           ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("  %s-o, --options%s %s<str>%s   Pass additional options to gric-cluster.\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);
    printf("                        Can be used multiple times to add more options.\n");
    printf("  %s-mp4r%s                 Re-use mp4/txt files if they already exist\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    printf("  %s-maxcl%s %s<N>%s            Set max number of clusters "
           "(%sDefault:%s%s 1000%s)\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET,
           ANSI_COLOR_CYAN, ANSI_COLOR_RESET, ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("  %s-maxim%s %s<N>%s            Set max number of frames to process "
           "(%sDefault:%s%s 100000%s)\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET,
           ANSI_COLOR_CYAN, ANSI_COLOR_RESET, ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("  %s-b, --build%s           Rebuild project before running benchmarks\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    printf("  %s-entropy%s              Enable Shannon entropy-reduction target selection\n\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET);

    printf("%sEXAMPLES%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s$%s %s%s%s -p 2Dspiral -t mp4\n", ANSI_COLOR_GREY, ANSI_COLOR_RESET,
           ANSI_BOLD_GREEN, progname, ANSI_COLOR_RESET);
    printf("  %s$%s %s%s%s -n 1000 -r 0.05\n", ANSI_COLOR_GREY, ANSI_COLOR_RESET,
           ANSI_BOLD_GREEN, progname, ANSI_COLOR_RESET);
    printf("  %s$%s %s%s%s -f custom_tests.txt -o \"-gprob\"\n",
           ANSI_COLOR_GREY, ANSI_COLOR_RESET, ANSI_BOLD_GREEN, progname, ANSI_COLOR_RESET);

    printf("\n%sCOLOR MODE%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  ANSI colors are enabled by default.\n");
    printf("  Set the NO_COLOR environment variable to disable colored output.\n");
} // print_help

/**
 * @brief Run a command and redirect its stdout/stderr to a file.
 *
 * @param path     Executable path/name.
 * @param argv     NULL-terminated array of arguments.
 * @param log_path Path to the log file (or "/dev/null").
 * @return 0 on success, non-zero on failure.
 */
int run_command_redirect(
    const char  *path,
    char *const  argv[],
    const char  *log_path)
{
    /* Print the command in color to stdout */
    printf("%sCMD:%s", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    for (int ii = 0; argv[ii] != NULL; ii++)
    {
        if (strchr(argv[ii], ' ') != NULL)
        {
            printf(" %s\"%s\"%s", ANSI_COLOR_GREEN, argv[ii], ANSI_COLOR_RESET);
        }
        else
        {
            printf(" %s%s%s", ANSI_COLOR_GREEN, argv[ii], ANSI_COLOR_RESET);
        }
    }
    printf("\n");

    pid_t pid = fork();
    if (pid < 0)
    {
        perror("fork");
        return -1;
    }

    if (pid == 0)
    {
        /* Child process */
        int fd = open(log_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0)
        {
            perror("open log file");
            exit(1);
        }
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        close(fd);

        execvp(path, argv);
        perror("execvp");
        exit(1);
    }

    /* Parent process */
    int status;
    if (waitpid(pid, &status, 0) < 0)
    {
        perror("waitpid");
        return -1;
    }

    if (WIFEXITED(status))
    {
        return WEXITSTATUS(status);
    }
    return -1;
} // run_command_redirect

/**
 * @brief Rebuild the project using cmake and make.
 *
 * @param bin_dir The directory where build files are generated.
 * @return 0 on success, non-zero on failure.
 */
int rebuild_project(
    const char *bin_dir)
{
    printf("Building project...\n");

    char orig_cwd[1024];
    if (getcwd(orig_cwd, sizeof(orig_cwd)) == NULL)
    {
        perror("getcwd");
        return -1;
    }

    struct stat st = {0};
    if (stat(bin_dir, &st) == -1)
    {
        if (mkdir(bin_dir, 0755) != 0)
        {
            perror("mkdir bin_dir");
            return -1;
        }
    }

    if (chdir(bin_dir) != 0)
    {
        perror("chdir bin_dir");
        return -1;
    }

    char *cmake_argv[] = {"cmake", "..", "-DCMAKE_BUILD_TYPE=Release", NULL};
    int cmake_status = run_command_redirect("cmake", cmake_argv, "/dev/null");
    if (cmake_status != 0)
    {
        fprintf(stderr, "Error: cmake configuration failed (exit code %d)\n", cmake_status);
        if (chdir(orig_cwd) != 0)
        {
            perror("chdir back");
        }
        return -1;
    }

    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc < 1)
    {
        nproc = 1;
    }
    char j_flag[32];
    snprintf(j_flag, sizeof(j_flag), "-j%ld", nproc);

    char *make_argv[] = {"make", j_flag, NULL};
    int make_status = run_command_redirect("make", make_argv, "/dev/null");
    if (make_status != 0)
    {
        fprintf(stderr, "Error: make compilation failed (exit code %d)\n", make_status);
        if (chdir(orig_cwd) != 0)
        {
            perror("chdir back");
        }
        return -1;
    }

    if (chdir(orig_cwd) != 0)
    {
        perror("chdir back");
        return -1;
    }

    printf("Build completed successfully.\n");
    return 0;
} // rebuild_project

/**
 * @brief Split space-separated option string into separate arguments.
 *
 * @param str      The space-separated options string.
 * @param argv     The arguments array to populate.
 * @param argc     Pointer to the argument count.
 * @param max_args Maximum capacity of argv array.
 */
void split_args(
    const char  *str,
    char        *argv[],
    int         *argc,
    int          max_args)
{
    char *dup = strdup(str);
    if (dup == NULL)
    {
        return;
    }

    char *token = strtok(dup, " \t\r\n");
    while (token != NULL && *argc < max_args - 1)
    {
        argv[*argc] = strdup(token);
        if (argv[*argc] == NULL)
        {
            break;
        }
        (*argc)++;
        token = strtok(NULL, " \t\r\n");
    }
    free(dup);
} // split_args

/**
 * @brief Parse the benchmark log file to extract metrics.
 *
 * @param log_path     Path to the log file.
 * @param out_time     Buffer for processing time.
 * @param out_dists    Buffer for framedist calls.
 * @param out_clusters Buffer for total clusters.
 * @param out_mem      Buffer for maximum resident set size.
 */
void parse_metrics(
    const char  *log_path,
    char        *out_time,
    char        *out_dists,
    char        *out_dists_sample,
    char        *out_dists_intercluster,
    char        *out_clusters,
    char        *out_mem)
{
    FILE *fp = fopen(log_path, "r");
    if (fp == NULL)
    {
        strcpy(out_time, "N/A");
        strcpy(out_dists, "N/A");
        strcpy(out_dists_sample, "N/A");
        strcpy(out_dists_intercluster, "N/A");
        strcpy(out_clusters, "N/A");
        strcpy(out_mem, "N/A");
        return;
    }

    strcpy(out_time, "N/A");
    strcpy(out_dists, "N/A");
    strcpy(out_dists_sample, "N/A");
    strcpy(out_dists_intercluster, "N/A");
    strcpy(out_clusters, "N/A");
    strcpy(out_mem, "N/A");

    char line[1024];
    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (strstr(line, "Processing time:") != NULL)
        {
            char val[128] = "";
            if (sscanf(line, "Processing time: %127s", val) == 1)
            {
                strncpy(out_time, val, 63);
                out_time[63] = '\0';
            }
        }
        else if (strstr(line, "Framedist calls:") != NULL)
        {
            char val_tot[64] = "";
            char val_sample[64] = "";
            char val_inter[64] = "";
            int parsed = sscanf(line, "Framedist calls: %63s (sample-to-cluster: %63[^,], inter-cluster: %63[^)])",
                                val_tot, val_sample, val_inter);
            if (parsed >= 1)
            {
                strncpy(out_dists, val_tot, 63);
                out_dists[63] = '\0';
                if (parsed >= 2 && val_sample[0])
                {
                    strncpy(out_dists_sample, val_sample, 63);
                    out_dists_sample[63] = '\0';
                }
                if (parsed >= 3 && val_inter[0])
                {
                    strncpy(out_dists_intercluster, val_inter, 63);
                    out_dists_intercluster[63] = '\0';
                }
            }
        }
        else if (strstr(line, "Total clusters:") != NULL)
        {
            char val[128] = "";
            if (sscanf(line, "Total clusters: %127s", val) == 1)
            {
                strncpy(out_clusters, val, 63);
                out_clusters[63] = '\0';
            }
        }
        else if (strstr(line, "Maximum resident set size") != NULL)
        {
            char *colon = strchr(line, ':');
            if (colon != NULL)
            {
                char val[128] = "";
                if (sscanf(colon + 1, "%127s", val) == 1)
                {
                    strncpy(out_mem, val, 63);
                    out_mem[63] = '\0';
                }
            }
        }
    }
    fclose(fp);
} // parse_metrics

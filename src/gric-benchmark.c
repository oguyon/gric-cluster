#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAX_PATTERNS 32
#define MAX_OPTIONS  64

static const char *ansi_color_green = "";
static const char *ansi_color_reset = "";
static const char *ansi_bold = "";
static const char *ansi_bold_cyan = "";
static const char *ansi_bold_green = "";
static const char *ansi_color_magenta = "";
static const char *ansi_color_cyan = "";
static const char *ansi_color_grey = "";

#define ANSI_COLOR_GREEN   ansi_color_green
#define ANSI_COLOR_RESET   ansi_color_reset
#define ANSI_BOLD          ansi_bold
#define ANSI_BOLD_CYAN     ansi_bold_cyan
#define ANSI_BOLD_GREEN    ansi_bold_green
#define ANSI_COLOR_MAGENTA ansi_color_magenta
#define ANSI_COLOR_CYAN    ansi_color_cyan
#define ANSI_COLOR_GREY    ansi_color_grey

/**
 * @brief Initialize colors if NO_COLOR environment variable is not present.
 */
static void init_colors(void)
{
    const char *no_color = getenv("NO_COLOR");

    if (no_color == NULL)
    {
        ansi_color_green = "\x1b[32m";
        ansi_color_reset = "\x1b[0m";
        ansi_bold = "\x1b[1m";
        ansi_bold_cyan = "\x1b[1;36m";
        ansi_bold_green = "\x1b[1;32m";
        ansi_color_magenta = "\x1b[35m";
        ansi_color_cyan = "\x1b[36m";
        ansi_color_grey = "\x1b[90m";
    }
} // init_colors

static const char *all_patterns[] =
{
    "2Dspiral",
    "2Dcircle-shuffle",
    "2Dspiral-shuffle",
    "2Drand",
    "3Drand",
    "2DcircleP10n",
    "3Dspiral"
};
#define ALL_PATTERNS_COUNT 7

typedef struct
{
    int   nsamples;
    char  rlim[32];
    int   rlim_set;
    int   maxcl;
    int   maxim;
    int   maxim_set;
    char  type[32];
    int   reuse_mp4;
    char *patterns[MAX_PATTERNS];
    int   pattern_count;
    char *extra_options[MAX_OPTIONS];
    int   extra_options_count;
    int   build_first;
} BenchmarkConfig;

/**
 * @brief Initialize configuration with default values.
 *
 * @param config Pointer to the BenchmarkConfig struct.
 */
static void init_config(
    BenchmarkConfig *config)
{
    config->nsamples = 20000;
    strcpy(config->rlim, "0.10");
    config->rlim_set = 0;
    config->maxcl = 1000;
    config->maxim = 100000;
    config->maxim_set = 0;
    strcpy(config->type, "txt");
    config->reuse_mp4 = 0;
    config->pattern_count = 0;
    config->extra_options_count = 0;
    config->build_first = 0;
} // init_config

/**
 * @brief Print usage/help information.
 *
 * @param progname The name of the executable.
 */
static void print_help(
    const char *progname)
{
    printf("%sNAME%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  gric-benchmark - Run performance benchmarks on gric-cluster\n\n");

    printf("%sUSAGE%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s%s%s %s[options]%s\n\n",
           ANSI_BOLD_GREEN, progname, ANSI_COLOR_RESET, ANSI_COLOR_GREY, ANSI_COLOR_RESET);

    printf("%sDESCRIPTION%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  Runs performance benchmarks on the gric-cluster algorithm.\n");
    printf("  All output files (txt points files, mp4 videos, logs, png plots, and the\n");
    printf("  benchmark summary md file) are written to the 'benchmarks' subdirectory,\n");
    printf("  which is created automatically if it does not exist.\n\n");

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
    for (int ii = 0; ii < ALL_PATTERNS_COUNT; ii++)
    {
        printf("                        - %s\n", all_patterns[ii]);
    }
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
    printf("  %s-b, --build%s           Rebuild project before running benchmarks\n\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET);

    printf("%sEXAMPLES%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s$%s %s%s%s -p 2Dspiral -t mp4\n", ANSI_COLOR_GREY, ANSI_COLOR_RESET,
           ANSI_BOLD_GREEN, progname, ANSI_COLOR_RESET);
    printf("  %s$%s %s%s%s -n 1000 -r 0.05\n", ANSI_COLOR_GREY, ANSI_COLOR_RESET,
           ANSI_BOLD_GREEN, progname, ANSI_COLOR_RESET);
    printf("  %s$%s %s%s%s -p 2Dspiral-shuffle -o \"-gprob\" -o \"-fmatcha 1.0\"\n",
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
static int run_command_redirect(
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
 * @return 0 on success, non-zero on failure.
 */
static int rebuild_project(
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
static void split_args(
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
static void parse_metrics(
    const char  *log_path,
    char        *out_time,
    char        *out_dists,
    char        *out_clusters,
    char        *out_mem)
{
    FILE *fp = fopen(log_path, "r");
    if (fp == NULL)
    {
        strcpy(out_time, "N/A");
        strcpy(out_dists, "N/A");
        strcpy(out_clusters, "N/A");
        strcpy(out_mem, "N/A");
        return;
    }

    strcpy(out_time, "N/A");
    strcpy(out_dists, "N/A");
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
            char val[128] = "";
            if (sscanf(line, "Framedist calls: %127s", val) == 1)
            {
                strncpy(out_dists, val, 63);
                out_dists[63] = '\0';
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

int main(
    int   argc,
    char *argv[])
{
    BenchmarkConfig config;
    init_config(&config);
    init_colors();

    static struct option long_options[] =
    {
        {"help",     no_argument,       0, 'h'},
        {"nsamples", required_argument, 0, 'n'},
        {"rlim",     required_argument, 0, 'r'},
        {"pattern",  required_argument, 0, 'p'},
        {"type",     required_argument, 0, 't'},
        {"options",  required_argument, 0, 'o'},
        {"build",    no_argument,       0, 'b'},
        {"mp4r",     no_argument,       0, 1001},
        {"maxcl",    required_argument, 0, 1002},
        {"maxim",    required_argument, 0, 1003},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long_only(argc, argv, "hn:r:p:t:o:b", long_options, &option_index)) != -1)
    {
        switch (opt)
        {
            case 'h':
                print_help(argv[0]);
                return 0;
            case 'n':
                config.nsamples = atoi(optarg);
                break;
            case 'r':
                strncpy(config.rlim, optarg, sizeof(config.rlim) - 1);
                config.rlim[sizeof(config.rlim) - 1] = '\0';
                config.rlim_set = 1;
                break;
            case 'p':
                if (config.pattern_count < MAX_PATTERNS)
                {
                    config.patterns[config.pattern_count++] = optarg;
                }
                break;
            case 't':
                strncpy(config.type, optarg, sizeof(config.type) - 1);
                config.type[sizeof(config.type) - 1] = '\0';
                break;
            case 'o':
                if (config.extra_options_count < MAX_OPTIONS)
                {
                    config.extra_options[config.extra_options_count++] = optarg;
                }
                break;
            case 'b':
                config.build_first = 1;
                break;
            case 1001: /* -mp4r */
                config.reuse_mp4 = 1;
                break;
            case 1002: /* -maxcl */
                config.maxcl = atoi(optarg);
                break;
            case 1003: /* -maxim */
                config.maxim = atoi(optarg);
                config.maxim_set = 1;
                break;
            default:
                fprintf(stderr, "Error: Unknown option\n");
                print_help(argv[0]);
                return 1;
        }
    }

    /* Validate type */
    if (strcmp(config.type, "txt") != 0 &&
        strcmp(config.type, "mp4") != 0 &&
        strcmp(config.type, "stream") != 0)
    {
        fprintf(stderr, "Error: Invalid type '%s'. Use 'txt', 'mp4', or 'stream'.\n", config.type);
        return 1;
    }

    /* Default patterns if none selected */
    if (config.pattern_count == 0)
    {
        for (int ii = 0; ii < ALL_PATTERNS_COUNT; ii++)
        {
            config.patterns[config.pattern_count++] = (char *)all_patterns[ii];
        }
    }

    /* Resolve binary directory paths relative to argv[0] */
    char bin_dir[1024] = "";
    char *last_slash = strrchr(argv[0], '/');
    if (last_slash != NULL)
    {
        size_t len = last_slash - argv[0] + 1;
        if (len < sizeof(bin_dir))
        {
            strncpy(bin_dir, argv[0], len);
            bin_dir[len] = '\0';
        }
    }
    else
    {
        strcpy(bin_dir, "../build/");
    }

    /* Build project if requested */
    if (config.build_first)
    {
        if (rebuild_project(bin_dir) != 0)
        {
            return 1;
        }
    }

    /* Check if current directory ends with "/benchmarks" */
    char cwd[1024];
    int in_benchmarks_dir = 0;
    if (getcwd(cwd, sizeof(cwd)) != NULL)
    {
        size_t len = strlen(cwd);
        if (len >= 11 && strcmp(cwd + len - 11, "/benchmarks") == 0)
        {
            in_benchmarks_dir = 1;
        }
    }

    char prefix[256] = "";
    if (!in_benchmarks_dir)
    {
        strcpy(prefix, "benchmarks/");
        mkdir("benchmarks", 0755);
    }

    char mkseq_path[1024];
    char rnuc_path[1024];
    char clplot_path[1024];
    char txt2mp4_path[1024];

    snprintf(mkseq_path, sizeof(mkseq_path), "%sgric-mktxtseq", bin_dir);
    snprintf(rnuc_path, sizeof(rnuc_path), "%sgric-cluster", bin_dir);
    snprintf(clplot_path, sizeof(clplot_path), "%sgric-plot", bin_dir);
    snprintf(txt2mp4_path, sizeof(txt2mp4_path), "%sgric-ascii-spot-2-video", bin_dir);

    /* Verify that required binaries exist */
    if (access(mkseq_path, X_OK) != 0 || access(rnuc_path, X_OK) != 0)
    {
        fprintf(stderr, "Error: Required binaries not found or not executable.\n");
        fprintf(stderr, "  %s\n  %s\n", mkseq_path, rnuc_path);
        fprintf(stderr, "Please run with --build flag or compile using CMake first.\n");
        return 1;
    }

    /* Create target output directories */
    char log_dir[512];
    snprintf(log_dir, sizeof(log_dir), "%sbenchmark_out", prefix);
    mkdir(log_dir, 0755);

    char cluster_out_dir_parent[512];
    snprintf(cluster_out_dir_parent, sizeof(cluster_out_dir_parent), "%sclusteroutdir", prefix);
    mkdir(cluster_out_dir_parent, 0755);

    /* Sync MAXIM with nsamples if not explicitly configured */
    if (!config.maxim_set)
    {
        config.maxim = config.nsamples;
    }

    char nbsample_str[32];
    char maxcl_str[32];
    char maxim_str[32];
    snprintf(nbsample_str, sizeof(nbsample_str), "%d", config.nsamples);
    snprintf(maxcl_str, sizeof(maxcl_str), "%d", config.maxcl);
    snprintf(maxim_str, sizeof(maxim_str), "%d", config.maxim);

    /* Initialize summary file header if missing */
    char summary_path[512];
    snprintf(summary_path, sizeof(summary_path), "%sbenchmark_summary.md", prefix);

    FILE *sum_fp = fopen(summary_path, "r");
    if (sum_fp == NULL)
    {
        sum_fp = fopen(summary_path, "w");
        if (sum_fp != NULL)
        {
            fprintf(sum_fp,
                    "| Pattern | Type | Algo | Time (ms) | "
                    "Dist Calls | Clusters | Memory (KB) |\n");
            fprintf(sum_fp, "|---|---|---|---|---|---|---|\n");
            fclose(sum_fp);
        }
    }
    else
    {
        fclose(sum_fp);
    }

    /* Run selected benchmarks */
    for (int ii = 0; ii < config.pattern_count; ii++)
    {
        const char *pattern = config.patterns[ii];
        printf("========================================================\n");
        printf("Benchmark: Pattern=%s Type=%s Algo=gric\n", pattern, config.type);

        /* 1. Generate text data */
        char txt_file[512];
        snprintf(txt_file, sizeof(txt_file), "%s%s.txt", prefix, pattern);

        int skip_gen = 0;
        if (config.reuse_mp4 && access(txt_file, F_OK) == 0)
        {
            printf("Re-using existing data file: %s\n", txt_file);
            skip_gen = 1;
        }

        if (!skip_gen)
        {
            printf("Generating data for pattern: %s\n", pattern);
            char *gen_args[16];
            int gen_argc = 0;
            gen_args[gen_argc++] = mkseq_path;
            gen_args[gen_argc++] = nbsample_str;
            gen_args[gen_argc++] = txt_file;

            if (strcmp(pattern, "2Dspiral") == 0)
            {
                gen_args[gen_argc++] = "2Dspiral";
            }
            else if (strcmp(pattern, "2Dcircle-shuffle") == 0)
            {
                gen_args[gen_argc++] = "2Dcircle";
                gen_args[gen_argc++] = "-shuffle";
            }
            else if (strcmp(pattern, "2Dspiral-shuffle") == 0)
            {
                gen_args[gen_argc++] = "2Dspiral";
                gen_args[gen_argc++] = "-shuffle";
            }
            else if (strcmp(pattern, "2Drand") == 0)
            {
                gen_args[gen_argc++] = "2Drand";
            }
            else if (strcmp(pattern, "3Drand") == 0)
            {
                gen_args[gen_argc++] = "3Drand";
            }
            else if (strcmp(pattern, "2DcircleP10n") == 0)
            {
                gen_args[gen_argc++] = "2Dcircle10";
                gen_args[gen_argc++] = "-noise";
                gen_args[gen_argc++] = "0.04";
            }
            else if (strcmp(pattern, "3Dspiral") == 0)
            {
                gen_args[gen_argc++] = "3Dspiral";
            }
            else
            {
                fprintf(stderr, "Error: Unknown pattern '%s'\n", pattern);
                continue;
            }
            gen_args[gen_argc] = NULL;

            int gen_status = run_command_redirect(mkseq_path, gen_args, "/dev/null");
            if (gen_status != 0)
            {
                fprintf(stderr, "Error: Data generation failed (exit code %d)\n", gen_status);
                continue;
            }
        }

        /* 2. Prepare Input File (TXT or MP4 conversion) */
        char input_file[512];
        if (strcmp(config.type, "txt") == 0)
        {
            strcpy(input_file, txt_file);
        }
        else if (strcmp(config.type, "mp4") == 0)
        {
            snprintf(input_file, sizeof(input_file), "%s%s.mp4", prefix, pattern);
            int skip_vid = 0;
            if (config.reuse_mp4 && access(input_file, F_OK) == 0)
            {
                printf("Re-using existing video file: %s\n", input_file);
                skip_vid = 1;
            }

            if (!skip_vid)
            {
                printf("Converting %s to %s...\n", txt_file, input_file);
                char *vid_args[] =
                {
                    txt2mp4_path,
                    "64",        /* VID_SIZE */
                    "0.1",       /* VID_ALPHA */
                    txt_file,
                    input_file,
                    "0.0",       /* VID_NOISE */
                    nbsample_str,
                    NULL
                };
                int vid_status = run_command_redirect(txt2mp4_path, vid_args, "/dev/null");
                if (vid_status != 0)
                {
                    fprintf(stderr, "Error: Video conversion failed (exit code %d)\n", vid_status);
                    continue;
                }
            }
        }
        else if (strcmp(config.type, "stream") == 0)
        {
            strcpy(input_file, pattern);
        }

        /* 3. Determine Radius Limit */
        char cur_rlim[32];
        if (strcmp(config.type, "mp4") == 0 || strcmp(config.type, "stream") == 0)
        {
            if (!config.rlim_set)
            {
                /* Default rlim for video/stream is 1000.0 */
                strcpy(cur_rlim, "1000.0");
            }
            else
            {
                strcpy(cur_rlim, config.rlim);
            }
        }
        else
        {
            strcpy(cur_rlim, config.rlim);
        }

        /* 4. Construct and Run gric-cluster Command */
        char log_file[512];
        snprintf(log_file, sizeof(log_file),
                 "%sbenchmark_out/%s_%s_gric.log",
                 prefix, pattern, config.type);

        char out_dir[512];
        snprintf(out_dir, sizeof(out_dir), "%s%s.cluster.out", prefix, pattern);

        printf("Running gric-cluster on %s (rlim=%s)...\n", input_file, cur_rlim);

        char *cluster_args[256];
        int cluster_argc = 0;

        cluster_args[cluster_argc++] = "/usr/bin/time";
        cluster_args[cluster_argc++] = "-v";
        cluster_args[cluster_argc++] = rnuc_path;
        cluster_args[cluster_argc++] = cur_rlim;
        cluster_args[cluster_argc++] = "-maxcl";
        cluster_args[cluster_argc++] = maxcl_str;
        cluster_args[cluster_argc++] = "-maxim";
        cluster_args[cluster_argc++] = maxim_str;
        cluster_args[cluster_argc++] = "-outdir";
        cluster_args[cluster_argc++] = out_dir;
        cluster_args[cluster_argc++] = "-clustered";

        for (int jj = 0; jj < config.extra_options_count; jj++)
        {
            /* Reserve 3 slots for -stream, input_file, and the NULL terminator */
            split_args(config.extra_options[jj], cluster_args, &cluster_argc, 256 - 3);
        }

        if (strcmp(config.type, "stream") == 0)
        {
            cluster_args[cluster_argc++] = "-stream";
        }

        cluster_args[cluster_argc++] = input_file;
        cluster_args[cluster_argc] = NULL;

        int run_status = run_command_redirect("/usr/bin/time", cluster_args, log_file);
        if (run_status != 0)
        {
            fprintf(stderr, "Warning: gric-cluster exited with status %d\n", run_status);
        }

        /* Clean up split arguments memory */
        /* Start checking from index 11 which is the start of split args */
        for (int jj = 11; jj < cluster_argc; jj++)
        {
            /* Only free if it's not one of static args or input_file */
            if (cluster_args[jj] != NULL &&
                strcmp(cluster_args[jj], "-stream") != 0 &&
                cluster_args[jj] != input_file)
            {
                free(cluster_args[jj]);
            }
        }

        /* 5. Optional: Plot result for txt inputs */
        if (strcmp(config.type, "txt") == 0 && access(clplot_path, X_OK) == 0)
        {
            char cluster_log[1024];
            snprintf(cluster_log, sizeof(cluster_log), "%s/cluster_run.log", out_dir);
            char *plot_args[] = {clplot_path, input_file, cluster_log, NULL};
            run_command_redirect(clplot_path, plot_args, "/dev/null");
        }

        /* 6. Extract Metrics and Log to Summary */
        char m_time[64], m_dists[64], m_clusters[64], m_mem[64];
        parse_metrics(log_file, m_time, m_dists, m_clusters, m_mem);

        printf("Result: Time=%sms, Dists=%s, Clusters=%s, Mem=%sKB\n",
               m_time, m_dists, m_clusters, m_mem);

        sum_fp = fopen(summary_path, "a");
        if (sum_fp != NULL)
        {
            fprintf(sum_fp, "| %s | %s | gric | %s | %s | %s | %s |\n",
                    pattern, config.type, m_time, m_dists, m_clusters, m_mem);
            fclose(sum_fp);
        }
    }

    printf("========================================================\n");
    printf("Benchmarks Complete. Summary appended to %s\n", summary_path);

    return 0;
} // main

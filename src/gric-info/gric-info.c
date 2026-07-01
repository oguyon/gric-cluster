#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *ansi_color_green = "";
static const char *ansi_color_red = "";
static const char *ansi_color_reset = "";
static const char *ansi_bold = "";
static const char *ansi_bold_cyan = "";
static const char *ansi_bold_green = "";
static const char *ansi_color_magenta = "";
static const char *ansi_color_cyan = "";
static const char *ansi_color_grey = "";
static const char *ansi_color_yellow = "";

#define ANSI_COLOR_GREEN   ansi_color_green
#define ANSI_COLOR_RED     ansi_color_red
#define ANSI_COLOR_RESET   ansi_color_reset
#define ANSI_BOLD          ansi_bold
#define ANSI_BOLD_CYAN     ansi_bold_cyan
#define ANSI_BOLD_GREEN    ansi_bold_green
#define ANSI_COLOR_MAGENTA ansi_color_magenta
#define ANSI_COLOR_CYAN    ansi_color_cyan
#define ANSI_COLOR_GREY    ansi_color_grey
#define ANSI_COLOR_YELLOW  ansi_color_yellow

static void init_colors(void)
{
    const char *no_color = getenv("NO_COLOR");

    if (no_color == NULL)
    {
        ansi_color_green = "\x1b[32m";
        ansi_color_red = "\x1b[31m";
        ansi_color_reset = "\x1b[0m";
        ansi_bold = "\x1b[1m";
        ansi_bold_cyan = "\x1b[1;36m";
        ansi_bold_green = "\x1b[1;32m";
        ansi_color_magenta = "\x1b[35m";
        ansi_color_cyan = "\x1b[36m";
        ansi_color_grey = "\x1b[90m";
        ansi_color_yellow = "\x1b[33m";
    }
} // init_colors

static void print_color_mode(void)
{
    const char *no_color = getenv("NO_COLOR");
    printf("\n%sCOLOR MODE%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    if (no_color == NULL)
    {
        printf("  %sENABLED%s (color escape codes are active; disable by setting NO_COLOR=1)\n",
               ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    }
    else
    {
        printf("  %sDISABLED%s (NO_COLOR environment variable is present)\n",
               ANSI_COLOR_RED, ANSI_COLOR_RESET);
    }
} // print_color_mode

static void print_help(
    const char *progname)
{
    printf("%sNAME%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  gric-info - Prints support and build status of optional modules\n\n");

    printf("%sUSAGE%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s%s%s %s[options]%s\n\n", ANSI_BOLD_GREEN, progname, ANSI_COLOR_RESET,
           ANSI_COLOR_GREY, ANSI_COLOR_RESET);

    printf("%sDESCRIPTION%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  Checks and prints status of optional modules: CFITSIO, LibPNG, FFmpeg,\n");
    printf("  ImageStreamIO, and OpenMP. Identifies supported formats and modes.\n\n");

    printf("%sOPTIONS%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s-h, --help%s           Show this help message\n\n", ANSI_COLOR_GREEN, ANSI_COLOR_RESET);

    printf("%sEXAMPLES%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s$%s %s%s%s\n", ANSI_COLOR_GREY, ANSI_COLOR_RESET, ANSI_BOLD_GREEN, progname,
           ANSI_COLOR_RESET);
    print_color_mode();
} // print_help

/**
 * @brief Prints details for a specific optional build module.
 *
 * @param name     Name of the module.
 * @param enabled  Whether the module is enabled in the current build.
 * @param version  Version string of the library.
 * @param location Installed library path.
 */
static void print_module_info(
    const char *name,
    int         enabled,
    const char *version,
    const char *location)
{
    printf("%-20s: ", name);
    if (enabled)
    {
        printf("%sENABLED%s", ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
        if (version && version[0] != '\0')
        {
            printf(" (Version: %s)", version);
        }
        if (location && location[0] != '\0')
        {
            printf("\n%22s Location: %s", "", location);
        }
    }
    else
    {
        printf("%sDISABLED%s", ANSI_COLOR_RED, ANSI_COLOR_RESET);
    }
    printf("\n");
} // print_module_info

int main(
    int   argc,
    char *argv[])
{
    init_colors();

    // Check for help option early
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_help(argv[0]);
            return 0;
        }
    }

    printf("%sGRIC-CLUSTER Optional Modules Information%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("=========================================\n\n");

#ifdef USE_CFITSIO
    print_module_info("CFITSIO", 1, CFITSIO_VERSION_STR, CFITSIO_LOCATION_STR);
#else
    print_module_info("CFITSIO", 0, NULL, NULL);
#endif

#ifdef USE_PNG
    print_module_info("LibPNG", 1, PNG_VERSION_STR, PNG_LOCATION_STR);
#else
    print_module_info("LibPNG", 0, NULL, NULL);
#endif

#ifdef USE_FFMPEG
    print_module_info("FFmpeg", 1, FFMPEG_VERSION_STR, FFMPEG_LOCATION_STR);
#else
    print_module_info("FFmpeg", 0, NULL, NULL);
#endif

#ifdef USE_IMAGESTREAMIO
    print_module_info("ImageStreamIO", 1, IMAGESTREAMIO_VERSION_STR, IMAGESTREAMIO_LOCATION_STR);
#else
    print_module_info("ImageStreamIO", 0, NULL, NULL);
#endif

#ifdef USE_OPENMP
    print_module_info("OpenMP", 1, OPENMP_VERSION_STR, NULL);
#else
    print_module_info("OpenMP", 0, NULL, NULL);
#endif

    return 0;
} // main

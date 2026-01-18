#include <stdio.h>

#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

void print_module_info(const char *name, int enabled, const char *version, const char *location) {
    printf("%-20s: ", name);
    if (enabled) {
        printf("%sENABLED%s", ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
        if (version && version[0] != '\0') {
            printf(" (Version: %s)", version);
        }
        if (location && location[0] != '\0') {
            printf("\n%22s Location: %s", "", location);
        }
    } else {
        printf("%sDISABLED%s", ANSI_COLOR_RED, ANSI_COLOR_RESET);
    }
    printf("\n");
}

int main() {
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
}

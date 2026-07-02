cmake_minimum_required(VERSION 3.10)
project(image_cluster C)

set(CMAKE_C_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -funroll-loops")

find_package(PkgConfig REQUIRED)

# Optional Dependencies
pkg_check_modules(CFITSIO cfitsio)
if (CFITSIO_FOUND)
    add_definitions(-DUSE_CFITSIO)
    message(STATUS "CFITSIO found. FITS format supported.")
else()
    message(WARNING "CFITSIO not found. FITS format disabled.")
endif()

find_package(PNG)
if (PNG_FOUND)
    add_definitions(-DUSE_PNG)
    message(STATUS "PNG found. PNG output supported.")
else()
    message(WARNING "PNG not found. PNG output disabled.")
endif()

pkg_check_modules(FFMPEG libavformat libavcodec libswscale libavutil)
if (FFMPEG_FOUND)
    add_definitions(-DUSE_FFMPEG)
    message(STATUS "FFmpeg found. Video input supported.")
else()
    message(WARNING "FFmpeg not found. Video input disabled.")
endif()

# Check for ImageStreamIO via pkg-config
pkg_check_modules(IMAGESTREAMIO ImageStreamIO)
if (IMAGESTREAMIO_FOUND)
    add_definitions(-DUSE_IMAGESTREAMIO)
    message(STATUS "ImageStreamIO found. Streaming input supported.")
else()
    message(WARNING "ImageStreamIO not found. Streaming input disabled.")
endif()

find_package(OpenMP)
if (OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    message(STATUS "OpenMP found.")
else()
    message(WARNING "OpenMP not found. Multithreading disabled.")
endif()

include_directories(
    ${CFITSIO_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
    ${IMAGESTREAMIO_INCLUDE_DIRS}
    src
    src/gric-cluster/core
    src/gric-cluster/math
    src/gric-cluster/io
    src/gric-cluster/help
    src/gric-cluster/steps
)

link_directories(
    ${CFITSIO_LIBRARY_DIRS}
    ${PNG_LIBRARY_DIRS}
    ${FFMPEG_LIBRARY_DIRS}
    ${IMAGESTREAMIO_LIBRARY_DIRS}
)

# Sources
set(CLUSTER_SRCS
    src/gric-cluster/core/main.c
    src/gric-cluster/core/cluster_core.c
    src/gric-cluster/core/cluster_step.c
    src/gric-cluster/core/cluster_mgmt.c
    src/gric-cluster/core/config_utils.c
    src/gric-cluster/core/cluster_bounds.c
    src/gric-cluster/core/cluster_shm.c
    src/gric-cluster/math/cluster_math.c
    src/gric-cluster/math/cluster_prune.c
    src/gric-cluster/math/cluster_scandist.c
    src/gric-cluster/math/framedistance.c
    src/gric-cluster/io/cluster_io.c
    src/gric-cluster/io/frameread.c
    src/gric-cluster/io/png_io.c
    src/gric-cluster/help/cluster_help.c
    src/gric-cluster/steps/initialize_initial_cluster.c
    src/gric-cluster/steps/compute_priors_and_mixing.c
    src/gric-cluster/steps/select_next_measurement_target.c
    src/gric-cluster/steps/measure_distance_to_cluster.c
    src/gric-cluster/steps/update_probabilities_and_pruning.c
    src/gric-cluster/steps/update_geometric_probabilities.c
    src/gric-cluster/steps/handle_new_cluster_creation.c
    src/gric-cluster/steps/record_step_assignment.c
    src/gric-cluster/steps/update_consistency_mask.c
)

add_executable(gric-cluster ${CLUSTER_SRCS})
if (OpenMP_C_FOUND)
    target_link_libraries(gric-cluster OpenMP::OpenMP_C)
endif()
target_link_libraries(gric-cluster
    ${CFITSIO_LIBRARIES}
    ${FFMPEG_LIBRARIES}
    ${IMAGESTREAMIO_LIBRARIES}
    m
)
if (PNG_FOUND)
    target_link_libraries(gric-cluster ${PNG_LIBRARIES})
endif()

add_executable(gric-mktxtseq src/gric-mktxtseq/mktestseq.c)
target_link_libraries(gric-mktxtseq m)

add_executable(gric-plot
    src/gric-plot/plot_clusters.c
    src/gric-plot/canvas/canvas.c
)
if (PNG_FOUND)
    target_link_libraries(gric-plot ${PNG_LIBRARIES} m)
else()
    target_link_libraries(gric-plot m)
endif()

add_executable(gric-mkclusteredfile src/gric-mkclusteredfile/mkclusteredfile.c)
target_link_libraries(gric-mkclusteredfile m)

add_executable(gric-NDmodel src/gric-NDmodel/model_nd.c)
target_link_libraries(gric-NDmodel m)

add_executable(gric-ascii-spot-2-video src/gric-ascii-spot-2-video/ascii-spot-2-video.c)
target_link_libraries(gric-ascii-spot-2-video m ${IMAGESTREAMIO_LIBRARIES})

if (IMAGESTREAMIO_FOUND)
    add_executable(gric-stream-to-pipe src/gric-stream-to-pipe/stream_to_pipe.c)
    target_link_libraries(gric-stream-to-pipe ${IMAGESTREAMIO_LIBRARIES} m)
endif()

# gric-status tool
add_executable(gric-status src/gric-status/gric-status.c)

# gric-info tool
add_executable(gric-info src/gric-info/gric-info.c)

# gric-help tool
add_executable(gric-help src/gric-help/gric-help.c)

# gric-benchmark tool
add_executable(gric-benchmark
    src/gric-benchmark/gric-benchmark.c
    src/gric-benchmark/benchmark_utils.c
    src/gric-benchmark/test_list.c
)

if (CFITSIO_FOUND)
    target_compile_definitions(gric-info PRIVATE USE_CFITSIO)
    target_compile_definitions(gric-info PRIVATE CFITSIO_VERSION_STR=\"${CFITSIO_VERSION}\")
    target_compile_definitions(gric-info PRIVATE CFITSIO_LOCATION_STR=\"${CFITSIO_LIBDIR}\")
endif()

if (PNG_FOUND)
    target_compile_definitions(gric-info PRIVATE USE_PNG)
    target_compile_definitions(gric-info PRIVATE PNG_VERSION_STR=\"${PNG_VERSION_STRING}\")
    # PNG_LIBRARIES is often a list, assume first item is main lib
    list(GET PNG_LIBRARIES 0 PNG_MAIN_LIB)
    target_compile_definitions(gric-info PRIVATE PNG_LOCATION_STR=\"${PNG_MAIN_LIB}\")
endif()

if (FFMPEG_FOUND)
    target_compile_definitions(gric-info PRIVATE USE_FFMPEG)
    
    set(MY_FFMPEG_VER "${FFMPEG_VERSION}")
    if ("${MY_FFMPEG_VER}" STREQUAL "" AND FFMPEG_libavformat_VERSION)
        set(MY_FFMPEG_VER "${FFMPEG_libavformat_VERSION}")
    endif()
    target_compile_definitions(gric-info PRIVATE FFMPEG_VERSION_STR=\"${MY_FFMPEG_VER}\")

    set(MY_FFMPEG_LOC "${FFMPEG_LIBDIR}")
    if ("${MY_FFMPEG_LOC}" STREQUAL "")
         set(MY_FFMPEG_LOC "System Default")
    endif()
    target_compile_definitions(gric-info PRIVATE FFMPEG_LOCATION_STR=\"${MY_FFMPEG_LOC}\")
endif()

if (IMAGESTREAMIO_FOUND)
    target_compile_definitions(gric-info PRIVATE USE_IMAGESTREAMIO)
    target_compile_definitions(gric-info PRIVATE IMAGESTREAMIO_VERSION_STR=\"${IMAGESTREAMIO_VERSION}\")
    target_compile_definitions(gric-info PRIVATE IMAGESTREAMIO_LOCATION_STR=\"${IMAGESTREAMIO_LIBDIR}\")
endif()

if (OpenMP_C_FOUND)
    target_compile_definitions(gric-info PRIVATE USE_OPENMP)
    target_compile_definitions(gric-info PRIVATE OPENMP_VERSION_STR=\"${OpenMP_C_VERSION}\")
endif()

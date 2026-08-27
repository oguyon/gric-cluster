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

find_package(Python3 COMPONENTS Interpreter REQUIRED)

file(GLOB HELP_TOPIC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/docs/help/*.md")
set(HELP_TOPICS_GEN_C "${CMAKE_CURRENT_BINARY_DIR}/help_topics_gen.c")
add_custom_command(
    OUTPUT "${HELP_TOPICS_GEN_C}"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_help_c.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/docs/help"
            "${HELP_TOPICS_GEN_C}"
    DEPENDS ${HELP_TOPIC_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_help_c.py"
    COMMENT "Generating embedded Markdown help topics database"
)

include_directories(
    ${CFITSIO_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
    ${IMAGESTREAMIO_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    src
    src/shared
    src/gric-cluster/core
    src/gric-cluster/math
    src/gric-cluster/io
    src/gric-cluster/help
    src/gric-cluster/steps
    src/gric-cluster/trace
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
    src/gric-cluster/core/cluster_reassign.c
    src/gric-cluster/core/tile_state.c
    src/gric-cluster/core/cluster_core_multitile.c
    src/gric-cluster/core/tile_map.c
    src/gric-cluster/io/frame_scatter.c
    src/gric-cluster/io/cluster_io_multitile.c
    src/gric-cluster/math/cluster_math.c
    src/gric-cluster/math/cluster_prune.c
    src/gric-cluster/math/cluster_scandist.c
    src/gric-cluster/math/framedistance.c
    src/gric-cluster/math/tuple_retrieval.c
    src/gric-cluster/io/cluster_io.c
    src/gric-cluster/io/cluster_io_log.c
    src/gric-cluster/io/cluster_io_results.c
    src/gric-cluster/io/frameread.c
    src/gric-cluster/io/frameread_ascii.c
    src/gric-cluster/io/frameread_fits.c
    src/gric-cluster/io/frameread_ffmpeg.c
    src/gric-cluster/io/frameread_stream.c
    src/gric-cluster/io/png_io.c
    src/gric-cluster/help/cluster_help.c
    src/gric-cluster/help/cluster_help_format.c
    src/gric-cluster/help/cluster_help_content.c
    src/gric-cluster/help/help_md_render.c
    ${HELP_TOPICS_GEN_C}
    src/gric-cluster/steps/initialize_initial_cluster.c
    src/gric-cluster/steps/compute_priors_and_mixing.c
    src/gric-cluster/steps/select_next_measurement_target.c
    src/gric-cluster/steps/measure_distance_to_cluster.c
    src/gric-cluster/steps/update_probabilities_and_pruning.c
    src/gric-cluster/steps/update_geometric_probabilities.c
    src/gric-cluster/steps/handle_new_cluster_creation.c
    src/gric-cluster/steps/record_step_assignment.c
    src/gric-cluster/steps/update_consistency_mask.c
    src/gric-cluster/trace/cluster_trace.c
    src/shared/cli_colors.c
    src/shared/gric_bin_io.c
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

add_executable(gric-mktxtseq
    src/gric-mktxtseq/mktestseq.c
    src/shared/gric_gen_patterns.c
    src/shared/cli_colors.c
)
target_link_libraries(gric-mktxtseq m)

add_executable(gric-plot
    src/gric-plot/plot_clusters.c
    src/gric-plot/plot_log_parser.c
    src/gric-plot/plot_render.c
    src/gric-plot/canvas/canvas.c
    src/shared/cli_colors.c
)
if (PNG_FOUND)
    target_link_libraries(gric-plot ${PNG_LIBRARIES} m)
else()
    target_link_libraries(gric-plot m)
endif()

add_executable(gric-mkclusteredfile src/gric-mkclusteredfile/mkclusteredfile.c src/shared/cli_colors.c)
target_link_libraries(gric-mkclusteredfile m)

add_executable(gric-NDmodel src/gric-NDmodel/model_nd.c src/shared/cli_colors.c)
target_link_libraries(gric-NDmodel m)

if (CFITSIO_FOUND)
    add_executable(gric-gen-balls tools/gen_bouncing_balls.c)
    target_link_libraries(gric-gen-balls m ${CFITSIO_LIBRARIES})
    target_include_directories(gric-gen-balls PRIVATE ${CFITSIO_INCLUDE_DIRS})
endif()

add_executable(gric-ascii-spot-2-video src/gric-ascii-spot-2-video/ascii-spot-2-video.c src/shared/cli_colors.c)
target_link_libraries(gric-ascii-spot-2-video m ${IMAGESTREAMIO_LIBRARIES})

if (IMAGESTREAMIO_FOUND)
    add_executable(gric-txt2stream src/gric-txt2stream/txt2stream.c src/shared/cli_colors.c)
    target_link_libraries(gric-txt2stream ${IMAGESTREAMIO_LIBRARIES} m)

    add_executable(gric-stream-to-pipe src/gric-stream-to-pipe/stream_to_pipe.c src/shared/cli_colors.c)
    target_link_libraries(gric-stream-to-pipe ${IMAGESTREAMIO_LIBRARIES} m)
endif()

# gric-status tool
add_executable(gric-status
    src/gric-status/gric-status.c
    src/gric-status/status_classic.c
    src/gric-status/status_watch.c
    src/gric-status/status_terminal.c
    src/gric-status/status_shm.c
    src/gric-status/status_tui.c
    src/shared/cli_colors.c
)


# gric-info tool
add_executable(gric-info src/gric-info/gric-info.c src/shared/cli_colors.c)

# gric-help tool
add_executable(gric-help src/gric-help/gric-help.c src/shared/cli_colors.c)

# gric-benchmark tool
add_executable(gric-benchmark
    src/gric-benchmark/gric-benchmark.c
    src/gric-benchmark/benchmark_cli.c
    src/gric-benchmark/benchmark_runner.c
    src/gric-benchmark/benchmark_report.c
    src/gric-benchmark/benchmark_utils.c
    src/gric-benchmark/test_list.c
    src/shared/cli_colors.c
)

# gric-tune tool
add_executable(gric-tune
    src/gric-tune/gric-tune.c
    src/shared/cli_colors.c
)
target_link_libraries(gric-tune m)

# gric-cluster-analysis tool
add_executable(gric-cluster-analysis
    src/gric-cluster-analysis/gric-cluster-analysis.c
    src/gric-cluster-analysis/analysis_parser.c
    src/gric-cluster-analysis/analysis_stats.c
    src/gric-cluster-analysis/analysis_report.c
    src/shared/cli_colors.c
)
target_link_libraries(gric-cluster-analysis m)

# gric-server tool (Native C desktop micro-server for GUI)
add_executable(gric-server
    src/gric-server/server_main.c
    src/gric-server/http_server.c
    src/gric-server/api_handlers.c
    src/gric-server/server_http_utils.c
    src/gric-server/server_job_mgmt.c
    src/gric-server/server_fs_api.c
    src/gric-server/server_cli_runner.c
    src/shared/cli_colors.c
)
target_include_directories(gric-server PRIVATE src/gric-server src src/gric-cluster/core)

# gric-knn tool
add_executable(gric-knn
    src/gric-knn/main.c
    src/gric-knn/knn_heap.c
    src/gric-knn/knn_loader.c
    src/gric-knn/knn_tree.c
    src/gric-knn/knn_reader.c
    src/gric-knn/knn_engine.c
    src/gric-knn/knn_writer.c
    src/shared/cli_colors.c
    src/shared/gric_bin_io.c
)
target_include_directories(gric-knn PRIVATE src/gric-knn src)
target_link_libraries(gric-knn m)
if (OpenMP_C_FOUND)
    target_link_libraries(gric-knn OpenMP::OpenMP_C)
endif()
if (CFITSIO_FOUND)
    target_link_libraries(gric-knn ${CFITSIO_LIBRARIES})
    target_include_directories(gric-knn PRIVATE ${CFITSIO_INCLUDE_DIRS})
endif()

# gric-ascii2bin tool
add_executable(gric-ascii2bin
    src/gric-ascii2bin/ascii2bin.c
    src/shared/gric_bin_io.c
    src/shared/cli_colors.c
)
target_link_libraries(gric-ascii2bin m)

# gric-bin2ascii tool
add_executable(gric-bin2ascii
    src/gric-bin2ascii/bin2ascii.c
    src/shared/gric_bin_io.c
    src/shared/cli_colors.c
)
target_link_libraries(gric-bin2ascii m)

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

# ==============================================================================
# Testing & CTest Integration
# ==============================================================================
enable_testing()

add_test(NAME test_coordinate_clustering
    COMMAND gric-cluster 0.5 "${CMAKE_SOURCE_DIR}/tests/test_strat.txt" -maxim 1000 -outdir /tmp/ctest_strat_out)

add_test(NAME test_sequence_generator
    COMMAND gric-mktxtseq 1000 /tmp/ctest_spiral.txt 2Dspiral)

add_test(NAME test_spiral_clustering
    COMMAND gric-cluster 0.1 /tmp/ctest_spiral.txt -maxim 1000 -outdir /tmp/ctest_spiral_out)

add_test(NAME test_knn_spiral
    COMMAND gric-knn /tmp/ctest_spiral.txt /tmp/ctest_spiral_out -k 10 -dtmin 5 -o /tmp/ctest_knn_spiral.txt)
set_tests_properties(test_knn_spiral PROPERTIES DEPENDS test_spiral_clustering)

if (CFITSIO_FOUND)
    add_test(NAME test_bouncing_balls_single_gen
        COMMAND gric-gen-balls -n 1 -r 5.0 -W 32 -H 32 -f 500 -s 42 /tmp/ctest_balls_1.fits)
    add_test(NAME test_bouncing_balls_single_cluster
        COMMAND gric-cluster 3.0 /tmp/ctest_balls_1.fits -outdir /tmp/ctest_balls_1_out)
    set_tests_properties(test_bouncing_balls_single_cluster PROPERTIES DEPENDS test_bouncing_balls_single_gen)

    add_test(NAME test_knn_balls
        COMMAND gric-knn /tmp/ctest_balls_1.fits /tmp/ctest_balls_1_out -k 5 -dtmin 2 -o /tmp/ctest_knn_balls.fits)
    set_tests_properties(test_knn_balls PROPERTIES DEPENDS test_bouncing_balls_single_cluster)

    add_test(NAME test_bouncing_balls_coll_gen
        COMMAND gric-gen-balls -n 3 -r 5.0 -W 32 -H 32 -f 500 -s 42 /tmp/ctest_balls_3.fits)
    add_test(NAME test_bouncing_balls_coll_cluster
        COMMAND gric-cluster 3.0 /tmp/ctest_balls_3.fits -outdir /tmp/ctest_balls_3_out)
    set_tests_properties(test_bouncing_balls_coll_cluster PROPERTIES DEPENDS test_bouncing_balls_coll_gen)
endif()

add_test(NAME test_benchmark_smoke
    COMMAND gric-benchmark -p 2Dspiral -n 500)

add_executable(test_pass2_reassign tests/test_pass2_reassign.c)
target_link_libraries(test_pass2_reassign m)
add_test(NAME test_pass2_reassign
    COMMAND test_pass2_reassign)
set_tests_properties(test_pass2_reassign PROPERTIES DEPENDS gric-cluster)

if (IMAGESTREAMIO_FOUND)
    add_executable(test_txt2stream tests/test_txt2stream.c)
    target_link_libraries(test_txt2stream m)
    add_test(NAME test_txt2stream
        COMMAND test_txt2stream)
    set_tests_properties(test_txt2stream PROPERTIES DEPENDS "gric-txt2stream;gric-cluster")
endif()

add_executable(test_bin_ascii_roundtrip
    tests/test_bin_ascii_roundtrip.c
    src/shared/gric_bin_io.c
)
target_link_libraries(test_bin_ascii_roundtrip m)
add_test(NAME test_bin_ascii_roundtrip
    COMMAND test_bin_ascii_roundtrip)
set_tests_properties(test_bin_ascii_roundtrip PROPERTIES DEPENDS "gric-ascii2bin;gric-bin2ascii")

add_executable(test_bin_clustering_pipeline
    tests/test_bin_clustering_pipeline.c
    src/shared/gric_bin_io.c
)
target_link_libraries(test_bin_clustering_pipeline m)
add_test(NAME test_bin_clustering_pipeline
    COMMAND test_bin_clustering_pipeline)
set_tests_properties(test_bin_clustering_pipeline PROPERTIES DEPENDS "gric-cluster;gric-ascii2bin;gric-bin2ascii;gric-mktxtseq;gric-knn")

# Target to regenerate benchmark figures and documentation pages
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_target(benchmark-docs
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tools/gen_benchmark_docs.py"
    DEPENDS gric-cluster gric-mktxtseq gric-plot gric-gen-balls
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Generating benchmark figures with gric-plot and updating MkDocs benchmark pages"
)

# Target to build and deploy WebAssembly module to site/ and docs/
add_custom_target(wasm
    COMMAND make -f "${CMAKE_SOURCE_DIR}/Makefile.wasm"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Compiling WebAssembly module and deploying to site/ and docs/"
)


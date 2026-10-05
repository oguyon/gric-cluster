# ==============================================================================
# src/gric-fps/CMakeLists.txt - GRIC Milk Framework Adapter
# ==============================================================================

# Header compatibility mapping: milk_types.h includes "libprocessinfo/processtools.h",
# but Milk's CMake installs processtools.h directly under ${MILK_INCLUDE_DIRS}.
set(MILK_COMPAT_DIR "${CMAKE_CURRENT_BINARY_DIR}/include_compat")
file(MAKE_DIRECTORY "${MILK_COMPAT_DIR}")
if (NOT EXISTS "${MILK_COMPAT_DIR}/libprocessinfo")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${MILK_INCLUDE_DIRS}"
            "${MILK_COMPAT_DIR}/libprocessinfo"
    )
endif()

if (CMAKE_C_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wno-stringop-truncation)
endif()

set(MILK_FPS_SYSTEM_INCLUDES
    ${MILK_COMPAT_DIR}
    ${MILK_INCLUDE_DIRS}
    ${MILK_INCLUDE_DIRS}/libfps
    ${MILK_INCLUDE_DIRS}/libmilkcommon
    ${MILK_INCLUDE_DIRS}/libmilkdata
    ${MILK_INCLUDE_DIRS}/CommandLineInterface
    ${IMAGESTREAMIO_INCLUDE_DIRS}
)

# 1. Standalone FPS Daemon Executable: milk-fpsexec-gric-cluster
add_executable(milk-fpsexec-gric-cluster
    gric_fps_main.c
    gric_fps_common.c
)

target_compile_definitions(milk-fpsexec-gric-cluster PRIVATE
    FPS_STANDALONE
    MILK_NO_CLI
)

target_include_directories(milk-fpsexec-gric-cluster PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/gric-cluster/core
)

target_include_directories(milk-fpsexec-gric-cluster SYSTEM PRIVATE
    ${MILK_FPS_SYSTEM_INCLUDES}
)

target_link_libraries(milk-fpsexec-gric-cluster PRIVATE
    gric_static
    milkfps
    milkfpsStandalone
    milkfpsseq
    milkprocessinfo
    ${IMAGESTREAMIO_LIBRARIES}
    milkCOREMODmemory
    milkCOREMODtools
    milkCOREMODarith
    milkcommon
    milkdata
    m
    pthread
    rt
    -Wl,--allow-shlib-undefined
)

# Create convenient symlink: gric-fps-cluster -> milk-fpsexec-gric-cluster
add_custom_command(TARGET milk-fpsexec-gric-cluster POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        $<TARGET_FILE_NAME:milk-fpsexec-gric-cluster>
        ${CMAKE_CURRENT_BINARY_DIR}/gric-fps-cluster
)

install(TARGETS milk-fpsexec-gric-cluster
    DESTINATION bin
)

install(CODE "
    file(CREATE_LINK milk-fpsexec-gric-cluster
        \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/gric-fps-cluster
        SYMBOLIC)
")

# 2. Standalone FPS Daemon Executable: milk-fpsexec-gric-knn
add_executable(milk-fpsexec-gric-knn
    gric_knn_fps_main.c
    gric_knn_fps_common.c
)

target_compile_definitions(milk-fpsexec-gric-knn PRIVATE
    FPS_STANDALONE
    MILK_NO_CLI
)

target_include_directories(milk-fpsexec-gric-knn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/gric-cluster/core
    ${CMAKE_SOURCE_DIR}/src/gric-knn
)

target_include_directories(milk-fpsexec-gric-knn SYSTEM PRIVATE
    ${MILK_FPS_SYSTEM_INCLUDES}
)

target_link_libraries(milk-fpsexec-gric-knn PRIVATE
    gric_knn_static
    gric_static
    milkfps
    milkfpsStandalone
    milkfpsseq
    milkprocessinfo
    ${IMAGESTREAMIO_LIBRARIES}
    milkCOREMODmemory
    milkCOREMODtools
    milkCOREMODarith
    milkcommon
    milkdata
    m
    pthread
    rt
    -Wl,--allow-shlib-undefined
)

# Create convenient symlink: gric-fps-knn -> milk-fpsexec-gric-knn
add_custom_command(TARGET milk-fpsexec-gric-knn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        $<TARGET_FILE_NAME:milk-fpsexec-gric-knn>
        ${CMAKE_CURRENT_BINARY_DIR}/gric-fps-knn
)

install(TARGETS milk-fpsexec-gric-knn
    DESTINATION bin
)

install(CODE "
    file(CREATE_LINK milk-fpsexec-gric-knn
        \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/gric-fps-knn
        SYMBOLIC)
")

# 3. Standalone FPS Daemon Executable: milk-fpsexec-gric-reconstruct
add_executable(milk-fpsexec-gric-reconstruct
    gric_recon_fps_main.c
    gric_recon_fps_common.c
)

target_compile_definitions(milk-fpsexec-gric-reconstruct PRIVATE
    FPS_STANDALONE
    MILK_NO_CLI
)

target_include_directories(milk-fpsexec-gric-reconstruct PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/gric-cluster/core
    ${CMAKE_SOURCE_DIR}/src/gric-knn
)

target_include_directories(milk-fpsexec-gric-reconstruct SYSTEM PRIVATE
    ${MILK_FPS_SYSTEM_INCLUDES}
)

target_link_libraries(milk-fpsexec-gric-reconstruct PRIVATE
    gric_knn_static
    gric_static
    milkfps
    milkfpsStandalone
    milkfpsseq
    milkprocessinfo
    ${IMAGESTREAMIO_LIBRARIES}
    milkCOREMODmemory
    milkCOREMODtools
    milkCOREMODarith
    milkcommon
    milkdata
    m
    pthread
    rt
    -Wl,--allow-shlib-undefined
)

# Create convenient symlink: gric-fps-reconstruct -> milk-fpsexec-gric-reconstruct
add_custom_command(TARGET milk-fpsexec-gric-reconstruct POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        $<TARGET_FILE_NAME:milk-fpsexec-gric-reconstruct>
        ${CMAKE_CURRENT_BINARY_DIR}/gric-fps-reconstruct
)

install(TARGETS milk-fpsexec-gric-reconstruct
    DESTINATION bin
)

install(CODE "
    file(CREATE_LINK milk-fpsexec-gric-reconstruct
        \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/gric-fps-reconstruct
        SYMBOLIC)
")

# 4. Shared Milk CLI Plugin: libmilkgric.so
add_library(milkgric SHARED
    gric_module.c
    gric_fps_common.c
    gric_knn_module.c
    gric_knn_fps_common.c
    gric_recon_module.c
    gric_recon_fps_common.c
)

set_target_properties(milkgric PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "milkgric"
)

target_include_directories(milkgric PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/gric-cluster/core
    ${CMAKE_SOURCE_DIR}/src/gric-knn
)

target_include_directories(milkgric SYSTEM PRIVATE
    ${MILK_FPS_SYSTEM_INCLUDES}
    ${MILK_INCLUDE_DIRS}/CommandLineInterface/CLIcore
)

target_link_libraries(milkgric PRIVATE
    gric_knn_static
    gric_static
    CLIcore
    milkfps
    milkfpsCLI
    milkprocessinfo
    ${IMAGESTREAMIO_LIBRARIES}
    milkCOREMODmemory
    milkcommon
    milkdata
    m
    pthread
    rt
)

install(TARGETS milkgric
    DESTINATION lib
)

get_property(_fps_progs GLOBAL PROPERTY GRIC_INSTALLED_PROGRAMS)
list(APPEND _fps_progs
    milk-fpsexec-gric-cluster
    gric-fps-cluster
    milk-fpsexec-gric-knn
    gric-fps-knn
    milk-fpsexec-gric-reconstruct
    gric-fps-reconstruct
)
set_property(GLOBAL PROPERTY GRIC_INSTALLED_PROGRAMS "${_fps_progs}")

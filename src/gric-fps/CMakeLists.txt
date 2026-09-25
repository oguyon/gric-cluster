# ==============================================================================
# src/gric-fps/CMakeLists.txt - GRIC Milk Framework Adapter
# ==============================================================================

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
    ${MILK_INCLUDE_DIRS}
    ${MILK_INCLUDE_DIRS}/libfps
    ${MILK_INCLUDE_DIRS}/libmilkdata
    ${MILK_INCLUDE_DIRS}/libmilkcommon
    ${MILK_INCLUDE_DIRS}/CommandLineInterface
    ${IMAGESTREAMIO_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(milk-fpsexec-gric-cluster PRIVATE
    gric
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

# 2. Shared Milk CLI Plugin: libmilkgric.so
add_library(milkgric SHARED
    gric_module.c
    gric_fps_common.c
)

set_target_properties(milkgric PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "milkgric"
)


target_include_directories(milkgric PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MILK_INCLUDE_DIRS}
    ${MILK_INCLUDE_DIRS}/libfps
    ${MILK_INCLUDE_DIRS}/libmilkdata
    ${MILK_INCLUDE_DIRS}/libmilkcommon
    ${MILK_INCLUDE_DIRS}/CommandLineInterface
    ${MILK_INCLUDE_DIRS}/CommandLineInterface/CLIcore
    ${IMAGESTREAMIO_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(milkgric PRIVATE
    gric
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

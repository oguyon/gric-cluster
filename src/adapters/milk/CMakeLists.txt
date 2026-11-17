# ==============================================================================
# src/adapters/milk/CMakeLists.txt - GRIC Milk (feat/bash-engine) Adapter
# ==============================================================================

if (NOT EXISTS "${MILK_INCLUDE_DIRS}/libcomputeunit/milk_runnable.h")
    message(FATAL_ERROR
        "Milk feat/bash-engine headers missing "
        "(${MILK_INCLUDE_DIRS}/libcomputeunit/milk_runnable.h).")
endif()

if (CMAKE_C_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wno-stringop-truncation)
endif()

set(MILK_FPS_SYSTEM_INCLUDES
    ${MILK_INCLUDE_DIRS}
    ${MILK_INCLUDE_DIRS}/CommandLineInterface
    ${MILK_INCLUDE_DIRS}/CommandLineInterface/CLIcore
    ${MILK_INCLUDE_DIRS}/libmilkcommon
    ${MILK_INCLUDE_DIRS}/libmilkdata
    ${MILK_INCLUDE_DIRS}/libfps
    ${MILK_INCLUDE_DIRS}/libprocessinfo
    ${MILK_INCLUDE_DIRS}/libcomputeunit
    ${IMAGESTREAMIO_INCLUDE_DIRS}
)

# 1. Shared Milk CLI + Runnable Module: libmilkgric.so
add_library(milkgric SHARED
    milkgric.c
    gric_cluster_FPS.c
    gric_fps_common.c
    gric_knn_FPS.c
    gric_knn_fps_common.c
    gric_recon_FPS.c
    gric_recon_fps_common.c
)

set_target_properties(milkgric PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "milkgric"
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_BINARY_DIR};${CMAKE_INSTALL_RPATH}"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${CMAKE_INSTALL_RPATH}"
)

target_compile_definitions(milkgric PRIVATE MILK_NO_CLI)

target_include_directories(milkgric PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/gric-cluster/core
    ${CMAKE_SOURCE_DIR}/src/gric-knn
)

target_include_directories(milkgric SYSTEM PRIVATE
    ${MILK_FPS_SYSTEM_INCLUDES}
)

target_link_libraries(milkgric PRIVATE
    gric_knn_static
    gric_static
    milkcomputeunit
    milkfps
    milkprocessinfo
    milkdata
    milkcommon
    milkCOREMODmemory
    milkCOREMODarith
    milkCOREMODtools
    ${IMAGESTREAMIO_LIBRARIES}
    m
    pthread
    rt
)

install(TARGETS milkgric DESTINATION lib)

# 2. Standalone FPS Daemons generated from EXPORT_FPSEXEC stubs
set(_GRIC_FPSEXEC_STUB_TPL [=[
#include "milk_skinny_runnable.h"

extern const MILK_RUNNABLE @UNIT@;

int main(
    int   argc,
    char *argv[])
{
    MILK_EMBED_BUILD_TAG();
    return MILK_FPSEXEC_MAIN(argc, argv, &@UNIT@);
}
]=])

set(_GRIC_FPSEXEC_SPECS
    "gric-cluster:runnable_gric_cluster:gric-fps-cluster"
    "gric-knn:runnable_gric_knn:gric-fps-knn"
    "gric-reconstruct:runnable_gric_reconstruct:gric-fps-reconstruct"
)

foreach(_spec IN LISTS _GRIC_FPSEXEC_SPECS)
    string(REPLACE ":" ";" _parts "${_spec}")
    list(GET _parts 0 _id)
    list(GET _parts 1 _unit)
    list(GET _parts 2 _alias)
    set(_exe "milk-fpsexec-${_id}")

    string(REPLACE "@UNIT@" "${_unit}" _stub_src "${_GRIC_FPSEXEC_STUB_TPL}")
    set(_stub_file "${CMAKE_CURRENT_BINARY_DIR}/fpsexec_stubs/${_exe}.c")
    file(GENERATE OUTPUT "${_stub_file}" CONTENT "${_stub_src}")

    add_executable(${_exe} "${_stub_file}")
    target_compile_definitions(${_exe} PRIVATE
        MILK_NO_CLI
        MILK_BUILD_BINNAME="${_exe}"
    )
    target_include_directories(${_exe} SYSTEM PRIVATE
        ${MILK_FPS_SYSTEM_INCLUDES}
    )
    target_link_libraries(${_exe} PRIVATE
        milkgric
        milkcomputeunit
        milkfps
        milkprocessinfo
        milkdata
        milkcommon
        milkCOREMODmemory
        ${IMAGESTREAMIO_LIBRARIES}
        m
        rt
        pthread
        -Wl,--allow-shlib-undefined
    )
    set_target_properties(${_exe} PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_BINARY_DIR};${CMAKE_INSTALL_RPATH}"
        INSTALL_RPATH "$ORIGIN/../lib;${CMAKE_INSTALL_RPATH}"
    )

    add_custom_command(TARGET ${_exe} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE_NAME:${_exe}>
            "${CMAKE_CURRENT_BINARY_DIR}/${_alias}"
    )

    install(TARGETS ${_exe} DESTINATION bin)
    install(CODE "
        file(CREATE_LINK ${_exe}
            \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/${_alias}
            SYMBOLIC)
    ")
endforeach()

# 3. Refresh Milk command catalog at install time when milk-cli is available
install(CODE "
    set(_milk_prefix \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\")
    set(_milk_cli \"\${_milk_prefix}/bin/milk-cli\")
    if(EXISTS \"\${_milk_cli}\")
        message(STATUS \"Refreshing milk module catalog via \${_milk_cli} --catalog\")
        execute_process(
            COMMAND \${CMAKE_COMMAND} -E env
                MILK_INSTALLDIR=\${_milk_prefix}
                LD_LIBRARY_PATH=\${_milk_prefix}/lib:\$ENV{LD_LIBRARY_PATH}
                \"\${_milk_cli}\" --catalog
            RESULT_VARIABLE _cat_rc)
        if(NOT _cat_rc EQUAL 0)
            message(WARNING \"milk-cli --catalog returned \${_cat_rc}\")
        endif()
    endif()
")

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

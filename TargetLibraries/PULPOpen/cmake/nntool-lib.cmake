
set(NNTOOL_LIB_HOME third_party/nntool-lib-CNN-SQ8)

# JUNGVI: PULP LLVM15 crashes when compiling this library, enable it for GCC only
# JUNGVI: For LLVM add a dummy library
if(TOOLCHAIN STREQUAL GCC)

    set(NNTOOL_LIB_C_SOURCE
        ${NNTOOL_LIB_HOME}/src/CNN_Conv_SQ8.c
        ${NNTOOL_LIB_HOME}/src/CNN_Conv_DW_SQ8.c
    )

    set(NNTOOL_LIB_INCLUDE
        ${NNTOOL_LIB_HOME}/inc
    )

    add_compile_options(
        -D__builtin_shuffle=__builtin_pulp_shuffle4b
    )

    if(NOT DEFINED $ENV{VERBOSE})
        add_compile_options(
            -Wno-sign-conversion
            -Wno-conversion
            -Wno-unused-variable
            -Wno-unused-parameter
        )
    endif()

    add_library(nntool-lib OBJECT ${NNTOOL_LIB_C_SOURCE}) 
    target_include_directories(nntool-lib PUBLIC ${NNTOOL_LIB_INCLUDE})
else()
    set(NNTOOL_LIB_INCLUDE
        ${NNTOOL_LIB_HOME}/temp
    )

    add_library(nntool-lib OBJECT ${NNTOOL_LIB_INCLUDE}/CNN_BasicKernels_SQ8.h) 
    target_include_directories(nntool-lib PUBLIC ${NNTOOL_LIB_INCLUDE})
endif()


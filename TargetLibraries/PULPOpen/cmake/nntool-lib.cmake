set(NNTOOL_LIB_HOME third_party/nntool-lib-CNN-SQ8)

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

add_library(nntool-lib OBJECT ${NNTOOL_LIB_C_SOURCE}) 
target_include_directories(nntool-lib PUBLIC ${NNTOOL_LIB_INCLUDE})
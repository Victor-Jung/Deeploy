set(ProjectId ${TESTNAME})

file(GLOB_RECURSE SOURCES
    main.c
)

link_directories(${ProjectId}/../../${GENERATED_SOURCE})

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES} )
target_link_libraries(${ProjectId} PRIVATE network deeploylib)
# RUN WANG: Link math Lib to Generic Target
target_link_libraries(${ProjectId} PRIVATE m)

link_compile_dump(${TESTNAME})

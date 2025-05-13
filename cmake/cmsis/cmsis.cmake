add_compile_definitions(
    DEEPLOY_CMSIS_PLATFORM
)

set(DEEPLOY_ARCH CMSIS)

add_compile_options(
  -ffast-math
)

add_link_options(
  -ffast-math
  -Wl,--gc-sections
)

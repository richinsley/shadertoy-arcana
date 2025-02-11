project(shadertoyarcana_c)

# Find pkg-config
find_package(PkgConfig REQUIRED)

# Set PKG_CONFIG_PATH to include ARCANA_PREFIX
set(ENV{PKG_CONFIG_PATH} "${ARCANA_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")

# Find all the required libraries using pkg-config
pkg_check_modules(ARCANA REQUIRED
    libswscale_arcana
    libpostproc_arcana
    libavformat_arcana
    libavcodec_arcana
    libswresample_arcana
    libavutil_arcana
)

# Create the shared library
add_library(shadertoyarcana SHARED arcana_loader.c vf_shadertoy_arcana.c)

# Add dependency on the Go library
add_dependencies(shadertoyarcana shadertoyarcana_go)

# Set include directories and link libraries
target_include_directories(shadertoyarcana PRIVATE 
    ${ARCANA_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/go  # For the generated Go header
)

target_link_directories(shadertoyarcana PRIVATE ${ARCANA_LIBRARY_DIRS})
target_link_libraries(shadertoyarcana PRIVATE 
    ${CMAKE_BINARY_DIR}/go/libshadertoyarcana_go.a
    ${ARCANA_LINK_LIBRARIES}
)

# Add runtime library search path
set_target_properties(shadertoyarcana PROPERTIES
    INSTALL_RPATH "${ARCANA_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

target_compile_options(shadertoyarcana PRIVATE ${ARCANA_CFLAGS})

# Installation rules
install(TARGETS shadertoyarcana
    LIBRARY DESTINATION lib
)
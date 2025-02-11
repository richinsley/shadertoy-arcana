project(shadertoyarcana_c)

# Find pkg-config
find_package(PkgConfig REQUIRED)

# Set PKG_CONFIG_PATH to include ARCANA_PREFIX
set(ENV{PKG_CONFIG_PATH} "${ARCANA_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")

# Find all the required libraries using pkg-config
# Use IMPORTED_TARGET to create proper targets
pkg_check_modules(ARCANA IMPORTED_TARGET
    libavfilter_arcana
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

# Set include directories
target_include_directories(shadertoyarcana PRIVATE 
    ${ARCANA_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/go  # For the generated Go header
)

# Base libraries
set(LINK_LIBS
    ${CMAKE_BINARY_DIR}/go/libshadertoyarcana_go.a
)

# Add macOS frameworks needed by Go
if(APPLE)
    list(APPEND LINK_LIBS
        "-framework CoreFoundation"
        "-framework Security"
    )
endif()

# Link against the imported pkg-config targets
target_link_libraries(shadertoyarcana PRIVATE 
    ${LINK_LIBS}
    PkgConfig::ARCANA
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
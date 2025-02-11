cmake_minimum_required(VERSION 3.10)
project(shadertoyarcana)

# Ensure ARCANA_PREFIX is defined
if(NOT DEFINED ARCANA_PREFIX)
    message(FATAL_ERROR "ARCANA_PREFIX must be defined")
endif()

# Build the Go static library first
add_subdirectory(go)
# Then build the C shared library that depends on it
add_subdirectory(c)
project(shadertoyarcana_go)

# Find Go compiler
find_program(GO_COMPILER go REQUIRED)

# Get all Go source files
file(GLOB GO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.go)

# Custom command to build the Go code as an archive
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libshadertoyarcana_go.a
           ${CMAKE_CURRENT_BINARY_DIR}/libshadertoyarcana_go.h
    COMMAND ${GO_COMPILER} build -buildmode=c-archive
        -o ${CMAKE_CURRENT_BINARY_DIR}/libshadertoyarcana_go.a
        ${CMAKE_CURRENT_SOURCE_DIR}/main.go
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${GO_SOURCES}
    COMMENT "Building Go static library"
)

add_custom_target(shadertoyarcana_go ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libshadertoyarcana_go.a
            ${CMAKE_CURRENT_BINARY_DIR}/libshadertoyarcana_go.h
)

# Installation rules
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libshadertoyarcana_go.a
    DESTINATION lib
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libshadertoyarcana_go.h
    DESTINATION include
)
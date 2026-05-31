file(GLOB_RECURSE SRCS *.cpp)

add_library(aby3-GORAM-Core ${SRCS})
include_directories("/usr/include/jsoncpp")  
include_directories("/usr/include/mpich")

#######################################
# json
find_package(jsoncpp CONFIG QUIET)
if(TARGET JsonCpp::JsonCpp)
    set(ABY3_GORAM_CORE_JSONCPP_TARGET JsonCpp::JsonCpp)
else()
    find_path(JSON_INCLUDE_DIR
      NAMES "json/json.h"
      REQUIRED)
    find_library(JSONCPP_LIBRARY
      NAMES jsoncpp
      REQUIRED)
    message(STATUS "In aby3-GORAM-Core, JSON_INCLUDE_DIR=${JSON_INCLUDE_DIR}")
    if(NOT TARGET jsoncpp_fallback)
        add_library(jsoncpp_fallback INTERFACE)
        target_include_directories(jsoncpp_fallback INTERFACE
                        $<BUILD_INTERFACE:${JSON_INCLUDE_DIR}>
                        $<INSTALL_INTERFACE:>)
        target_link_libraries(jsoncpp_fallback INTERFACE ${JSONCPP_LIBRARY})
    endif()
    set(ABY3_GORAM_CORE_JSONCPP_TARGET jsoncpp_fallback)
endif()

#######################################
# mpi
if(NOT TARGET mpi)
    find_path(MPI_INCLUDE_DIR 
      NAMES "mpi.h" 
      REQUIRED)
    if(NOT EXISTS ${MPI_INCLUDE_DIR})
        message(FATAL_ERROR "failed to find mpi.h")
    endif()
    message(STATUS "In aby3-GORAM-Core, MPI_INCLUDE_DIR=${MPI_INCLUDE_DIR}\n\n")
    add_library(mpi INTERFACE IMPORTED)
    target_include_directories(mpi INTERFACE 
                    $<BUILD_INTERFACE:${MPI_INCLUDE_DIR}>
                    $<INSTALL_INTERFACE:>)
endif()

target_link_libraries(aby3-GORAM-Core aby3 ${ABY3_GORAM_CORE_JSONCPP_TARGET} mpi)


# file(GLOB_RECURSE SRCS *.cpp)
 

# add_library(aby3-GORAM-Core ${SRCS})


# target_link_libraries(aby3-GORAM-Core aby3)

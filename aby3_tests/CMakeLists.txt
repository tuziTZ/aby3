
#project(aby3_Tests)

file(GLOB_RECURSE SRCS *.cpp)
add_library(aby3_Tests STATIC ${SRCS})

target_include_directories(aby3_Tests PUBLIC ${CMAKE_SOURCE_DIR}) 
target_link_libraries(aby3_Tests aby3 aby3-ML aby3-GORAM aby3-RTR)

# global graph folder
target_compile_definitions(aby3_Tests PUBLIC "GRAPH_FOLDER=\"${GRAPH_FOLDER}\"")
message(STATUS "In aby3_Tests, GRAPH_FOLDER=${GRAPH_FOLDER}")


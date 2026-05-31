
find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)

set(ABY3_BSPN_SRCS
    FlatBSPN.cpp
    MultiplierPreprocessing.cpp
)

add_library(aby3-BSPN STATIC ${ABY3_BSPN_SRCS})

target_include_directories(aby3-BSPN PUBLIC ${CMAKE_SOURCE_DIR} ${OPENSSL_INCLUDE_DIR}) 
target_link_libraries(aby3-BSPN OpenMP::OpenMP_CXX OpenSSL::SSL OpenSSL::Crypto aby3)

target_compile_options(aby3-BSPN PRIVATE ${OpenMP_CXX_FLAGS})

cmake_minimum_required(VERSION 3.11)

project(matrix CXX)

set(CMAKE_CXX_STANDARD 23)

if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS true)

    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    endif()
endif()
find_package(OpenMP REQUIRED)


add_executable(matrix "matrix.cpp")
target_link_libraries(matrix PRIVATE OpenMP::OpenMP_CXX)
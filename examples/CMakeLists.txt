cmake_minimum_required(VERSION 3.7)

project(bench LANGUAGES CXX)

set(APP_SOURCES
  bench.cpp
)

add_executable(${PROJECT_NAME} ${APP_SOURCES})

target_link_libraries(${PROJECT_NAME} LibXPSNR)

target_compile_options(${PROJECT_NAME} PRIVATE
  -mavx
  -mavx2
  -mssse3
  -msse4.1
) 

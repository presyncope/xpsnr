cmake_minimum_required(VERSION 3.7)

project(LibXPSNR LANGUAGES CXX)

set(LIB_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/xpsnr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/sa_act.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/ta_act.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/sources/x86/sa_act_sse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/x86/sa_act_avx2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/x86/ta_act_sse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/x86/ta_act_avx2.cpp
)

add_library(${PROJECT_NAME} STATIC ${LIB_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(${PROJECT_NAME} PRIVATE
  -mavx
  -mavx2
  -mssse3
  -msse4.1
  -Werror
  -Wall
) 

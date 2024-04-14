project(XpsnrTest)

add_executable(
  ${PROJECT_NAME}
  taact_test.cc
  saact_test.cc
  dist_test.cc
)
target_link_libraries(
  ${PROJECT_NAME}
  LibXPSNR
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
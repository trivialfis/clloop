cmake_minimum_required(VERSION 3.1)

option(BUILD_TEST "Enable building tests." 1)

include_directories("${CMAKE_CURRENT_LIST_DIR}")

if (BUILD_TEST)
  set(CMAKE_CXX_STANDARD 11)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/tests")
  file(COPY ${CMAKE_CURRENT_LIST_DIR}/tests DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()
install(FILES
  clloop.h
  DESTINATION ${CMAKE_INSTALL_PREFIX}/clloop/clloop.h)

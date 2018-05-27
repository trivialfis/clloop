enable_testing()

find_package(OpenCL REQUIRED)
add_executable(test testwrapper.cc)
target_link_libraries(test OpenCL::OpenCL)

add_test("test_copy" test)


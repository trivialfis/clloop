#include <vector>
#include <numeric>
#include <cassert>

#include "clloop.h"

constexpr int SIZE = 128;

struct dummy
{
  int i;
  int j;
};

int main()
{
  using namespace clloop;

  CLWrapper wrapper;

  CLKernel kernel = wrapper.eval("test_copy.cl", "copy_block");
  float *x = (float*) malloc (SIZE * sizeof(float));
  float *y = (float*) malloc (SIZE * sizeof(float));
  for (size_t i = 0; i < SIZE; ++i)
    {
      x[i] = (float)i;
    }

  kernel.in("x");
  kernel.out("y");

  kernel.set("x", x, SIZE);
  kernel.set("y", y, SIZE);

  kernel.apply(SIZE, true);
  // std::cout << std::endl;
  kernel.retrieve("y");
  for (size_t i = 0; i < SIZE; ++i)
    {
      // std::cout << "x[" << i << "]:" << x[i] << " y[" << i << "]:" << y[i] << std::endl;
      assert(x[i] * 2 - y[i] < 0.001);
    }

  std::vector<float> v (SIZE), w (SIZE);
  std::iota (v.begin(), v.end(), 0);
  for (size_t i = 0; i < 1000; ++i)
    {
      kernel.set("x", v);
      kernel.set("y", w);

      kernel.apply(SIZE, true);
    }
  for (size_t i = 0; i < SIZE; ++i)
    {
      assert(x[i] * 2 - y[i] < 0.001);
    }

  CLKernel kernel_struct = wrapper.eval("test_struct_copy.cl", "struct_copy");
  std::vector<dummy> structs_a (SIZE), structs_b (SIZE);

  for (size_t i = 0; i < SIZE; ++i)
    {
      structs_a[i].i = i;
      structs_b[i].j = i;
    }
  kernel_struct.inout("x");
  kernel_struct.inout("y");
  kernel_struct.set("x", structs_a);
  kernel_struct.set("y", structs_b);

  kernel_struct.apply(SIZE, true);

  for (size_t i = 0; i < SIZE; ++i)
    {
      assert(structs_a[i].i == structs_b[i].i);
      assert(structs_a[i].j == structs_b[i].j);
    }

  CLKernel ptr_kernel = wrapper.eval("test_pointer.cl", "walk");

  ptr_kernel.out("x");
  ptr_kernel.in("s");

  ptr_kernel.set("x", structs_a);
  ptr_kernel.set("s", SIZE);

  ptr_kernel.apply(SIZE, true);

  for (size_t i = 0; i < SIZE; ++i)
    {
      assert(structs_a[i].i == 1);
    }

  return 0;
}

#ifndef __OPENCL_VERSION__

#define __constant
#define __global
#define __local
#define __kernel
#include <math.h>
#include <stdio.h>
int get_global_id(int);

#endif
struct dummy
{
  int i;
  int j;
};

__kernel void walk(__global struct dummy *begin, int s)
{
  for (size_t i = 0; i < s; ++i)
    {
      __global struct dummy *current = begin + i;
      current->i = 1;
    }
}

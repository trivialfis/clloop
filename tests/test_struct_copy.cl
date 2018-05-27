struct dummy
{
  int i;
  int j;
};

__kernel void struct_copy(__global struct dummy *x, __global struct dummy *y)
{
  size_t gid = get_global_id(0);

  x[gid].j = y[gid].j;
  y[gid].i = x[gid].i;
}

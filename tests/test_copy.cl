__kernel void copy_block(__global float *x, __global float *y)
{
  size_t gid = get_global_id(0);
  y[gid] = x[gid] * 2;
}


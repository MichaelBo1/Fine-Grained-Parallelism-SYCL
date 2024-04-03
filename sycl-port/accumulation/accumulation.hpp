#ifndef __ACCUMULATION_H__
#define __ACCUMULATION_H__

#include "../fmm/BoxId_Interface.hpp"
namespace accumulation
{

// TO DO: CONSIDER GRAPH DR DATA STRUCTURE
using CudaOctreeInt = OctreeStorageDense<
    gtl::simple_vector<int, gtl::cuda_allocator<int>>,
    gtl::cuda_allocator<int>>;

using CudaOctreeReal= OctreeStorageDense<
    gtl::simple_vector<float, gtl::cuda_allocator<float>>,
    gtl::cuda_allocator<float>>;


typedef BoxID<BoxID4D<int>> BoxId;
// FIXME: has to be a host device function - however, we need to call device-only function
// atomicAdd()
__device__ void
acc(BoxId boxid,
    CudaOctreeInt &dc_octree,
    CudaOctreeReal &comp_octree)
{
    // execute could get laneid as parameter to specify on which vector-element to work on
    BoxId parent = boxid.parent();
    // TODO: accumulate (32 threads should step into this function at once; each should work on
    // comp_octree[parent][laneid]
    // TODO: syncthreads
    // update dependency manager

    // if (dc_octree[parent][0] > 0)
    // {
    atomicAdd(&(comp_octree[parent][0]), comp_octree[boxid][0]);
    atomicSub(&(dc_octree[parent][0]), 1);
    // printf("B%dT%d, doing the task D%dI%ld\n", blockIdx.x, threadIdx.x,  boxid.d(), boxid.sfcindex());
    // }
    // else
    // {
    //   printf("Error B%dT%d, the task D%dI%ld was done twice\n", blockIdx.x, threadIdx.x,  boxid.d(), boxid.sfcindex());
    //   BoxId root(0, 0, 0, 0);
    //   atomicExch(&(dc_octree[root][0]), -1); // Exit
    // }
}



__device__ void
acc_down(BoxId boxid,
         CudaOctreeInt &dc_octree,
         CudaOctreeReal &comp_octree)
{
    for (unsigned int i = 0; i < 2; i ++)
    {
        for (unsigned int j = 0; j < 2; j ++)
        {
            for (unsigned int k = 0; k< 2; k ++)
            {
                BoxId child = boxid.child(i, j, k);
                atomicAdd(&(comp_octree[child][1]), comp_octree[boxid][1]);
                atomicAdd(&(comp_octree[child][2]), comp_octree[boxid][2]);
                atomicSub(&(dc_octree[child][1]), 1);
            }
        }
    }
}


}  // namespace accumulation
#endif

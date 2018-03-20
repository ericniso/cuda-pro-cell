#include <iostream>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include "simulation/proliferation.h"
#include "simulation/cell.h"
#include "simulation/data_types.h"

namespace procell { namespace simulation
{

__host__
void
proliferate(cell_type* d_params, uint64_t params_size,
            uint64_t size, cell* h_cells)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0 /* TODO check devices number */);
    
    uint64_t random_seed = time(NULL);

    uint64_t new_size = size;
    cell* d_current_stage = NULL;
    cudaMalloc((void**) &d_current_stage, size * sizeof(cell));
    cudaMemcpy(d_current_stage, h_cells, size * sizeof(cell), cudaMemcpyHostToDevice);

    uint64_t i = 0;
    // TODO: Decide stop policy
    while (i < 1)
    {
        uint64_t original_size = new_size;
        uint16_t n_threads_per_block = prop.maxThreadsPerBlock;
        uint16_t n_blocks = round(0.5 + original_size / n_threads_per_block);
        new_size = new_size * 2; // Double the size
        
        cell* d_next_stage = NULL;
        cudaMalloc((void**) &d_next_stage, new_size * sizeof(cell));

        device::proliferate<<<n_blocks, n_threads_per_block>>>
            (d_params, params_size,
            original_size, d_current_stage, d_next_stage,
            random_seed);

        cudaDeviceSynchronize();
        
        i++;
    }

    cudaFree(d_current_stage);
}
    
namespace device
{
    
__global__
void
proliferate(cell_type* d_params, uint64_t size,
            uint64_t original_size, cell* current_stage, cell* next_stage,
            uint64_t seed)
{
    uint64_t id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < original_size)
    {
        cell current = current_stage[id];
        current.t += current.timer;
        uint64_t shifted_id = id * 2; // Each thread generates two cells
        double_t fluorescence = current.fluorescence / 2;
        int32_t type = current.type;
        double_t t = current.t;

        cell c1 = create_cell(d_params, size, seed + id,
            type, fluorescence, t);

        cell c2 = create_cell(d_params, size, seed + id,
            type, fluorescence, t);

        next_stage[shifted_id] = c1;
        next_stage[shifted_id + 1] = c2;
    }

}
    
} // End device namespace
    
} // End simulation namespace
    
} // End procell namespace

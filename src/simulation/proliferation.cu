#include <iostream>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <thrust/device_vector.h>
#include "simulation/proliferation.h"
#include "simulation/cell.h"
#include "simulation/data_types.h"

namespace procell { namespace simulation
{

__host__
void
proliferate(cell_type* d_params, uint64_t params_size,
            uint64_t size, cell* h_cells, double_t t_max, double_t threshold)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0 /* TODO check devices number */);

    cell* h_active_cells = NULL;
    uint64_t new_size = remove_quiescent_cells(h_cells, &h_active_cells, size);
    cell* d_current_stage = NULL;
    cudaMalloc((void**) &d_current_stage, new_size * sizeof(cell));
    cudaMemcpy(d_current_stage, h_active_cells, new_size * sizeof(cell),
        cudaMemcpyHostToDevice);

    uint64_t i = 0;
    // TODO: Decide stop policy
    while (i < 1)
    {
        uint64_t events_size = new_size;
        uint8_t* d_proliferation_events = NULL;
        cudaMalloc((void**) &d_proliferation_events, new_size * sizeof(uint8_t));

        uint64_t random_seed = time(NULL);

        uint64_t original_size = new_size;
        uint16_t n_threads_per_block = prop.maxThreadsPerBlock;
        uint16_t n_blocks = round(0.5 + original_size / n_threads_per_block);
        new_size = new_size * 2; // Double the size
        
        cell* d_next_stage = NULL;
        cudaMalloc((void**) &d_next_stage, new_size * sizeof(cell));

        device::proliferate<<<n_blocks, n_threads_per_block>>>
            (d_params, params_size,
            original_size, d_current_stage, d_next_stage,
            d_proliferation_events,
            threshold,
            t_max,
            random_seed);

        cudaDeviceSynchronize();
        
        cudaFree(d_current_stage);
        d_current_stage = d_next_stage;

        cudaFree(d_proliferation_events);

        i++;
    }

    cudaFree(d_current_stage);
}

__host__
uint64_t
remove_quiescent_cells(cell* h_cells, cell** h_new_population, uint64_t size)
{
    device_cells d_c;

    for (uint64_t i = 0; i < size; i++)
    {
        if (h_cells[i].timer > 0.0)
            d_c.push_back(h_cells[i]);
    }

    uint64_t new_size = d_c.size();
    *h_new_population = (cell*) malloc(new_size * sizeof(cell));
    thrust::copy(d_c.begin(), d_c.end(), *h_new_population);
    
    d_c.clear();
    d_c.shrink_to_fit();


    return new_size;
}

namespace device
{
    
__global__
void
proliferate(cell_type* d_params, uint64_t size,
            uint64_t original_size, cell* current_stage, cell* next_stage,
            uint8_t* proliferation_events,
            double_t fluorescence_threshold,
            double_t t_max,
            uint64_t seed)
{
    uint64_t id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < original_size)
    {
        cell current = current_stage[id];

        if (current.timer > 0.0)
            current.t += current.timer;
            
        uint64_t shifted_id = id * 2; // Each thread generates two cells
        double_t fluorescence = current.fluorescence / 2;

        // Don't divide fluorescence if cell is quiescent
        if (current.timer < 0.0)
            fluorescence = current.fluorescence;

        int32_t type = current.type;
        double_t t = current.t;

        seed += current.timer * 10000;

        cell c1 = create_cell(d_params, size, seed + id,
            type, fluorescence, t);
        
        seed -= current.timer * 10000;

        cell c2 = create_cell(d_params, size, seed + id,
            type, fluorescence, t);

        proliferation_events[id] = 
            (current.timer < 0.0
                || fluorescence < fluorescence_threshold
                || current.t > t_max)
                ? 0 : 1;

        next_stage[shifted_id] = c1;
        next_stage[shifted_id + 1] = c2;
    }

}
    
} // End device namespace
    
} // End simulation namespace
    
} // End procell namespace

#include <iostream>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "simulation/data_types.h"
#include "simulation/cell.h"
#include "simulation/cells_population.h"
#include "utils/util.h"

namespace procell { namespace simulation
{

__host__
void
create_cells_population(cell_type* d_params, uint64_t params_size,
                        uint64_t initial_size,
                        fluorescences& h_input, initial_bounds& h_bounds,
                        cell* h_cells)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0 /* TODO check devices number */);
    
    uint16_t n_threads_per_block = prop.maxThreadsPerBlock;
    uint16_t n_blocks = round(0.5 + h_input.size() / n_threads_per_block);

    uint64_t bytes = initial_size * sizeof(cell);

    uint64_t random_seed = time(NULL);

    cell* d_cells = NULL;
    fluorescence* d_fluorescences = NULL;
    uint64_t* d_bounds = NULL;

    cudaMalloc((void**) &d_cells, bytes);
    cudaMalloc((void**) &d_fluorescences, h_input.size() * sizeof(fluorescence));
    cudaMalloc((void**) &d_bounds, h_input.size() * sizeof(uint64_t));
    
    thrust::copy(h_input.begin(), h_input.end(), d_fluorescences);
    thrust::copy(h_bounds.begin(), h_bounds.end(), d_bounds);

    device::create_cells_from_fluorescence<<<n_blocks, n_threads_per_block>>>
        (n_threads_per_block, d_params, params_size, random_seed,
        h_input.size(), d_fluorescences,
        d_bounds,
        initial_size, d_cells);

    cudaDeviceSynchronize();

    cudaMemcpy(h_cells, d_cells, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_fluorescences);
    cudaFree(d_bounds);
    cudaFree(d_cells);
}

__host__
cell_type
create_cell_type(int32_t name, double_t probability,
                    double_t timer, double_t sigma)
{

    cell_type type =
    {
        .name = name,
        .probability = probability,
        .timer = timer,
        .sigma = sigma
    };

    return type;
}

namespace device
{

__global__
void
create_cells_from_fluorescence(uint64_t n_threads_per_block,
                                cell_type* d_params, uint64_t size,
                                uint64_t seed,
                                uint64_t groups_count, fluorescence* data,
                                uint64_t* bounds,
                                uint64_t initial_size, cell* d_cells)
{
    uint64_t id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < groups_count)
    {
        double_t f = data[id].value;
        uint64_t total = data[id].frequency;
        uint16_t n_blocks = round(0.5 + total / n_threads_per_block);

        device::create_cells_population<<<n_blocks, n_threads_per_block>>>
            (d_params, size, seed, total, bounds[id], d_cells, f);

    }
}

__global__
void
create_cells_population(cell_type* d_params, uint64_t size,
                        uint64_t seed, uint64_t initial_size,
                        uint64_t offset,
                        cell* d_cells, double_t fluorescence_value)
{
    uint64_t id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < initial_size)
    {
        seed = seed + id + fluorescence_value * 10000;

        cell c = create_cell(d_params, size, seed,
                            -1, fluorescence_value, 0);
        d_cells[id + offset] = c;
    }

}
    
} // End device namespace

} // End simulation namespace

} // End procell namespace

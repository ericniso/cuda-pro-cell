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
__device__
bool
operator<(const cell_type& lhs, const cell_type& rhs)
{
    return lhs.probability > rhs.probability;
};

struct cell_type_reduce_binary :
    public thrust::binary_function<cell_type, cell_type, cell_type>
{
    __device__
    __host__
    cell_type
    operator()(cell_type& c1, cell_type& c2)
    {
        cell_type t =
        {
            .name = 0,
            .probability = c1.probability + c2.probability,
            .timer = 0.0,
            .sigma = 0.0
        };

        return t;
    }

};

__host__
void
assert_probability_sum(cell_types& h_params)
{
    cell_type base =
    {
        .name = 0,
        .probability = 0.0,
        .timer = 0.0,
        .sigma = 0.0
    };

    cell_type result =
        thrust::reduce(h_params.begin(), h_params.end(),
                        base, cell_type_reduce_binary());

    double_t err = 1 / pow(10.0, 15.0);
    if ((1.0 - result.probability) > err)
    {
        std::cout << "ERROR: probability distribution of cell types does not sum to 1, aborting." << std::endl;
        exit(EXIT_FAILURE);
    }
}

__host__
void
create_cells_population(cell_types& h_params, uint64_t initial_size,
                        fluorescences& h_input, cell* h_cells)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0 /* TODO check devices number */);
    
    uint16_t n_threads_per_block = prop.maxThreadsPerBlock;
    uint16_t n_blocks = round(0.5 + h_input.size() / n_threads_per_block);

    uint64_t bytes = initial_size * sizeof(cell);

    uint64_t random_seed = time(NULL);

    cell* d_cells = NULL;
    cell_type* d_params = NULL;
    fluorescence* d_fluorescences = NULL;
    cudaMalloc((void**) &d_cells, bytes);
    cudaMalloc((void**) &d_fluorescences, h_input.size() * sizeof(fluorescence));
    cudaMalloc((void**) &d_params, h_params.size() * sizeof(cell_type));
    
    assert_probability_sum(h_params);

    thrust::sort(h_params.begin(), h_params.end());
    thrust::copy(h_params.begin(), h_params.end(), d_params);
    thrust::copy(h_input.begin(), h_input.end(), d_fluorescences);

    device::create_cells_from_fluorescence<<<n_blocks, n_threads_per_block>>>
        (n_threads_per_block, d_params, h_params.size(), random_seed,
        h_input.size(), d_fluorescences,
        initial_size, d_cells);

    cudaDeviceSynchronize();

    cudaMemcpy(h_cells, d_cells, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_params);
    cudaFree(d_fluorescences);
    cudaFree(d_cells);
}

__host__
cell_type
create_cell_type(uint32_t name, double_t probability,
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
                                uint64_t initial_size, cell* d_cells)
{
    uint64_t id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < groups_count)
    {
        double_t f = data[id].value;
        uint64_t total = data[id].frequency;
        uint16_t n_blocks = round(0.5 + total / n_threads_per_block);

        device::create_cells_population<<<n_blocks, n_threads_per_block>>>
            (d_params, size, seed, initial_size, d_cells, f);

    }
}

__global__
void
create_cells_population(cell_type* d_params, uint64_t size,
                        uint64_t seed, uint64_t initial_size,
                        cell* d_cells, double_t fluorescence_value)
{
    uint64_t id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < initial_size)
    {
        cell c = create_cell(d_params, size, seed + id, fluorescence_value);
        d_cells[id] = c;
    }

}
    
} // End device namespace

} // End simulation namespace

} // End procell namespace

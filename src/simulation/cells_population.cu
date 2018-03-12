#include <inttypes.h>
#include <math.h>
#include <time.h>
#include "simulation/cell.h"
#include "simulation/cells_population.h"
#include "utils/util.h"

namespace procell { namespace simulation
{

void
create_cells_population(cells_population_parameters& h_params,
                        uint64_t initial_size, cell* h_cells)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0 /* TODO check devices number */);
    
    uint16_t n_threads_per_block = prop.maxThreadsPerBlock;
    uint16_t n_blocks = ceil(initial_size / n_threads_per_block);

    uint64_t bytes = initial_size * sizeof(cell);

    uint64_t random_seed = time(NULL);

    cell* d_cells = NULL;
    cells_population_parameters* d_params = NULL;
    cudaMalloc((void**) &d_cells, bytes);
    cudaMalloc((void**) &d_params, sizeof(cells_population_parameters));
    cudaMemcpy(d_params, &h_params,
                sizeof(cells_population_parameters), cudaMemcpyHostToDevice);

    device::create_cells_population<<<n_blocks, n_threads_per_block>>>
        (d_params, random_seed, initial_size, d_cells);

    cudaDeviceSynchronize();

    cudaMemcpy(h_cells, d_cells, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_params);
    cudaFree(d_cells);
}

namespace device
{
    
__global__
void
create_cells_population(cells_population_parameters* d_params,
                        uint64_t seed, uint64_t initial_size,
                        cell* d_cells)
{
    uint64_t id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < initial_size)
    {
        // TODO remove, just for testing
        double_t fluorescence = utils::device::uniform_random(seed + id);
        cell c = create_cell(fluorescence);
        d_cells[id] = c;
    }

}
    
} // End device namespace

} // End simulation namespace

} // End procell namespace

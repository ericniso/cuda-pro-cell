#include <inttypes.h>
#include <math.h>
#include "simulation/cell.h"
#include "simulation/cells_population.h"

namespace procell { namespace simulation
{

void
create_cells_population(uint64_t initial_size, cell* h_cells)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0 /* TODO check devices number */);
    
    uint16_t n_threads_per_block = prop.maxThreadsPerBlock;
    uint16_t n_blocks = ceil(initial_size / n_threads_per_block);

    uint64_t bytes = initial_size * sizeof(cell);

    cell* d_cells;
    cudaError_t err = cudaMalloc((void**) &d_cells, bytes);
    
    device::create_cells_population<<<n_blocks, n_threads_per_block>>>
        (initial_size, d_cells);

    cudaDeviceSynchronize();

    cudaMemcpy(h_cells, d_cells, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_cells);
}

namespace device
{
    
__global__
void
create_cells_population(uint64_t n, cell* d_cells)
{
    uint64_t id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < n)
    {
        cell c;
        c.t = 0;
        d_cells[id] = c;
    }

}
    
} // End device namespace

} // End simulation namespace

} // End procell namespace

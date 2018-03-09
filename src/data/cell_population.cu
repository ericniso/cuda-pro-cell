#include "data/cell_population.h"

namespace procell
{

void create_cell_population()
{
    device::create_cell_population<<<(1 << 10), (1 << 10)>>>();
    cudaDeviceSynchronize();
}

namespace device
{
    
__global__ void create_cell_population()
{
    
}
    
} // End device namespace

} // End procell namespace

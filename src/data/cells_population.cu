#include <inttypes.h>
#include "data/cell.h"
#include "data/cells_population.h"

namespace procell
{

void
create_cells_population(uint64_t initial_size)
{
    device::create_cells_population<<<(1 << 10), (1 << 10)>>>();
    cudaDeviceSynchronize();
}

namespace device
{
    
__global__
void
create_cells_population()
{
    
}
    
} // End device namespace

} // End procell namespace

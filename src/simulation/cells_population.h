#ifndef PROCELL_CELLS_POPULATION_H
#define PROCELL_CELLS_POPULATION_H

#include <inttypes.h>
#include "simulation/cell.h"

namespace procell { namespace simulation
{

struct cells_population_parameters
{
    
};

void
create_cells_population(cells_population_parameters& h_params,
                        uint64_t initial_size, cell* h_cells);

namespace device
{
    
__global__
void
create_cells_population(cells_population_parameters* d_params,
                        uint64_t seed, uint64_t initial_size,
                        cell* d_cells);
    
} // End device namespace

} // End simulation namespace

} // End procell namespace

#endif // PROCELL_CELLS_POPULATION_H

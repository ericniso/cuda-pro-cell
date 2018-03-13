#ifndef PROCELL_CELLS_POPULATION_H
#define PROCELL_CELLS_POPULATION_H

#include <inttypes.h>
#include <math.h>
#include "simulation/data_types.h"
#include "simulation/cell.h"

namespace procell { namespace simulation
{

__host__
__device__
bool
operator<(const cell_type& lhs, const cell_type& rhs);

void
create_cells_population(cell_types& h_params,
                        uint64_t initial_size, cell* h_cells);

__host__
cell_type
create_cell_type(uint32_t name, double_t probability,
                    double_t timer, double_t sigma);

namespace device
{
    
__global__
void
create_cells_population(cell_type* d_params, uint64_t size,
                        uint64_t seed, uint64_t initial_size,
                        cell* d_cells);
    
} // End device namespace

} // End simulation namespace

} // End procell namespace

#endif // PROCELL_CELLS_POPULATION_H

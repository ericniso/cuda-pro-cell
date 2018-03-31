#ifndef PROCELL_CELLS_POPULATION_H
#define PROCELL_CELLS_POPULATION_H

#include <inttypes.h>
#include <math.h>
#include "simulation/data_types.h"
#include "simulation/cell.h"

namespace procell { namespace simulation
{

__host__
void
create_cells_population(simulation::cell_types& h_params,
                        uint64_t initial_size,
                        simulation::fluorescences& h_input,
                        simulation::initial_bounds& h_bounds,
                        cell* h_cells);

__host__
cell_type
create_cell_type(int32_t name, double_t probability,
                    double_t timer, double_t sigma);

namespace device
{
    
__global__
void
create_cells_from_fluorescence(uint64_t n_threads_per_block,
                                cell_type* d_params, uint64_t size,
                                uint64_t seed,
                                fluorescence* data, uint64_t groups_count,
                                uint64_t* bounds,
                                uint64_t initial_size, cell* d_cells);

__global__
void
create_cells_population(cell_type* d_params, uint64_t size,
                        uint64_t seed, uint64_t initial_size,
                        uint64_t offset,
                        cell* d_cells, double_t fluorescence_value);
    
} // End device namespace

} // End simulation namespace

} // End procell namespace

#endif // PROCELL_CELLS_POPULATION_H

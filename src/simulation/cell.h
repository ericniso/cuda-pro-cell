#ifndef PROCELL_CELL_H
#define PROCELL_CELL_H

#include <inttypes.h>
#include <math.h>
#include "simulation/data_types.h"

namespace procell { namespace simulation
{

namespace device
{

__device__
cell
create_cell(cell_type* params, uint64_t size, uint64_t random_seed, double_t fluorescence);

__device__
uint64_t
determine_cell_type(cell& c, cell_type* params, uint64_t size, uint64_t random_seed);

__device__
void
determine_cell_timer(cell& c, cell_type& param, uint64_t random_seed);

} // End device namespace

} // End simulation namespace

} // End procell namespace

#endif // PROCELL_CELL_H

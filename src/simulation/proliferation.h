#ifndef PROCELL_PROLIFERATION_H
#define PROCELL_PROLIFERATION_H

#include <inttypes.h>
#include <math.h>
#include "simulation/data_types.h"
#include "simulation/cell.h"

namespace procell { namespace simulation
{

__host__
void
proliferate(cell_type* d_params, uint64_t params_size,
            uint64_t size, cell* h_cells, double_t t_max, double_t threshold);

__host__
uint64_t
remove_quiescent_cells(cell* h_cells, cell** h_new_population, uint64_t size);

namespace device
{

__global__
void
proliferate(cell_type* d_params, uint64_t size,
            uint64_t original_size, cell* current_stage, cell* next_stage,
            uint8_t* future_proliferation_events,
            double_t fluorescence_threshold,
            double_t t_max,
            uint64_t seed);

} // End device namespace

} // End simulation namespace

} // End procell namespace

#endif // PROCELL_PROLIFERATION_H

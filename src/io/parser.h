#ifndef PROCELL_LOADER_H
#define PROCELL_LOADER_H

#include <inttypes.h>
#include <thrust/device_vector.h>
#include "simulation/data_types.h"
#include "simulation/cells_population.h"

namespace procell { namespace io
{

__host__
uint64_t
load_fluorescences(char* histogram, simulation::fluorescences& data,
                    simulation::initial_bounds& bounds);

__host__
simulation::cell_type*
load_cell_types(char* types, simulation::cell_types& data);

__host__
bool
save_fluorescences(char* filename,
                    uint64_t size, simulation::fluorescence* data);

} // End io namespace

} // End procell namespace

#endif // PROCELL_LOADER_H

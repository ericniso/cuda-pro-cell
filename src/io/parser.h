#ifndef PROCELL_LOADER_H
#define PROCELL_LOADER_H

#include <inttypes.h>
#include <thrust/device_vector.h>
#include <map>
#include <math.h>
#include "simulation/data_types.h"
#include "simulation/cells_population.h"

namespace procell { namespace io
{

__host__
void
load_fluorescences(char* histogram, simulation::fluorescences& data,
                    simulation::initial_bounds& bounds,
                    simulation::fluorescences& predicted_values,
                    double_t& threshold,
                    uint64_t* size);

__host__
void
load_cell_types(char* types, simulation::cell_types& data);

__host__
bool
save_fluorescences(char* filename, 
                    simulation::fluorescences& results);

} // End io namespace

} // End procell namespace

#endif // PROCELL_LOADER_H
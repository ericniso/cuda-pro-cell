#ifndef PROCELL_LOADER_H
#define PROCELL_LOADER_H

#include <iostream>
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
load_fluorescences(const char* histogram, simulation::fluorescences& data,
                    simulation::initial_bounds& bounds,
                    simulation::fluorescences_result& predicted_values,
                    double_t& threshold,
                    uint64_t* size);

__host__
void
load_cell_types(const char* types, simulation::cell_types& data);

__host__
bool
save_fluorescences(std::ostream& stream,
                    bool save_ratio,
                    int32_t ratio_size,
                    simulation::fluorescences_result& results);

} // End io namespace

} // End procell namespace

#endif // PROCELL_LOADER_H

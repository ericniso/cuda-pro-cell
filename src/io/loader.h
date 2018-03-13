#ifndef PROCELL_LOADER_H
#define PROCELL_LOADER_H

#include <inttypes.h>
#include <thrust/device_vector.h>
#include "simulation/data_types.h"

namespace procell { namespace io
{

__host__
uint64_t
load_fluorescences(char* histogram, simulation::fluorescences& data);

} // End io namespace

} // End procell namespace

#endif // PROCELL_LOADER_H
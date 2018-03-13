#ifndef PROCELL_DATA_TYPES_H
#define PROCELL_DATA_TYPES_H

#include <inttypes.h>
#include <thrust/device_vector.h>

namespace procell { namespace simulation
{

struct cell
{
    uint8_t type;
    double_t fluorescence;
    double_t timer;
    double_t t;
};

struct cell_type
{
    uint32_t name;
    double_t probability;
    double_t timer;
    double_t sigma;
};

typedef thrust::device_vector<cell_type> cell_types;

} // End simulation namespace

} // End procell namespace

#endif // PROCELL_CELLS_POPULATION_H

#ifndef PROCELL_DATA_TYPES_H
#define PROCELL_DATA_TYPES_H

#include <inttypes.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace procell { namespace simulation
{

struct cell
{
    int32_t type;
    double_t fluorescence;
    double_t timer;
    double_t t;
};

struct cell_type
{
    int32_t name;
    double_t probability;
    double_t timer;
    double_t sigma;
};

struct fluorescence
{
    double_t value;
    uint64_t frequency;
};

typedef thrust::host_vector<cell_type> cell_types;

typedef thrust::host_vector<fluorescence> fluorescences;

typedef thrust::host_vector<fluorescence> fluorescences_result;

typedef thrust::host_vector<uint64_t> initial_bounds;

typedef uint8_t proliferation_event;

namespace device
{

typedef thrust::device_vector<cell_type> cell_types;

typedef thrust::device_vector<fluorescence> fluorescences;

typedef thrust::device_vector<uint64_t> initial_bounds;

typedef thrust::device_vector<cell> device_cells;

} // End device namespace

} // End simulation namespace

} // End procell namespace

#endif // PROCELL_CELLS_POPULATION_H

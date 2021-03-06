#ifndef PROCELL_DATA_TYPES_H
#define PROCELL_DATA_TYPES_H

#include <inttypes.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <map>

namespace procell { namespace simulation
{

struct cell
{
    int32_t type;
    double_t fluorescence;
    double_t timer;
    double_t t;
    uint8_t state;
};

struct cell_type
{
    int32_t name;
    double_t proportion;
    double_t timer;
    double_t sigma;
};

struct fluorescence
{
    double_t value;
    uint64_t frequency;
};

struct fluorescence_with_ratio
{
    double_t value;
    uint64_t frequency;
    int32_t* ratio;
};

typedef thrust::host_vector<cell_type> cell_types;

typedef thrust::host_vector<fluorescence> fluorescences;

typedef thrust::host_vector<fluorescence_with_ratio> fluorescences_result;

typedef thrust::host_vector<uint64_t> initial_bounds;

typedef thrust::host_vector<cell> host_cells;

typedef thrust::host_vector<double_t> host_fluorescences;

typedef thrust::host_vector<double_t> host_histogram_values;

typedef thrust::host_vector<uint64_t> host_histogram_counts;

typedef std::map<double_t, uint64_t> host_map_results;

typedef thrust::host_vector<cell*> host_tree_levels;

namespace device
{

typedef thrust::device_vector<cell_type> cell_types;

typedef thrust::device_vector<fluorescence> fluorescences;

typedef thrust::device_vector<uint64_t> initial_bounds;

typedef thrust::device_vector<cell> device_cells;

typedef thrust::device_vector<double_t> device_fluorescences;

typedef thrust::device_vector<double_t> device_histogram_values;

typedef thrust::device_vector<uint64_t> device_histogram_counts;

typedef thrust::device_vector<cell*> device_tree_levels;

} // End device namespace

} // End simulation namespace

} // End procell namespace

#endif // PROCELL_CELLS_POPULATION_H

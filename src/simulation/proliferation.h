#ifndef PROCELL_PROLIFERATION_H
#define PROCELL_PROLIFERATION_H

#include <inttypes.h>
#include <math.h>
#include "simulation/data_types.h"
#include "simulation/cell.h"

namespace procell { namespace simulation
{

__host__
bool
proliferate(simulation::cell_types& h_params,
            uint64_t size, cell* h_cells, double_t t_max, double_t threshold,
            host_histogram_values& result_values,
            host_histogram_counts& result_counts);

__host__
void
run_iteration(device::cell_types& d_params, double_t t_max, double_t threshold,
                uint32_t max_threads_per_block, cell** d_current_stage,
                uint64_t& current_size, uint64_t depth);

__host__
void
copy_result(host_histogram_values& result_values,
            host_histogram_counts& result_counts,
            host_fluorescences& h_results);
            
__host__
uint64_t
count_future_proliferation_events(cell** d_stage,
    uint64_t size,
    host_fluorescences& h_results);

__host__
void
create_histogram(device::device_histogram_values& result_values,
                device::device_histogram_counts& result_counts,
                host_fluorescences& result_stage);

namespace device
{

__global__
void
proliferate(cell_type* d_params, uint64_t size,
            uint64_t original_size,
            cell** cell_tree_levels,
            double_t fluorescence_threshold,
            double_t t_max,
            uint64_t seed,
            uint64_t depth,
            uint64_t current_depth,
            uint64_t offset);

__device__
bool
cell_will_divide(cell& c, double_t fluorescence_threshold, double_t t_max);

} // End device namespace

} // End simulation namespace

} // End procell namespace

#endif // PROCELL_PROLIFERATION_H

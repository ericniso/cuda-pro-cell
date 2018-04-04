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
copy_result(host_histogram_values& result_values,
            host_histogram_counts& result_counts,
            device::device_histogram_values& partial_result_values,
            device::device_histogram_counts& partial_result_counts);
            
__host__
uint64_t
count_future_proliferation_events(cell** d_stage, proliferation_event* d_events,
    uint64_t size,
    device::device_histogram_values& result_values,
    device::device_histogram_counts& result_counts);

__host__
void
update_results(device::device_histogram_values& result_values,
                device::device_histogram_counts& result_counts,
                host_fluorescences& result_stage);


__host__
void
merge_histograms(device::device_histogram_values& result_values,
                device::device_histogram_counts& result_counts,
                device::device_histogram_values& new_result_values,
                device::device_histogram_counts& new_result_counts);

namespace device
{

__global__
void
proliferate(cell_type* d_params, uint64_t size,
            uint64_t original_size, cell* current_stage, cell* next_stage,
            proliferation_event* future_proliferation_events,
            double_t fluorescence_threshold,
            double_t t_max,
            uint64_t seed);

__device__
bool
cell_will_divide(cell& c, double_t fluorescence_threshold, double_t t_max);

} // End device namespace

} // End simulation namespace

} // End procell namespace

#endif // PROCELL_PROLIFERATION_H

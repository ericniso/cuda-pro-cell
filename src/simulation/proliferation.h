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
            uint64_t size,
            uint64_t tree_depth,
            cell* h_cells, double_t t_max, double_t threshold,
            simulation::fluorescences& m_results);

__host__
void
run_iteration(device::cell_types& d_params, double_t t_max, double_t threshold,
                uint32_t max_threads_per_block, cell** d_current_stage,
                uint64_t& current_size,
                fluorescence* d_results, uint64_t d_results_size,
                bool* still_alive,
                uint64_t depth);

__host__
void
copy_result(host_histogram_values& result_values,
            host_histogram_counts& result_counts,
            host_fluorescences& h_results);

namespace device
{

__global__
void
proliferate(cell_type* d_params, uint64_t size,
            uint64_t starting_size,
            uint64_t original_size,
            cell** cell_tree_levels,
            fluorescence* d_results,
            uint64_t d_results_size,
            bool* still_alive,
            double_t fluorescence_threshold,
            double_t t_max,
            uint64_t seed,
            uint64_t depth,
            uint64_t current_depth,
            uint64_t offset);

__global__
void
apply_bounding(uint64_t original_size,
                cell** cell_tree_levels,
                uint64_t depth,
                uint64_t current_depth,
                uint64_t offset);

__device__
bool
out_of_time(cell& c, double_t t_max);

} // End device namespace

} // End simulation namespace

} // End procell namespace

#endif // PROCELL_PROLIFERATION_H

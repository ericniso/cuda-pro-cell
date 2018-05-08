#include <curand.h>
#include <curand_kernel.h>
#include "utils/util.h"
#include "simulation/data_types.h"

#define MAX_DEPTH (24)
#define BASE_TWO (2)

namespace procell { namespace utils
{

__host__
uint64_t
get_device_available_memory()
{
    uint64_t free_byte;
    uint64_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);

    return free_byte;
}

__host__
uint64_t
log_two(uint64_t n)
{
    return log(n) / log(BASE_TWO);
}

__host__
double_t
log_n(double_t n, double_t base)
{
    return log(n) / log(base);
}

__host__
uint64_t
max_recursion_depth(uint64_t initial_stage_size)
{
    uint64_t free_byte = get_device_available_memory();
    uint64_t mem_usage = initial_stage_size * (sizeof(simulation::proliferation_event)
        + sizeof(simulation::cell)
        + sizeof(simulation::proliferation_event_gap));
    uint64_t final_stage_size = initial_stage_size;
    uint64_t internal_nodes = (final_stage_size * 2) - 2 * initial_stage_size;
    uint64_t internal_nodes_mem_usage =
        internal_nodes * (sizeof(simulation::proliferation_event)
            + sizeof(simulation::cell));

    if ((mem_usage * 2 + internal_nodes_mem_usage) > free_byte)
        return 0;

    while (mem_usage + internal_nodes_mem_usage < free_byte)
    {
        if ((mem_usage * 2 + internal_nodes_mem_usage) < free_byte)
        {
            final_stage_size = final_stage_size * 2;
            internal_nodes = final_stage_size - 2 * initial_stage_size;
            internal_nodes_mem_usage =
                internal_nodes * (sizeof(simulation::proliferation_event)
                    + sizeof(simulation::cell));
        }
        mem_usage = mem_usage * 2;
    }

    uint64_t actual_depth =
        log_two(final_stage_size) - log_two(initial_stage_size);

    actual_depth = actual_depth * 0.9; // 90% memory usage

    return min(actual_depth, (uint64_t) MAX_DEPTH);
}

__host__
std::vector<double_t>
linear_space(double_t start, double_t end, uint64_t nbins)
{
    std::vector<double_t> space(nbins);
    double_t delta = (end - start) / (nbins - 1);

    for (uint64_t i = 0; i < nbins; i++)
    {
        space[i] = start + delta * i;
    }

    return space;
}

__host__
std::vector<double_t>
log_space(double_t start, double_t end, uint64_t nbins, double_t base)
{
    start = ceil(log_n(start, base));
    end = ceil(log_n(end, base));

    std::vector<double_t> space(nbins);
    std::vector<double_t> lin_space = linear_space(start, end, nbins);

    for (uint64_t i = 0; i < nbins - 1; i++)
    {
        space[i] = pow(base, lin_space[i]);
    }

    space[space.size() - 1] = pow(base, end);

    return space;
}

__host__
void
rebin(simulation::host_histogram_values& values,
    simulation::host_histogram_counts& counts,
    simulation::host_histogram_values& new_values,
    simulation::host_histogram_counts& new_counts,
    double_t lower, double_t upper, uint64_t nbins)
{
    std::vector<double_t> raw_bins = utils::log_space(lower, upper, nbins, 10);
    simulation::host_histogram_values raw_new_values(raw_bins.begin(),
        raw_bins.begin() + raw_bins.size());
    simulation::host_histogram_counts raw_new_counts(nbins, 0);

    uint64_t pos = 0;

    for (uint64_t i = 0; i < values.size(); i++)
    {
        while (values[i] > raw_new_values[pos])
        {
            pos += 1;
        }

        raw_new_counts[pos] += counts[i];
    }

    new_values = raw_new_values;
    new_counts = raw_new_counts;
}

namespace device
{

__device__
curandState_t
init_random(uint64_t seed)
{
    curandState_t state;
    curand_init(seed, 0, 0, &state);

    return state;
}

__device__
double_t
uniform_random(uint64_t seed)
{
    curandState_t state = init_random(seed);
    return curand_uniform_double(&state);
}

__device__
double_t
normal_random(uint64_t seed, double_t mean, double_t sd)
{
    curandState_t state = init_random(seed);
    double_t std_normal = curand_normal_double(&state);
    double_t adjusted_normal = std_normal * sd + mean;
    return adjusted_normal;
}

} // End device namespace
    
} // End utils namespace
    
} // End procell namespace

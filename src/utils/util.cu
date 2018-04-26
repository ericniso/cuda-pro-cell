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

    actual_depth = actual_depth * 0.7; // 70% memory usage

    return min(actual_depth, (uint64_t) MAX_DEPTH);
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
    return curand_log_normal_double(&state, mean, sd);
}

} // End device namespace
    
} // End utils namespace
    
} // End procell namespace

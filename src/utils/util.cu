#include <curand.h>
#include <curand_kernel.h>
#include "utils/util.h"
#include "simulation/data_types.h"

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
bool
compute_new_population_size_multiplier(uint64_t size, uint64_t& multiplier)
{
    uint64_t cell_size = sizeof(simulation::cell);
    uint64_t proliferation_event_size = sizeof(simulation::proliferation_event);
    uint64_t free_byte = get_device_available_memory();
    free_byte = free_byte;
    uint64_t new_multiplier = 1;
    uint64_t new_size = size * 2;
    bool full = false;
    bool fraction = false;

    if (new_size * (cell_size + proliferation_event_size) > free_byte)
    {
        fraction = true;

        while (!full)
        {
            if (new_size * (cell_size + proliferation_event_size) < free_byte)
            {
                full = true;
            }
            else
            {
                new_multiplier = new_multiplier * 2;
                new_size = new_size / 2;
            }
        }
    }
    else
    {
        while (!full)
        {
            if (new_size * (cell_size + proliferation_event_size) > free_byte)
            {
                full = true;
            }
            else
            {
                new_multiplier = new_multiplier * 2;
                new_size = new_size * 2;
            }
        }
    }

    multiplier = new_multiplier;
    return fraction;
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

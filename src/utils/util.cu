#include <curand.h>
#include <curand_kernel.h>
#include "utils/util.h"

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

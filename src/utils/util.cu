#include <curand.h>
#include <curand_kernel.h>
#include "utils/util.h"

namespace procell { namespace utils
{
    
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
normal_random(uint64_t seed)
{
    curandState_t state = init_random(seed);
    return curand_normal_double(&state);
}

} // End device namespace
    
} // End utils namespace
    
} // End procell namespace

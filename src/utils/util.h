#ifndef PROCELL_UTIL_H
#define PROCELL_UTIL_H

#include <inttypes.h>
#include <math.h>

namespace procell { namespace utils
{

__host__
uint64_t
get_device_available_memory();

__host__
bool
compute_new_population_size_multiplier(uint64_t size, uint64_t& multiplier);

namespace device
{

__device__
double_t
uniform_random(uint64_t seed);

__device__
double_t
normal_random(uint64_t seed, double_t mean, double_t sd);

} // End device namespace

} // End utils namespace

} // End procell namespace

#endif // PROCELL_UTIL_H

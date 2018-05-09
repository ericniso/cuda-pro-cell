#ifndef PROCELL_UTIL_H
#define PROCELL_UTIL_H

#include <vector>
#include <inttypes.h>
#include <math.h>
#include "simulation/data_types.h"

namespace procell { namespace utils
{

__host__
uint64_t
get_device_available_memory();

__host__
uint64_t
log_two(uint64_t n);

__host__
double_t
log_n(double_t n, double_t base);

__host__
uint64_t
max_recursion_depth(uint64_t n);

__host__
std::vector<double_t>
linear_space(double_t start, double_t end, uint64_t nbins);

__host__
std::vector<double_t>
log_space(double_t start, double_t end, uint64_t nbins, double_t base);

__host__
void
rebin(simulation::host_histogram_values& values,
    simulation::host_histogram_counts& counts,
    simulation::host_histogram_values& new_values,
    simulation::host_histogram_counts& new_counts,
    double_t lower, double_t upper, uint64_t nbins);

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

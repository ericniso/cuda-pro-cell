#include <fstream>
#include <math.h>
#include <inttypes.h>
#include "simulation/data_types.h"
#include "io/parser.h"

namespace procell { namespace io
{

struct cell_type_comparator
{
    __host__
    __device__
    bool
    operator()(const simulation::cell_type& lhs, const simulation::cell_type& rhs)
    {
        return lhs.probability > rhs.probability;
    }
};

struct cell_type_reduce_binary :
    public thrust::binary_function<simulation::cell_type,
                                    simulation::cell_type,
                                    simulation::cell_type>
{
    __device__
    __host__
    simulation::cell_type
    operator()(simulation::cell_type& c1, simulation::cell_type& c2)
    {
        simulation::cell_type t =
        {
            .name = 0,
            .probability = c1.probability + c2.probability,
            .timer = 0.0,
            .sigma = 0.0
        };

        return t;
    }

};

__host__
void
assert_probability_sum(simulation::cell_types& h_params)
{
    simulation::cell_type base =
    {
        .name = 0,
        .probability = 0.0,
        .timer = 0.0,
        .sigma = 0.0
    };

    simulation::cell_type result =
        thrust::reduce(h_params.begin(), h_params.end(),
                        base, cell_type_reduce_binary());

    double_t err = 1 / pow(10.0, 15.0);
    if (abs(1.0 - result.probability) > err)
    {
        std::cout << "ERROR: probability distribution of cell types does not sum to 1, aborting." << std::endl;
        exit(EXIT_FAILURE);
    }
}

__host__
void
load_fluorescences(char* histogram, simulation::fluorescences& data,
                    simulation::initial_bounds& bounds, uint64_t* size)
{
    uint64_t total = 0;
    std::ifstream in(histogram);

    bool first = true;
    double_t value = 0.0;
    uint64_t frequency = 0;
    simulation::fluorescence previous;
    uint64_t previous_start = 0;
    while (in >> value >> frequency)
    {
        if (frequency == 0)
            continue;

        uint64_t start_index = 0;
        if (!first)
        {
            start_index = previous_start + previous.frequency;
        }
        first = false;

        total += frequency;
        simulation::fluorescence f =
        {
            .value = value,
            .frequency = frequency
        };
        previous = f;
        previous_start = start_index;
        data.push_back(f);
        bounds.push_back(start_index);
    }

    in.close();

    *size = total;
}

__host__
void
load_cell_types(char* types, simulation::cell_types& data)
{
    std::ifstream in(types);

    int32_t name = 0;
    double_t probability = 0.0;
    double_t timer = 0.0;
    double_t sigma = 0.0;

    while (in >> probability >> timer >> sigma)
    {
        simulation::cell_type c =
            simulation::create_cell_type(name, probability, timer, sigma);

        data.push_back(c);
        
        name++;
    }

    in.close();

    assert_probability_sum(data);

    simulation::cell_type* d_params = NULL;
    cudaMalloc((void**) &d_params, data.size() * sizeof(simulation::cell_type));
    
    thrust::sort(data.begin(), data.end(), cell_type_comparator());
}

__host__
bool
save_fluorescences(char* filename, 
                    simulation::host_histogram_values& result_values,
                    simulation::host_histogram_counts& result_counts)
{
    std::ofstream out(filename);

    if (!out.is_open())
        return false;

    out.precision(10);

    uint64_t size = result_values.size();
    double_t* values = thrust::raw_pointer_cast(result_values.data());
    uint64_t* counts = thrust::raw_pointer_cast(result_counts.data());

    for (uint64_t i = 0; i < size; i++)
    {
        out << values[i] << " " << counts[i] << std::endl;
    }

    out.close();

    return true;
}

} // End io namespace
    
} // End procell namespace

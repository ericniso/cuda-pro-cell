#include <fstream>
#include <math.h>
#include <inttypes.h>
#include <map>
#include <thrust/sort.h>
#include "simulation/data_types.h"
#include "io/parser.h"

#define ERROR_EXPONENT_CHECK 8.0

namespace procell { namespace io
{

struct cell_type_comparator
{
    __host__
    __device__
    bool
    operator()(const simulation::cell_type& lhs, const simulation::cell_type& rhs)
    {
        return lhs.proportion > rhs.proportion;
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
		simulation::cell_type t;
		t.name = 0;
		t.proportion = c1.proportion + c2.proportion;
		t.timer = 0.0;
		t.sigma = 0.0;

        return t;
    }

};

__host__
void
assert_proportion_sum(simulation::cell_types& h_params)
{
	simulation::cell_type base;
	base.name = 0;
	base.proportion = 0.0;
	base.timer = 0.0;
	base.sigma = 0.0;

    simulation::cell_type result =
        thrust::reduce(h_params.begin(), h_params.end(),
                        base, cell_type_reduce_binary());

    double_t err = 1 / pow(10.0, ERROR_EXPONENT_CHECK);
    if (abs(1.0 - result.proportion) > err)
    {
        std::cout << "ERROR: proportion distribution of cell types does not sum to 1, aborting." << std::endl;
        exit(EXIT_FAILURE);
    }
}

__host__
void
load_fluorescences(const char* histogram, simulation::fluorescences& data,
                    simulation::initial_bounds& bounds,
                    simulation::fluorescences_result& predicted_values,
                    double_t& threshold,
                    uint64_t* size)
{
    simulation::host_map_results m_results;
    uint64_t total = 0;
    std::ifstream in(histogram);

    if (threshold == 0.0)
    {
        double_t value = 0.0;
        uint64_t frequency = 0;

        while (in >> value >> frequency)
        {
            if (frequency > 0)
            {
                if (threshold == 0.0 || value < threshold)
                    threshold = value;
            }
        }

        in.clear();
        in.seekg(0);
    }

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
		
		simulation::fluorescence f;
		f.value = value;
		f.frequency = frequency;

        previous = f;
        previous_start = start_index;
        data.push_back(f);
        bounds.push_back(start_index);

        double_t curr_val = value;
        while (curr_val >= threshold)
        {
            simulation::host_map_results::iterator it
                = m_results.find(curr_val);
            if (it == m_results.end())
            {
                m_results.insert(std::make_pair(curr_val, 0));
            }

            curr_val = curr_val / 2;
        }
    }

    in.close();

    for (simulation::host_map_results::iterator it = m_results.begin();
        it != m_results.end(); it++)
    {
		simulation::fluorescence_with_ratio f;
		f.value = it->first;
		f.frequency = 0;
		f.ratio = NULL;

        predicted_values.push_back(f);
    }

    *size = total;
}

__host__
void
load_cell_types(const char* types, simulation::cell_types& data)
{
    std::ifstream in(types);

    int32_t name = 0;
    double_t proportion = 0.0;
    double_t timer = 0.0;
    double_t sigma = 0.0;

    while (in >> proportion >> timer >> sigma)
    {
        simulation::cell_type c =
            simulation::create_cell_type(name, proportion, timer, sigma);

        data.push_back(c);
        
        name++;
    }

    in.close();

    assert_proportion_sum(data);

    simulation::cell_type* d_params = NULL;
    cudaMalloc((void**) &d_params, data.size() * sizeof(simulation::cell_type));
    
    thrust::sort(data.begin(), data.end(), cell_type_comparator());
}

__host__
bool
save_fluorescences(std::ostream& stream,
                    bool save_ratio,
                    int32_t ratio_size,
                    simulation::fluorescences_result& results)
{
    int32_t precision = 10;
    stream.precision(precision);

    for (uint64_t i = 0; i < results.size(); i++)
    {
        if (results[i].frequency > 0)
        {
            stream << results[i].value << "\t"
                << results[i].frequency;

            if (save_ratio)
            {
                for (int32_t j = 0; j < ratio_size; j++)
                {
                    stream << "\t" << results[i].ratio[j];
                }
            }

            stream << std::endl;
        }
    }

    return true;
}

} // End io namespace
    
} // End procell namespace

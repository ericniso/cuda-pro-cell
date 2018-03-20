#include <fstream>
#include <math.h>
#include <inttypes.h>
#include "simulation/data_types.h"
#include "io/parser.h"

namespace procell { namespace io
{

__host__
uint64_t
load_fluorescences(char* histogram, simulation::fluorescences& data,
                    simulation::initial_bounds& bounds)
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

    return total;
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
}

__host__
bool
save_fluorescences(char* filename,
                    uint64_t size, simulation::fluorescence* data)
{
    std::ofstream out(filename);

    if (!out.is_open())
        return false;

    out.precision(10);

    for (uint64_t i = 0; i < size; i++)
    {
        out << data[i].value << " " << data[i].frequency << std::endl;
    }

    out.close();

    return true;
}

} // End io namespace
    
} // End procell namespace

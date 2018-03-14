#include <fstream>
#include <math.h>
#include <inttypes.h>
#include "simulation/data_types.h"
#include "io/loader.h"

namespace procell { namespace io
{

__host__
uint64_t
load_fluorescences(char* histogram, simulation::fluorescences& data)
{
    uint64_t total = 0;
    std::ifstream in(histogram);

    bool first = true;
    double_t value = 0.0;
    uint64_t frequency = 0;
    simulation::fluorescence previous;
    while (in >> value >> frequency)
    {
        uint64_t start_index = 0;
        if (!first)
        {
            start_index = previous.start_index + previous.frequency;
        }
        first = false;

        total += frequency;
        simulation::fluorescence f =
        {
            .value = value,
            .frequency = frequency,
            .start_index = start_index
        };
        previous = f;
        data.push_back(f);
    }

    in.close();

    return total;
}

__host__
void
load_cell_types(char* types, simulation::cell_types& data)
{
    std::ifstream in(types);

    uint32_t name = 0;
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

} // End io namespace
    
} // End procell namespace

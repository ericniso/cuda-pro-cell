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

} // End io namespace
    
} // End procell namespace

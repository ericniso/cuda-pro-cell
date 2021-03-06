#include <iostream>
#include <inttypes.h>
#include <math.h>
#include "simulation/data_types.h"
#include "simulation/cell.h"
#include "utils/util.h"

std::ostream&
operator<<(std::ostream& stream, procell::simulation::cell& c)
{
    stream << "Cell type: " << c.type
        << ", fluorescence: " << c.fluorescence
        << ", timer: " << c.timer
        << ", t: " << c.t;

    return stream;
}

namespace procell { namespace simulation
{

namespace device
{

__device__
cell
create_cell(cell_type* params, uint64_t size, uint64_t random_seed,
            int32_t type, double_t fluorescence, double_t t)
{
    cell c;
    c.type = type;
    c.fluorescence = fluorescence;

    uint64_t index = 0;
    
    if (type == -1)
    {
        index = determine_cell_type(c, params, size, random_seed);
    }
    else
    {
        bool found = false;
        for (uint64_t i = 0; i < size && !found; i++)
        {
            if (type == params[i].name)
            {
                index = i;
                found = true;
            }
        }
    }

    if (params[index].timer >= 0.0)
    {
        determine_cell_timer(c, params[index], random_seed);
    }
    else
    {
        c.timer = -1.0;
    }

    if (t > 0)
    {
        c.t = t;
    }
    else
    {
        if (params[index].timer >= 0.0)
        {
            determine_cell_initial_t(c, params[index], random_seed);
        }
        else
        {
            c.t = 0;
        }
    }

    return c;
}

__device__
uint64_t
determine_cell_type(cell& c, cell_type* params, uint64_t size, uint64_t random_seed)
{
    double_t rnd = utils::device::uniform_random(random_seed);
    double accumulator = 0.0;
    bool done = false;
    uint64_t j = 0;

    for (uint64_t i = 0; i < size && !done; i++)
    {
        accumulator += params[i].proportion;

        if (rnd < accumulator)
        {
            c.type = params[i].name;
            c.timer = params[i].timer;
            j = i;
            done = true;
        }
    }

    return j;
}

__device__
void
determine_cell_timer(cell& c, cell_type& param, uint64_t random_seed)
{
    if (param.timer > 0.0)
    {
        double_t rnd = -1.0;
        
        while (rnd <= 0.0)
        {
            rnd = utils::device::normal_random(random_seed, param.timer, param.sigma);
            random_seed = random_seed * param.sigma;
        }

        c.timer = rnd;
    }
}

__device__
void
determine_cell_initial_t(cell& c, cell_type& param, uint64_t random_seed)
{
    if (param.timer > 0.0)
    {
        double_t rnd = -1.0;
        
        while (rnd <= 0.0)
        {
            rnd = utils::device::normal_random(random_seed, param.timer, param.sigma);
            random_seed = random_seed * param.sigma;
        }

        double_t uniform_rnd_factor = 
            utils::device::uniform_random(random_seed * param.sigma);

        c.t = rnd * uniform_rnd_factor;
    }
}

} // End device namespace

} // End simulation namespace

} // End procell namespace

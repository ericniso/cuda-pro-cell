#include <inttypes.h>
#include <math.h>
#include "simulation/data_types.h"
#include "simulation/cell.h"
#include "utils/util.h"

namespace procell { namespace simulation
{

namespace device
{

__device__
cell
create_cell(cell_type* params, uint64_t size, uint64_t random_seed, double_t fluorescence)
{
    cell c;
    c.fluorescence = fluorescence;
    uint64_t index = determine_cell_type(c, params, size, random_seed);
    determine_cell_timer(c, params[index], random_seed);

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
        accumulator += params[i].probability;

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
        double_t rnd = utils::device::normal_random(random_seed, param.timer, param.sigma);

        while (rnd <= 0.0)
        {
            rnd = utils::device::normal_random(random_seed, param.timer, param.sigma);
        }

        c.timer = rnd;
    }
}

} // End device namespace

} // End simulation namespace

} // End procell namespace

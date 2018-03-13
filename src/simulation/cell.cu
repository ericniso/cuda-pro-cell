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
    determine_cell_type(c, params, size, random_seed);

    return c;
}

__device__
void
determine_cell_type(cell& c, cell_type* params, uint64_t size, uint64_t random_seed)
{
    double_t rnd = utils::device::uniform_random(random_seed);
    double accumulator = 0.0;
    bool done = false;

    for (uint64_t i = 0; i < size && !done; i++)
    {
        accumulator += params[i].probability;

        if (rnd < accumulator)
        {
            c.t = params[i].name;
            done = true;
        }
    }
}

} // End device namespace

} // End simulation namespace

} // End procell namespace

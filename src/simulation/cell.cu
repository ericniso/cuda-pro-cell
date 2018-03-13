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
create_cell(cell_types* params, uint64_t random_seed, double_t fluorescence)
{
    cell c;
    c.fluorescence = fluorescence;
    determine_cell_type(c, params, random_seed);

    return c;
}

__device__
void
determine_cell_type(cell& c, cell_types* params, uint64_t random_seed)
{
    double_t rnd = utils::device::uniform_random(random_seed);
    c.t = rnd;
}

} // End device namespace

} // End simulation namespace

} // End procell namespace

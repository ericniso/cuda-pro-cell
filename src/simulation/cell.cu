#include "simulation/cell.h"

namespace procell { namespace simulation
{

namespace device
{

__device__
cell
create_cell(double_t fluorescence)
{
    cell c;
    c.fluorescence = fluorescence;

    return c;
}

} // End device namespace

} // End simulation namespace

} // End procell namespace

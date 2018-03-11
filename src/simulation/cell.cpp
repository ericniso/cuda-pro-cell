#include "simulation/cell.h"

namespace procell { namespace simulation
{

cell
create_cell(char* type, double_t fluorescence,
            double_t timer, double_t t)
{
    cell c;

    c.type = type;
    c.fluorescence = fluorescence;
    c.timer = timer;
    c.t = t;

    return c;
}

} // End simulation namespace

} // End procell namespace

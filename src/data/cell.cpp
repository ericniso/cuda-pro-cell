#include "data/cell.h"

namespace procell
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

} // End procell namespace

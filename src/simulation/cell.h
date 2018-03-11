#ifndef PROCELL_CELL_H
#define PROCELL_CELL_H

#include <cmath>

namespace procell { namespace simulation
{

struct cell
{
    char* type;
    double_t fluorescence;
    double_t timer;
    double_t t;
};

namespace device
{

__device__
cell
create_cell(double_t fluorescence);

} // End device namespace

} // End simulation namespace

} // End procell namespace

#endif // PROCELL_CELL_H

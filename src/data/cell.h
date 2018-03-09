#ifndef PROCELL_CELL_H
#define PROCELL_CELL_H

#include <cmath>

namespace procell
{

struct cell
{
    char* type;
    double_t fluorescence;
    double_t timer;
    double_t t;
};

cell
create_cell(char* type,
            double_t fluorescence,
            double_t timer,
            double_t t);

}

#endif // PROCELL_CELL_H

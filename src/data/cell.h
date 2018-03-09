#ifndef PROCELL_CELL_H
#define PROCELL_CELL_H

#include <cmath>
#include <string>

struct cell
{
    std::string type;
    double_t fluorescence;
    double_t timer;
    double_t t;
};

#endif // PROCELL_CELL_H

#ifndef PROCELL_CELL_H
#define PROCELL_CELL_H

#include <cmath>
#include <cstdint>
#include <string>

struct cell
{
    std::string type;
    double_t fluorescence;
    uint64_t timer;
    uint64_t t;

};

#endif // PROCELL_CELL_H

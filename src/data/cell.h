#ifndef PROCELL_DATA_STRUCTURES_H
#define PROCELL_DATA_STRUCTURES_H

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

#endif // PROCELL_DATA_STRUCTURES_H

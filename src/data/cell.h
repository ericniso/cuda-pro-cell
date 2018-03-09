#ifndef PROCELL_DATA_STRUCTURES_H
#define PROCELL_DATA_STRUCTURES_H

#include <cmath>
#include <cstdint>
#include <string>

struct cell
{
    std::string type;
    double_t fluorescence;
    uint64_t global_time;
    uint64_t timer;
    uint64_t t;

    cell() : type("quiescent"), fluorescence(0.0), global_time(0), timer(0), t(0)
    {

    }

};

#endif // PROCELL_DATA_STRUCTURES_H

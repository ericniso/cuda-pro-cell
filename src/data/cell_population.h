#ifndef PROCELL_CELL_POPULATION_H
#define PROCELL_CELL_POPULATION_H

#include "data/cell.h"

namespace procell
{

struct cell_population
{
    
};

void create_cell_population();

namespace device
{
    
__global__ void create_cell_population();
    
} // End device namespace

} // End procell namespace

#endif // PROCELL_CELL_POPULATION_H

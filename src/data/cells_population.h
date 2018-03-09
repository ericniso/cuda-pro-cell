#ifndef PROCELL_CELLS_POPULATION_H
#define PROCELL_CELLS_POPULATION_H

#include "data/cell.h"

namespace procell
{

struct cells_population_parameters
{
    
};

void
create_cells_population();

namespace device
{
    
__global__
void
create_cells_population();
    
} // End device namespace

} // End procell namespace

#endif // PROCELL_CELLS_POPULATION_H

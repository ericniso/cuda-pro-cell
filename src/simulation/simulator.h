#ifndef PROCELL_SIMULATOR_H
#define PROCELL_SIMULATOR_H

#include "io/cmdargs.h"
#include "simulation/data_types.h"

namespace procell { namespace simulation
{

class Simulator
{

public:
    Simulator(io::CmdArgs& args) : args(args) { this->initial_population_size = 0; };

    void
    load_params();

    void
    create_cell_population();

    bool
    start_simulation();

    void
    save_results();

private:
    io::CmdArgs args;
    fluorescences in;
    initial_bounds bounds;
    fluorescences_result predicted_values;
    uint64_t initial_population_size;
    cell* cells;
    cell_types params;
};

} // end simulation namespace

} // end procell namespaces

#endif

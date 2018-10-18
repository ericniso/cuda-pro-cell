#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <assert.h>
#include <map>
#include "simulation/data_types.h"
#include "simulation/cell.h"
#include "simulation/cells_population.h"
#include "simulation/proliferation.h"
#include "io/cmdargs.h"
#include "io/parser.h"
#include "simulation/simulator.h"

using namespace procell;

int
main(int argc, char** argv)
{
    io::CmdArgs args(argc, argv);
    simulation::Simulator simulator(args);

    simulator.load_params();
    simulator.create_cell_population();
    bool success = simulator.start_simulation();

    if (!success)
        exit(EXIT_FAILURE);

    simulator.save_results();

    return 0;
}

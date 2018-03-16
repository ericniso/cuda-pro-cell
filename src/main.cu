#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <assert.h>
#include "simulation/data_types.h"
#include "simulation/cell.h"
#include "simulation/cells_population.h"
#include "cmdline/cmdline.h"
#include "io/loader.h"

using namespace procell;

typedef struct gengetopt_args_info ggo_args;

int
main(int argc, char** argv)
{
    ggo_args ai;
    assert(cmdline_parser(argc, argv, &ai) == 0);

    char* histogram = ai.histogram_arg;
    char* types = ai.cell_types_arg;

    simulation::fluorescences in;
    uint64_t n = io::load_fluorescences(histogram, in);
    uint64_t size = n * sizeof(simulation::cell);

    simulation::cell* cells = (simulation::cell*) malloc(size);

    simulation::cell_types params;
    io::load_cell_types(types, params);

    simulation::create_cells_population(params, n, in, cells);

    free(cells);
    cmdline_parser_free(&ai);

    return 0;
}

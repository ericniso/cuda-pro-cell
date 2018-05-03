#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <assert.h>
#include "simulation/data_types.h"
#include "simulation/cell.h"
#include "simulation/cells_population.h"
#include "simulation/proliferation.h"
#include "cmdline/cmdline.h"
#include "io/parser.h"

using namespace procell;

typedef struct gengetopt_args_info ggo_args;

int
main(int argc, char** argv)
{
    ggo_args ai;
    assert(cmdline_parser(argc, argv, &ai) == 0);

    char* histogram = ai.histogram_arg;
    char* types = ai.cell_types_arg;
    char* output_file = ai.output_file_arg;
    double_t threshold = ai.phi_arg;
    double_t t_max = ai.time_max_arg;

    // Load simulation params
    simulation::fluorescences in;
    simulation::initial_bounds bounds;
    uint64_t n = 0;
    io::load_fluorescences(histogram, in, bounds, &n);

    uint64_t size = n * sizeof(simulation::cell);
    simulation::cell* cells = (simulation::cell*) malloc(size);
    simulation::cell_types params;
    io::load_cell_types(types, params);

    // Create starting cell population
    simulation::create_cells_population(params,
        n, in, bounds, cells);
    
    // Run the simulation
    simulation::host_histogram_values result_values;
    simulation::host_histogram_counts result_counts;
    bool success = simulation::proliferate(params,
        n, cells, t_max, threshold, result_values, result_counts);
    
    free(cells);

    // Save results
    io::save_fluorescences(output_file, result_values, result_counts);
    
    if (!success)
        exit(EXIT_FAILURE);

    cmdline_parser_free(&ai);
    return 0;
}

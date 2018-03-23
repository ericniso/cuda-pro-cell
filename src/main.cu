#include <iostream>
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
    double_t threshold = ai.threshold_arg;
    double_t t_max = ai.max_time_arg;

    simulation::fluorescences in;
    simulation::initial_bounds bounds;
    uint64_t n = io::load_fluorescences(histogram, in, bounds);
    uint64_t size = n * sizeof(simulation::cell);
    simulation::cell* cells = (simulation::cell*) malloc(size);
    simulation::cell_types params;
    simulation::cell_type* d_params = io::load_cell_types(types, params);

    simulation::create_cells_population(d_params, params.size(),
        n, in, bounds, cells);

    simulation::fluorescences_result results;
    simulation::proliferate(d_params, params.size(), 
        n, cells, t_max, threshold, results);

    simulation::fluorescence* h_results =
        (simulation::fluorescence*) malloc(results.size() * sizeof(simulation::fluorescence));
    thrust::copy(results.begin(), results.end(), h_results);

    io::save_fluorescences(output_file, results.size(), h_results);

    free(cells);
    cmdline_parser_free(&ai);

    return 0;
}

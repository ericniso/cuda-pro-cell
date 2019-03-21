#include <iostream>
#include <fstream>
#include "simulation/simulator.h"
#include "io/parser.h"
#include "simulation/proliferation.h"

namespace procell { namespace simulation
{

void
Simulator::load_params()
{
    io::load_fluorescences(this->args.h0.c_str(), this->in, this->bounds,
        this->predicted_values, this->args.phi_min, 
        &(this->initial_population_size));
    
    uint64_t size = this->initial_population_size * sizeof(simulation::cell);
    this->cells = (simulation::cell*) malloc(size);
    io::load_cell_types(this->args.cell_types.c_str(), this->params);
}

void
Simulator::create_cell_population()
{
    simulation::create_cells_population(this->params,
        this->initial_population_size, this->in, this->bounds, this->cells);
}

bool
Simulator::start_simulation()
{
    bool success = simulation::proliferate(this->params,
        this->initial_population_size, this->args.tree_depth, this->cells, 
        this->args.t_max, this->args.phi_min, this->predicted_values, 
        this->args.track_ratio);
    
    return success;
}

void
Simulator::save_results()
{

    char* output_histogram_path = NULL;

    if (this->args.output_histogram_given)
    {
        output_histogram_path = this->args.output_histogram.c_str();
    }

    io::save_fluorescences(output_histogram_path, 
        this->args.track_ratio, 
        this->params.size(), 
        this->predicted_values);
}

} // end simulation namespace
    
} // end procell namespaces

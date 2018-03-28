#include <iostream>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <thrust/device_vector.h>
#include "simulation/proliferation.h"
#include "simulation/cell.h"
#include "simulation/data_types.h"

#define INACTIVE 0
#define ALIVE 1
#define REMOVE 2

namespace procell { namespace simulation
{

__host__
__device__
bool
operator==(const fluorescence& l, const fluorescence& r)
{
    return l.value == r.value;
}


__host__
uint64_t
proliferate(cell_type* d_params, uint64_t params_size,
            uint64_t size, cell* h_cells, double_t t_max, double_t threshold,
            fluorescence** h_results)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0 /* TODO check devices number */);
    
    fluorescences_result results;

    cell* h_active_cells = NULL;
    cell* d_current_stage = NULL;
    uint64_t new_size =
        remove_quiescent_cells(h_cells, &h_active_cells, size, results);
    cudaMalloc((void**) &d_current_stage, new_size * sizeof(cell));
    cudaMemcpy(d_current_stage, h_active_cells, new_size * sizeof(cell),
        cudaMemcpyHostToDevice);

    while (new_size > 0)
    {
        uint64_t random_seed = time(NULL);

        uint64_t original_size = new_size;
        uint16_t n_threads_per_block = prop.maxThreadsPerBlock;
        uint16_t n_blocks = round(0.5 + new_size / n_threads_per_block);
        new_size = new_size * 2; // Double the size
        
        uint8_t* d_future_proliferation_events = NULL;
        cudaMalloc((void**) &d_future_proliferation_events,
            new_size * sizeof(uint8_t));
        
        cell* d_next_stage = NULL;
        cudaMalloc((void**) &d_next_stage, new_size * sizeof(cell));

        device::proliferate<<<n_blocks, n_threads_per_block>>>
            (d_params, params_size,
            original_size, d_current_stage, d_next_stage,
            d_future_proliferation_events,
            threshold,
            t_max,
            random_seed);

        cudaDeviceSynchronize();

        cudaFree(d_current_stage);
        d_current_stage = d_next_stage;
        
        new_size = count_future_proliferation_events(
            &d_current_stage, d_future_proliferation_events, new_size, results);

        cudaFree(d_future_proliferation_events);
    }

    cudaFree(d_current_stage);

    *h_results = (fluorescence*) malloc(results.size() * sizeof(fluorescence));
    thrust::copy(results.begin(), results.end(), *h_results);

    return results.size();
}

__host__
uint64_t
remove_quiescent_cells(cell* h_cells, cell** h_new_population,
    uint64_t size, fluorescences_result& result)
{
    device_cells d_c;

    for (uint64_t i = 0; i < size; i++)
    {
        if (h_cells[i].timer > 0.0)
        {
            d_c.push_back(h_cells[i]);
        }
        else
        {
            update_results(result, h_cells[i].fluorescence);
        }
    }

    uint64_t new_size = d_c.size();
    *h_new_population = (cell*) malloc(new_size * sizeof(cell));
    thrust::copy(d_c.begin(), d_c.end(), *h_new_population);
    
    d_c.clear();
    d_c.shrink_to_fit();


    return new_size;
}

__host__
uint64_t
count_future_proliferation_events(cell** d_stage, uint8_t* d_events,
    uint64_t size, fluorescences_result& result)
{
    device_cells new_stage;
    uint8_t* h_events = (uint8_t*) malloc(size * sizeof(uint8_t));
    cell* h_stage = (cell*) malloc(size * sizeof(cell));
    cudaMemcpy(h_events, d_events, size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stage, *d_stage, size * sizeof(cell), cudaMemcpyDeviceToHost);

    for (uint64_t i = 0; i < size; i++)
    {
        switch (h_events[i])
        {
            case INACTIVE:
            {
                update_results(result, h_stage[i].fluorescence);
            }
            break;

            case ALIVE:
            {
                new_stage.push_back(h_stage[i]);
            }
            break;

            case REMOVE:
            {
                // Do nothing
            }
            break;
        }
    }
    
    uint64_t new_size = new_stage.size();

    cudaMalloc((void**) d_stage, new_size * sizeof(cell));
    thrust::copy(new_stage.begin(), new_stage.end(), *d_stage);
    new_stage.clear();
    new_stage.shrink_to_fit();

    free(h_stage);
    free(h_events);

    return new_size;
}

__host__
void
update_results(fluorescences_result& result, double_t value)
{
    fluorescence f;
    f.value = value;
    f.frequency = 0;

    fluorescences_result::iterator it = 
        thrust::find(result.begin(), result.end(), f);

    if (it != result.end())
    {
        (*it).frequency++;
    }
    else
    {
        f.frequency++;
        result.push_back(f);
    }
}

namespace device
{
    
__global__
void
proliferate(cell_type* d_params, uint64_t size,
            uint64_t original_size, cell* current_stage, cell* next_stage,
            uint8_t* future_proliferation_events,
            double_t fluorescence_threshold,
            double_t t_max,
            uint64_t seed)
{
    uint64_t id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < original_size)
    {
        uint64_t shifted_id = id * 2; // Each thread generates two cells
        cell current = current_stage[id];

        if ((current.timer < 0.0)
            || (current.t + current.timer > t_max)
            || (current.fluorescence / 2 < fluorescence_threshold))
        {
            future_proliferation_events[shifted_id] = INACTIVE;
            future_proliferation_events[shifted_id + 1] = REMOVE;

            next_stage[shifted_id] = current;
        }
        else
        {
            current.t += current.timer;
            
            double_t fluorescence = current.fluorescence / 2;
            int32_t type = current.type;
            double_t t = current.t;

            // Differentiate seeds
            uint64_t seed_c1 = seed + current.timer * 10000 + id;
            uint64_t seed_c2 = seed - current.timer * 10000 + id;

            cell c1 = create_cell(d_params, size, seed_c1,
                type, fluorescence, t);

            cell c2 = create_cell(d_params, size, seed_c2,
                type, fluorescence, t);

            future_proliferation_events[shifted_id] = ALIVE;
            future_proliferation_events[shifted_id + 1] = ALIVE;

            next_stage[shifted_id] = c1;
            next_stage[shifted_id + 1] = c2;
        }
    }

}
    
} // End device namespace
    
} // End simulation namespace
    
} // End procell namespace

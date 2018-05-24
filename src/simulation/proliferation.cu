#include <iostream>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <map>
#include "simulation/proliferation.h"
#include "simulation/cell.h"
#include "simulation/data_types.h"
#include "utils/util.h"

#define MAX_SYNC_DEPTH (24)

#define REMOVE 0
#define ALIVE 1
#define INACTIVE 2

namespace procell { namespace simulation
{

__host__
bool
proliferate(simulation::cell_types& h_params,
            uint64_t size, cell* h_cells, double_t t_max, double_t threshold,
            host_map_results& m_results)
{
    device::cell_types d_params = h_params;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0 /* TODO check devices number */);
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_SYNC_DEPTH);

    cell* h_active_cells = h_cells;
    cell* d_current_stage = NULL;
    uint64_t new_size = size;
    cudaMalloc((void**) &d_current_stage, new_size * sizeof(cell));
    cudaMemcpy(d_current_stage, h_active_cells, new_size * sizeof(cell),
        cudaMemcpyHostToDevice);

    uint64_t divisions = 0;
    while (new_size > 0)
    {
        uint64_t depth = utils::max_recursion_depth(new_size);

        // Check if GPU has enough memory to compute next stage
        if (depth == 0)
        {
            std::cout << "--- ERROR: out of GPU memory" << std::endl;
            std::cout << "--- Total iterations: " << divisions << std::endl;
            std::cout << "--- Copying partial results to file...";
            std::cout << "copied, aborting." << std::endl;
            return false;
        }

        run_iteration(d_params,
            t_max,
            threshold,
            prop.maxThreadsPerBlock,
            &d_current_stage,
            new_size,
            depth);

        new_size = new_size * pow(2, depth);
        new_size = count_future_proliferation_events(
            &d_current_stage, new_size, m_results);

        divisions++;
    }

    return true;
}

__host__
void
run_iteration(device::cell_types& d_params, double_t t_max, double_t threshold,
    uint32_t max_threads_per_block, cell** d_current_stage,
    uint64_t& current_size, uint64_t depth)
{
    host_tree_levels h_tree_levels;
    h_tree_levels.push_back(*d_current_stage);

    for (uint8_t i = 1; i < (depth + 1); i++)
    {
        cell* level_population = NULL;
        uint32_t cell_level_size = current_size * pow(2, i);
        cudaMalloc((void**) &level_population,
            cell_level_size * sizeof(cell));

        h_tree_levels.push_back(level_population);
    }

    device::device_tree_levels d_tree_levels = h_tree_levels;

    uint64_t random_seed = time(NULL);

    uint64_t original_size = current_size;
    uint16_t n_blocks = round(0.5 + current_size / max_threads_per_block);

    device::proliferate<<<n_blocks, max_threads_per_block>>>
        (thrust::raw_pointer_cast(d_params.data()), d_params.size(),
        original_size,
        thrust::raw_pointer_cast(d_tree_levels.data()),
        threshold,
        t_max,
        random_seed,
        depth,
        0,
        0);

    cudaDeviceSynchronize();

    for (uint8_t i = 0; i < depth; i++)
    {
        cudaFree(h_tree_levels[i]);
    }

    *d_current_stage = h_tree_levels[depth];
}

__host__
uint64_t
count_future_proliferation_events(cell** d_stage, uint64_t size,
    host_map_results& m_results)
{
    host_cells new_stage;
    cell* h_stage = (cell*) malloc(size * sizeof(cell));
    cudaMemcpy(h_stage, *d_stage, size * sizeof(cell), cudaMemcpyDeviceToHost);

    for (uint64_t i = 0; i < size; i++)
    {
        switch (h_stage[i].state)
        {
            case INACTIVE:
            {
                double_t fluorescence = h_stage[i].fluorescence;
                host_map_results::iterator it
                    = m_results.find(fluorescence);
                if (it == m_results.end())
                {
                    m_results.insert(std::make_pair(fluorescence, 1));
                }
                else
                {
                    it->second = it->second + 1;
                }
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
    cudaMemcpy(*d_stage, thrust::raw_pointer_cast(new_stage.data()),
        new_size * sizeof(cell), cudaMemcpyHostToDevice);
    new_stage.clear();
    new_stage.shrink_to_fit();

    free(h_stage);

    return new_size;
}

namespace device
{
    
__global__
void
proliferate(cell_type* d_params, uint64_t size,
            uint64_t original_size,
            cell** cell_tree_levels,
            double_t fluorescence_threshold,
            double_t t_max,
            uint64_t seed,
            uint64_t depth,
            uint64_t current_depth,
            uint64_t offset)
{

    __shared__ bool proliferation;

    uint64_t id = offset + threadIdx.x + blockIdx.x * blockDim.x;

    if (id < original_size)
    {
        uint64_t next_id = id * 2;
        uint64_t next_depth = current_depth + 1;
        cell current = cell_tree_levels[current_depth][id];

        if (current_depth < depth)
        {
            if (current_depth > 0 && cell_tree_levels[current_depth][id].state != ALIVE)
            {
                if (cell_tree_levels[current_depth][id].state == INACTIVE)
                {
                    cell_tree_levels[next_depth][next_id] = current;
                    cell_tree_levels[next_depth][next_id + 1].state = REMOVE;
                }
                else
                {
                    cell_tree_levels[next_depth][next_id].state = REMOVE;
                    cell_tree_levels[next_depth][next_id + 1].state = REMOVE;
                }
            }
            else if (!cell_will_divide(current, fluorescence_threshold, t_max))
            {
                current.state = INACTIVE;
                cell_tree_levels[next_depth][next_id] = current;
                cell_tree_levels[next_depth][next_id + 1].state = REMOVE;
            }
            else
            {
                double_t fluorescence = current.fluorescence / 2;
                int32_t type = current.type;
                double_t t = current.t + current.timer;

                // Differentiate seeds
                uint64_t seed_c1 = seed + current.timer * 10000 + id;
                uint64_t seed_c2 = seed - current.timer * 10000 + id;

                cell c1 = create_cell(d_params, size, seed_c1,
                    type, fluorescence, t);

                cell c2 = create_cell(d_params, size, seed_c2,
                    type, fluorescence, t);
                
                c1.state = ALIVE;
                c2.state = ALIVE;

                cell_tree_levels[next_depth][next_id] = c1;
                cell_tree_levels[next_depth][next_id + 1] = c2;

                proliferation = true;
            }

            if (threadIdx.x == 0)
            {
                __syncthreads();

                uint64_t next_offset = id * 2;

                
                if (((current_depth + 1) == depth) || proliferation)
                {
                    proliferate<<<2, blockDim.x>>>(d_params, size,
                        original_size * 2,
                        cell_tree_levels,
                        fluorescence_threshold, t_max, seed,
                        depth, next_depth,
                        next_offset);
                }
                else
                {
                    uint64_t last_level_size = pow(2, depth);
                    apply_bounding<<<2, blockDim.x>>>(last_level_size,
                            cell_tree_levels,
                            depth,
                            next_depth,
                            next_offset);
                }
            }
        }
    }

}

__global__
void
apply_bounding(uint64_t original_size,
                cell** cell_tree_levels,
                uint64_t depth,
                uint64_t current_depth,
                uint64_t offset)
{
    uint64_t id = offset + threadIdx.x + blockIdx.x * blockDim.x;

    if (id < original_size)
    {
        cell current = cell_tree_levels[current_depth][id];
        if (current.state == INACTIVE)
        {
            uint64_t final_index = id * pow(2, depth - current_depth);
            cell_tree_levels[depth][final_index] = current;
        }
    }
}

__device__
bool
cell_will_divide(cell& c, double_t fluorescence_threshold, double_t t_max)
{
    return (c.timer > 0.0) && 
        (c.t + c.timer < t_max) &&
        (c.fluorescence / 2 > fluorescence_threshold);
}
    
} // End device namespace
    
} // End simulation namespace
    
} // End procell namespace

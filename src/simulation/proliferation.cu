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
#define MAX_TREE_DEPTH (23)

#define REMOVE 0
#define ALIVE 1
#define INACTIVE 2

namespace procell { namespace simulation
{

__host__
bool
proliferate(simulation::cell_types& h_params,
            uint64_t size,
            uint64_t tree_depth,
            cell* h_cells, double_t t_max, double_t threshold,
            fluorescences& m_results)
{
    device::cell_types d_params = h_params;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0 /* TODO check devices number */);
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_SYNC_DEPTH);

    cell* h_active_cells = h_cells;
    cell* d_current_stage = NULL;
    fluorescence* d_results = NULL;
    uint64_t new_size = size;
    cudaMalloc((void**) &d_current_stage, new_size * sizeof(cell));
    cudaMalloc((void**) &d_results, m_results.size() * sizeof(fluorescence));
    cudaMemcpy(d_current_stage, h_active_cells, new_size * sizeof(cell),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_results,
        thrust::raw_pointer_cast(m_results.data()),
        m_results.size() * sizeof(fluorescence), cudaMemcpyHostToDevice);

    bool* still_alive = (bool*) malloc(sizeof(bool));
    *still_alive = true;
    uint64_t divisions = 0;
    while (*still_alive)
    {
        *still_alive = false;
        uint64_t depth = utils::max_recursion_depth(m_results.size(), new_size);

        // Check if GPU has enough memory to compute next stage
        if (depth == 0)
        {
            std::cout << "--- ERROR: out of GPU memory" << std::endl;
            std::cout << "--- Total iterations: " << divisions << std::endl;
            std::cout << "--- Copying partial results to file...";
            std::cout << "copied, aborting." << std::endl;
            free(still_alive);
            return false;
        }

        depth = min(depth, tree_depth);

        if (tree_depth == MAX_TREE_DEPTH + 1)
        {
            simulation::cell_type choosen_proportion = h_params.data()[0];
            for (uint64_t i = 1; i < h_params.size(); i++)
            {
                if (choosen_proportion.probability > h_params.data()[i].probability)
                    choosen_proportion = h_params.data()[i];
            }
    
            uint64_t proposed_depth = 0;
            uint64_t partial_sum = 0;
            while (partial_sum < t_max)
            {
                proposed_depth++;
                partial_sum += choosen_proportion.timer;
            }
    
            depth = min(depth, proposed_depth);
        }
        
        run_iteration(d_params,
            t_max,
            threshold,
            prop.maxThreadsPerBlock,
            &d_current_stage,
            new_size,
            d_results,
            m_results.size(),
            still_alive,
            depth);

        new_size = new_size * pow(2, depth);

        divisions++;
    }
    free(still_alive);
    cudaMemcpy(thrust::raw_pointer_cast(m_results.data()),
        d_results, m_results.size() * sizeof(fluorescence),
        cudaMemcpyDeviceToHost);

    return true;
}

__host__
void
run_iteration(device::cell_types& d_params, double_t t_max, double_t threshold,
    uint32_t max_threads_per_block, cell** d_current_stage,
    uint64_t& current_size,
    fluorescence* d_results, uint64_t d_results_size,
    bool* still_alive,
    uint64_t depth)
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

    bool* d_still_alive = NULL;
    cudaMalloc((void**) &d_still_alive, sizeof(bool));
    cudaMemcpy(d_still_alive, still_alive, sizeof(bool),
        cudaMemcpyHostToDevice);

    device::proliferate<<<n_blocks, max_threads_per_block>>>
        (thrust::raw_pointer_cast(d_params.data()), d_params.size(),
        current_size,
        original_size,
        thrust::raw_pointer_cast(d_tree_levels.data()),
        d_results,
        d_results_size,
        d_still_alive,
        threshold,
        t_max,
        random_seed,
        depth,
        0,
        0);

    cudaDeviceSynchronize();

    cudaMemcpy(still_alive, d_still_alive, sizeof(bool),
        cudaMemcpyDeviceToHost);

    for (uint8_t i = 0; i < depth; i++)
    {
        cudaFree(h_tree_levels[i]);
    }

    *d_current_stage = h_tree_levels[depth];
}

namespace device
{
    
__global__
void
proliferate(cell_type* d_params, uint64_t size,
            uint64_t starting_size,
            uint64_t original_size,
            cell** cell_tree_levels,
            fluorescence* d_results,
            uint64_t d_results_size,
            bool* still_alive,
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
            if (current.state == ALIVE)
            {
                if (!out_of_time(current, t_max))
                {
                    if (current.fluorescence / 2 > fluorescence_threshold)
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

                        if (current_depth == (depth - 1))
                            *still_alive = true;
                    }
                }
                else
                {
                    uint64_t l = 0;
                    uint64_t r = d_results_size - 1;

                    while (l <= r)
                    {
                        uint64_t m = (l + r) / 2;
                        if (d_results[m].value == current.fluorescence)
                        {
                            atomicAdd((int*) &d_results[m].frequency, 1);
                            l = r + 1;
                            
                        }
                        else if (d_results[m].value > current.fluorescence)
                        {
                            r = m - 1;
                        }
                        else
                        {
                            l = m + 1;
                        }
                    }

                    /*
                    uint64_t last_index = id * pow(2, depth - current_depth);
                    
                    current.state = INACTIVE;
                    cell_tree_levels[depth][last_index] = current;
                    */
                }
            }
            
            if (threadIdx.x == 0)
            {
                __syncthreads();

                uint64_t next_offset = id * 2;
                if (proliferation)
                {
                    proliferate<<<2, blockDim.x>>>(d_params, size,
                        starting_size,
                        original_size * 2,
                        cell_tree_levels,
                        d_results, d_results_size,
                        still_alive,
                        fluorescence_threshold, t_max, seed,
                        depth, next_depth,
                        next_offset);
                }
            }
        }
    }

}

__device__
bool
out_of_time(cell& c, double_t t_max)
{
    return (c.timer < 0.0) || 
        (c.t + c.timer > t_max);
}
    
} // End device namespace
    
} // End simulation namespace
    
} // End procell namespace

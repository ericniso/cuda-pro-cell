#include <iostream>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <stack>
#include <utility>
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
            fluorescences_result& m_results,
            bool track_ratio)
{
    device::cell_types d_params = h_params;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0 /* TODO check devices number */);
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_SYNC_DEPTH);

    cell* h_active_cells = h_cells;
    cell* d_current_stage = NULL;
    fluorescence_with_ratio* d_results = NULL;
    uint64_t new_size = size;

    if (track_ratio)
    {
        for (int32_t i = 0; i < m_results.size(); i++)
        {
            m_results.data()[i].ratio = NULL;
            cudaMalloc((void**) &m_results.data()[i].ratio, h_params.size() 
                * sizeof(int32_t));
            int32_t* tmp_ratio_arr = 
                (int32_t*) malloc(h_params.size() * sizeof(int32_t));
    
            for (int32_t j = 0; j < h_params.size(); j++)
            {
                tmp_ratio_arr[j] = 0;
            }
            
            cudaMemcpy(m_results.data()[i].ratio, tmp_ratio_arr, 
                h_params.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    
            free(tmp_ratio_arr);
        }
    }

    cudaMalloc((void**) &d_results, m_results.size() 
        * sizeof(fluorescence_with_ratio));
    cudaMemcpy(d_results,
        thrust::raw_pointer_cast(m_results.data()),
        m_results.size() * sizeof(fluorescence_with_ratio), 
        cudaMemcpyHostToDevice);

    std::stack< std::pair<uint64_t, cell*> > populations_stack;
    populations_stack.push(std::make_pair(new_size, h_active_cells));

    bool* still_alive = (bool*) malloc(sizeof(bool));
    *still_alive = false;
    uint64_t divisions = 0;
    while (!populations_stack.empty() || *still_alive)
    {
        if (!(*still_alive))
        {
            std::pair<uint64_t, cell*> population = populations_stack.top();
            populations_stack.pop();

            new_size = population.first;
            cudaMalloc((void**) &d_current_stage, new_size * sizeof(cell));
            cudaMemcpy(d_current_stage, population.second,
                new_size * sizeof(cell), cudaMemcpyHostToDevice);
            free(population.second);
        }

        *still_alive = false;
        uint64_t depth = utils::max_recursion_depth(m_results.size(), new_size);
        uint64_t new_size_multiplier = 2;
        uint64_t proposed_new_size = new_size / new_size_multiplier;

        // Check for available memory
        while (depth == 0)
        {
            depth = utils::max_recursion_depth(m_results.size(), 
                proposed_new_size);

            if (depth > 0)
            {
                uint64_t current_size = proposed_new_size;
                uint64_t blocks_needed = 0;

                for (int i = 1; i <= depth; i++)
                {
                    blocks_needed += round(0.5 + current_size / prop.maxThreadsPerBlock);
                    current_size = current_size * 2;
                }

                if (blocks_needed >= prop.maxGridSize[0])
                {
                    depth = 0;
                }
            }
            
            if (depth > 0)
            {
                // Split cell population
                cell* h_partial_pop = 
                    (cell*) malloc(sizeof(cell)
                        * (new_size - proposed_new_size));
                
                cell* d_partial_pop = NULL;
                cudaMalloc((void**) &d_partial_pop, 
                    proposed_new_size * sizeof(cell));
                cudaMemcpy(d_partial_pop, d_current_stage,
                    proposed_new_size 
                    * sizeof(cell), cudaMemcpyDeviceToDevice);
                
                cudaMemcpy(h_partial_pop, &(d_current_stage[proposed_new_size]),
                    (new_size - proposed_new_size)
                    * sizeof(cell), cudaMemcpyDeviceToHost);

                populations_stack.push(
                    std::make_pair((new_size - proposed_new_size), 
                        h_partial_pop));
                cudaFree(d_current_stage);
                d_current_stage = d_partial_pop;

                new_size = proposed_new_size;
            }

            new_size_multiplier += 1;
            proposed_new_size = new_size / new_size_multiplier;
        }

        depth = min(depth, tree_depth);

        if (tree_depth == MAX_TREE_DEPTH + 1)
        {
            cell_type choosen_proportion = h_params.data()[0];
            for (uint64_t i = 1; i < h_params.size(); i++)
            {
                if (choosen_proportion.proportion > h_params.data()[i].proportion)
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
            depth,
            track_ratio);

        new_size = new_size * pow(2, depth);

        if (!still_alive)
        {
            cudaFree(d_current_stage);
        }

        divisions++;
    }
    free(still_alive);
    cudaMemcpy(thrust::raw_pointer_cast(m_results.data()),
        d_results, m_results.size() * sizeof(fluorescence_with_ratio),
        cudaMemcpyDeviceToHost);

    if (track_ratio)
    {
        for (uint64_t i = 0; i < m_results.size(); i++)
        {
            int32_t* tmp_ratio_arr = 
                (int32_t*) malloc(h_params.size() * sizeof(int32_t));
            cudaMemcpy(tmp_ratio_arr, m_results.data()[i].ratio, 
                h_params.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
            m_results.data()[i].ratio = tmp_ratio_arr;
        }
    }

    return true;
}

__host__
void
run_iteration(device::cell_types& d_params, double_t t_max, double_t threshold,
    uint32_t max_threads_per_block, cell** d_current_stage,
    uint64_t& current_size,
    fluorescence_with_ratio* d_results, uint64_t d_results_size,
    bool* still_alive,
    uint64_t depth,
    bool track_ratio)
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

    bool* d_track_ratio = NULL;
    cudaMalloc((void**) &d_track_ratio, sizeof(bool));
    cudaMemcpy(d_track_ratio, &track_ratio, sizeof(bool),
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
        0,
        d_track_ratio);

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
            fluorescence_with_ratio* d_results,
            uint64_t d_results_size,
            bool* still_alive,
            double_t fluorescence_threshold,
            double_t t_max,
            uint64_t seed,
            uint64_t depth,
            uint64_t current_depth,
            uint64_t offset,
            bool* track_ratio)
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
                    int64_t l = 0;
                    int64_t r = d_results_size - 1;

                    while (l >= 0 && r >= 0 && l <= r)
                    {
                        uint64_t m = l + (r - l) / 2;
                        if (d_results[m].value == current.fluorescence)
                        {
                            atomicAdd((int*) &d_results[m].frequency, 1);

                            if (*track_ratio)
                            {
                                atomicAdd((int*) &d_results[m].ratio
                                    [current.type], 1);
                            }

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
                        next_offset, track_ratio);
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

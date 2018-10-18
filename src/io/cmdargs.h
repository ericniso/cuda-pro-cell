#ifndef PROCELL_CMDARGS_H
#define PROCELL_CMDARGS_H

#define MAX_TREE_DEPTH (23)

#include <string>
#include <inttypes.h>
#include <math.h>

namespace procell { namespace io
{

class CmdArgs
{

public:
    CmdArgs(int argc, char** argv);

    bool h0_given = false;
    std::string h0; // required
    bool cell_types_given = false;
    std::string cell_types; // required
    bool output_histogram_given = false;
    std::string output_histogram; // optional
    bool t_max_given = false;
    uint64_t t_max; // required
    bool phi_min_given = false;
    double_t phi_min; // required
    bool tree_depth_given = false;
    uint32_t tree_depth = MAX_TREE_DEPTH; // optional
    bool track_ratio = false; // optional

private:
    bool
    check_h0(int& argc, char** argv, uint32_t& i);
    bool
    check_cell_types(int& argc, char** argv, uint32_t& i);
    bool
    check_output(int& argc, char** argv, uint32_t& i);
    bool
    check_t_max(int& argc, char** argv, uint32_t& i);
    bool
    check_phi_min(int& argc, char** argv, uint32_t& i);
    bool
    check_tree_depth(int& argc, char** argv, uint32_t& i);
    bool
    check_track_ratio(int& argc, char** argv, uint32_t& i);
};

} // End io namespace

} // End procell namespace

#endif // PROCELL_CMDARGS_H

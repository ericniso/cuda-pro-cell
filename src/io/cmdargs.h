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

    bool h0_given;
    std::string h0; // required
    bool cell_types_given;
    std::string cell_types; // required
    bool output_histogram_given;
    std::string output_histogram; // optional
    bool t_max_given;
    double_t t_max; // required
    bool phi_min_given;
    double_t phi_min; // required
    bool tree_depth_given;
    uint32_t tree_depth; // optional
    bool track_ratio; // optional

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

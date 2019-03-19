#include <iostream> 
#include <stdlib.h>
#include "io/cmdargs.h"

namespace procell { namespace io
{

CmdArgs::CmdArgs(int argc, char** argv)
{
    for (uint32_t i = 1; i < argc; i++)
    {
        std::string str(argv[i]);

        bool legal = this->check_h0(argc, argv, i);
        legal = legal || this->check_cell_types(argc, argv, i);
        legal = legal || this->check_output(argc, argv, i);
        legal = legal || this->check_t_max(argc, argv, i);
        legal = legal || this->check_phi_min(argc, argv, i);
        legal = legal || this->check_tree_depth(argc, argv, i);
        legal = legal || this->check_track_ratio(argc, argv, i);

        if (str == "--help")
        {
            legal = true;
            std::cout << "Questo è l'help" << std::endl;
            exit(EXIT_SUCCESS);
        }

        if (!legal)
        {
            std::cout << "Invalid option " << str << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    if (!this->h0_given || !this->cell_types_given 
        || !this->t_max_given || !this->phi_min_given)
    {
        std::cout << "The following missing arguments are required:" 
            << std::endl;
        
        if (!this->h0_given)
        {
            std::cout << "--histogram (-h)" << std::endl;
        }

        if (!this->cell_types_given)
        {
            std::cout << "--cell-types (-c)" << std::endl;
        }   

        if (!this->t_max_given)
        {
            std::cout << "--t-max (-t)" << std::endl;
        }

        if (!this->phi_min_given)
        {
            std::cout << "--phi-min (-p)" << std::endl;
        }

        exit(EXIT_FAILURE);
    }
};

bool
CmdArgs::check_h0(int& argc, char** argv, uint32_t& i)
{
    bool res = false;
    std::string str(argv[i]);
    
    if (str == "--histogram" || str == "-h")
    {
        res = true;
        if (!this->h0_given)
        {
            if (i < argc - 1)
            {
                i++;
                this->h0_given = true;
                this->h0 = std::string(argv[i]);
            }
            else
            {
                std::cout << "Option --histogram (-h) requires a filename" 
                    << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            std::cout << "Option --histogram (-h) already given" 
                << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return res;
};

bool
CmdArgs::check_cell_types(int& argc, char** argv, uint32_t& i)
{
    bool res = false;
    std::string str(argv[i]);
    
    if (str == "--cell-types" || str == "-c")
    {
        res = true;
        if (!this->cell_types_given)
        {
            if (i < argc - 1)
            {
                i++;
                this->cell_types_given = true;
                this->cell_types = std::string(argv[i]);
            }
            else
            {
                std::cout << "Option --cell-types (-c) requires a filename" 
                    << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            std::cout << "Option --cell-types (-c) already given" 
                << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return res;
};

bool
CmdArgs::check_output(int& argc, char** argv, uint32_t& i)
{
    bool res = false;
    std::string str(argv[i]);
    
    if (str == "--output-histogram" || str == "-o")
    {
        res = true;
        if (!this->output_histogram_given)
        {
            if (i < argc - 1)
            {
                i++;
                this->output_histogram_given = true;
                this->output_histogram = std::string(argv[i]);
            }
            else
            {
                std::cout << "Option --output-histogram (-o) requires a filename" 
                    << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            std::cout << "Option --output-histogram (-o) already given" 
                << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return res;
};

bool
CmdArgs::check_t_max(int& argc, char** argv, uint32_t& i)
{
    bool res = false;
    std::string str(argv[i]);
    
    if (str == "--time-max" || str == "-t")
    {
        res = true;
        if (!this->t_max_given)
        {
            if (i < argc - 1)
            {
                i++;
                try
                {
                    std::string val(argv[i]);
                    double_t converted_val = std::stod(val);

                    if (converted_val >= 0)
                    {
                        this->t_max_given = true;
                        this->t_max = converted_val;
                    }
                    else
                    {
                        std::cout << "Option --time-max (-t) requires an integer" 
                            << " value >= 0" << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                catch(std::exception const& e)
                {
                    std::cout << "Invalid --time-max (-t) value" << std::endl;
                    exit(EXIT_FAILURE);
                }
                
            }
            else
            {
                std::cout << "Option --time-max (-t) requires an integer" 
                    << " value >= 0" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            std::cout << "Option --time-max (-t) already given" 
                << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return res;
};

bool
CmdArgs::check_phi_min(int& argc, char** argv, uint32_t& i)
{
    bool res = false;
    std::string str(argv[i]);
    
    if (str == "--phi-min" || str == "-p")
    {
        res = true;
        if (!this->phi_min_given)
        {
            if (i < argc - 1)
            {
                i++;
                try
                {
                    std::string val(argv[i]);
                    double_t converted_val = std::stod(val);

                    if (converted_val > 0)
                    {
                        this->phi_min_given = true;
                        this->phi_min = converted_val;
                    }
                    else
                    {
                        std::cout << "Option --phi-min (-p) requires a double" 
                            << " value > 0" << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                catch(std::exception const& e)
                {
                    std::cout << "Invalid --phi-min (-p) value" << std::endl;
                    exit(EXIT_FAILURE);
                }
                
            }
            else
            {
                std::cout << "Option --phi-min (-p) requires a double" 
                    << " value > 0" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            std::cout << "Option --phi-min (-p) already given" 
                << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return res;
};

bool
CmdArgs::check_tree_depth(int& argc, char** argv, uint32_t& i)
{
    bool res = false;
    std::string str(argv[i]);
    
    if (str == "--tree-depth" || str == "-d")
    {
        res = true;
        if (!this->tree_depth_given)
        {
            if (i < argc - 1)
            {
                i++;
                try
                {
                    std::string val(argv[i]);
                    uint32_t converted_val = std::stoi(val);

                    if (converted_val >= 1 && converted_val <= MAX_TREE_DEPTH)
                    {
                        this->tree_depth_given = true;
                        this->tree_depth = converted_val;
                    }
                    else
                    {
                        std::cout << "Option --tree-depth (-d) requires an integer" 
                            << " value >= 1 && <= " << MAX_TREE_DEPTH 
                            << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                catch(std::exception const& e)
                {
                    std::cout << "Invalid --tree-depth (-d) value" << std::endl;
                    exit(EXIT_FAILURE);
                }
                
            }
            else
            {
                std::cout << "Option --tree-depth (-d) requires an integer" 
                    << " value >= 1" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            std::cout << "Option --tree-depth (-d) already given" 
                << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return res;
};

bool
CmdArgs::check_track_ratio(int& argc, char** argv, uint32_t& i)
{
    bool res = false;
    std::string str(argv[i]);
    
    if (str == "--track-ratio" || str == "-r")
    {
        res = true;
        if (!this->track_ratio)
        {
            this->track_ratio = true;
        }
        else
        {
            std::cout << "Option --track-ratio (-r) already given" 
                << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return res;
};

} // End io namespace

} // End procell namespace

#ifndef _SYCL_UTILS_HPP_
#define _SYCL_UTILS_HPP_
#include <iostream>
#include <CL/sycl.hpp>

namespace sycl
{
    std::ostream &operator<< (std::ostream &os, sycl::device const &device)
    {
        os << "Running on: "
           << device.get_info<sycl::info::device::name>() << "\n"
           << "Max work items: " 
           << device.get_info<sycl::info::device::max_work_group_size>() << "\n"
           << "Max compute units: "
           << device.get_info<sycl::info::device::max_compute_units>() << "\n"


        //    << "Max work items sizes<1>: " 
        //    << device.get_info<sycl::info::device::max_work_item_sizes<1>>() << "\n"
        //    << "Max work items sizes<2>: " 
        //    << device.get_info<sycl::info::device::max_work_item_sizes<2>>() << "\n"
        //    << "Max work items sizes<3>: " 
        //    << device.get_info<sycl::info::device::max_work_item_sizes<3>>() << "\n"
           << "Max # sub groups: " 
           << device.get_info<sycl::info::device::max_num_sub_groups>() << "\n";

        return os;
    }   
}
/*
    Quick function to determine the correctness of a basic vector add
    --> assumes 1 + 0 for each element
*/
bool check_vector_add(int *R, size_t VecSize, int NElements = 3)
{
    bool IsCorrect = true;
    for (size_t i = 0; i < VecSize; i++)
    {
        if (i < NElements || i > VecSize - NElements - 1)
        {
            std::cout << R[i] << " ";
        }
        if (R[i] != 1) IsCorrect = false;
    }

    std::cout << std::endl;
    return IsCorrect;
}
#endif
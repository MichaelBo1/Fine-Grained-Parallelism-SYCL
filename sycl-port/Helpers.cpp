#include <iostream>
#include <CL/sycl.hpp>

void printDevInfo(const sycl::queue &Q)
{
    std::cout << "Running on "
              << Q.get_device().get_info<sycl::info::device::name>()
              << "\n";

    std::size_t maxWorkItems = Q.get_device().get_info<sycl::info::device::max_work_group_size>();
    std::cout << "Max Work Items: " << maxWorkItems << "\n";
}


    
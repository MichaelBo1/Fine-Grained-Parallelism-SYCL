#include <CL/sycl.hpp>
#include "../sycl_utils.hpp"
#include "../ArrayQueue.cpp"
#include "va_profiler.cpp"

template<std::size_t WorkGroupSize>
void profile_local_queue_va(sycl::queue &Q, const std::size_t VecSize, std::vector<VectorEventProfile> &EventProfiles, int ProfilingIters)
{
    std::cout << " ========== PROFILING ========== " << "\n"
    << "Vector Size: " << VecSize << "\n";

    int *A = sycl::malloc_shared<int>(VecSize, Q);
    int *B = sycl::malloc_shared<int>(VecSize, Q);
    int *R = sycl::malloc_shared<int>(VecSize, Q);

    for (int i = 0; i < VecSize; i++)
    {
        A[i] = 1;
        B[i] = 0;
    }

    for (int i = 0; i < ProfilingIters; i++)
    {

    }
        

    bool CorrectAdd = check_vector_add(A, B, R, VecSize);
    std::cout << "Correct? " << std::boolalpha << CorrectAdd << std::endl;

    sycl::free(TaskQueues, Q);
    sycl::free(A, Q);
    sycl::free(B, Q);
    sycl::free(R, Q);

    std::cout << " ========== END PROFILING ========== " << "\n";

}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: <exec> <profiling iterations> <write to file (1|0)>" << std::endl;
        return 1;
    }
    int ProfilingIters = std::stoi(argv[1]);
    int WriteToFile = std::stoi(argv[2]);


    sycl::default_selector device_selector;
    sycl::property_list props{sycl::property::queue::enable_profiling()}; // For measuring device execution times

    sycl::queue Q(device_selector, props);
    sycl::device Device = Q.get_device();

    std::cout << "\nBegin Device Information ====================================" << "\n";
    std::cout << Device;
    std::cout << "End Device Information   ====================================" << "\n";

    constexpr std::size_t WorkGroupSize = 1024;

    auto MaxGroupSize = Device.get_info<sycl::info::device::max_work_group_size>();
    if (WorkGroupSize > MaxGroupSize)
    {
        std::cerr << "Work Group Size cannot exceed " << MaxGroupSize << " on this device!" << "\n";
        return 1;
    }

    // Checks for local-mem support
    auto HasLocalMem = Device.is_host() || Device.get_info<sycl::info::device::local_mem_type>() != sycl::info::local_mem_type::none;
    auto LocalMemSize = Device.get_info<sycl::info::device::local_mem_size>();
    if (!HasLocalMem || LocalMemSize < (WorkGroupSize * sizeof(int)))
    {
        std::cerr << "Device does not have enough local memory!" << "\n";
        return 1;
    }

    // ------------------------
    // PROFILING
    // ------------------------
    
    std::vector<VectorEventProfile> EventProfiles;

    // for (int i = 10; i < 30; i++)
    // {
    //     std::size_t VecSize = std::pow(2, i);
    //     profile_multi_queue_va<WorkGroupSize>(Q, VecSize, EventProfiles, ProfilingIters);
    // }

    if (WriteToFile)
    {
        std::ofstream OutFile("profiling-results/profiling_local_queue.csv", std::ios::app);
    
        bool WriteHeaders = OutFile.tellp() == 0;
        if (WriteHeaders) {
            OutFile << "Event,MeanCGSubmissionTime(ms),MeanKernelExecTime(ms),MeanTotalExecTime(ms),VecSize,WorkGroupSize\n";
        }
        
        for (const auto &profile : EventProfiles)
        {
            OutFile << profile.name << "," << profile.profileData.cgSubmissionTime << "," 
            << profile.profileData.kernelExecTime << "," 
            << profile.profileData.totalExecTime << ","
            << profile.vecSize << ","
            << WorkGroupSize << "\n";
        }

        OutFile.close();
    }

    return 0;
}
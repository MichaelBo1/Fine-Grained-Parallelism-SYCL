#include "../../sycl_utils.hpp"
#include "../va_profiler.cpp"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <Vector Size>" << "\n";
        return 1;
    }

    std::size_t VecSize = std::atoi(argv[1]);
    if (VecSize <= 0) {
        std::cerr << "Invalid vector size: " << argv[1] << std::endl;
        return 1;
    }

    sycl::default_selector device_selector;
    sycl::property_list props{sycl::property::queue::enable_profiling()}; // For measuring device execution times

    sycl::queue Q(device_selector, props);
    sycl::device Device = Q.get_device();

    auto DevName = Device.get_info<sycl::info::device::name>();
    // ------------------------
    // PROFILING
    // ------------------------

    auto StartTimePoint = std::chrono::high_resolution_clock::now();

    int *A = sycl::malloc_device<int>(VecSize, Q);
    int *B = sycl::malloc_device<int>(VecSize, Q);
    int *R = sycl::malloc_device<int>(VecSize, Q);
    Q.parallel_for(VecSize, [=](sycl::id<1> idx) {
        A[idx] = 1;
        B[idx] = 0;
        R[idx] = 0;
    });
    Q.wait();

    auto MemorySetupTimePoint = std::chrono::high_resolution_clock::now();

    sycl::event AddEvent = Q.submit([&](sycl::handler &h)
    {
        h.parallel_for(VecSize, [=](sycl::id<1> idx)
        {
            R[idx] = A[idx] + B[idx];
        });
    });
    Q.wait();

    auto EndTimePoint = std::chrono::high_resolution_clock::now();

    // bring mem over to check correctness
    // int hostR[VecSize];
    // Q.memcpy(hostR, R, VecSize * sizeof(int));
    // Q.wait();

    // std::cout << "Checking correctness:" << "\n";
    // bool CorrectAdd = check_vector_add(hostR, VecSize);
    // std::cout << "Vector Addition for size: " << VecSize << "\n";
    // std::cout << "Correct? " << std::boolalpha << CorrectAdd << std::endl;


    durationMiliSecs ExecTime = EndTimePoint - StartTimePoint;
    durationMiliSecs MemTime = MemorySetupTimePoint - StartTimePoint;

    auto StartKernelExecTimePoint = AddEvent.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto EndKernelExecTimePoint = AddEvent.get_profiling_info<sycl::info::event_profiling::command_end>();
    double KernelProfileTime = to_mili(EndKernelExecTimePoint - StartKernelExecTimePoint);

    std::cout << "Total Exec Time" << "," << ExecTime.count() << "," << VecSize << "\n";
    std::cout << "Memory Setup Time" << "," << MemTime.count() << "," << VecSize << "\n";
    std::cout << "Kernel Exec Time" << "," << KernelProfileTime << "," << VecSize << "\n";

    sycl::free(A, Q);
    sycl::free(B, Q);
    sycl::free(R, Q);
    
    return 0;
}

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
    std::vector<TimingEvent> Events;

    auto StartTimePoint = std::chrono::high_resolution_clock::now();

    int *A = sycl::malloc_shared<int>(VecSize, Q);
    int *B = sycl::malloc_shared<int>(VecSize, Q);
    int *R = sycl::malloc_shared<int>(VecSize, Q);

    for (int i = 0; i < VecSize; i++)
    {
        A[i] = 1;
        B[i] = 0;
    }

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

    durationMiliSecs ExecTime = EndTimePoint - StartTimePoint;
    durationMiliSecs MemTime = MemorySetupTimePoint - StartTimePoint;

    auto StartKernelExecTimePoint = AddEvent.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto EndKernelExecTimePoint = AddEvent.get_profiling_info<sycl::info::event_profiling::command_end>();
    double KernelProfileTime = to_mili(EndKernelExecTimePoint - StartKernelExecTimePoint);

    Events.push_back({"Total Exec Time", VecSize, ExecTime.count()});
    Events.push_back({"Memory Setup Time", VecSize, MemTime.count()});
    Events.push_back({"Kernel Exec Time", VecSize, KernelProfileTime});

    // bool CorrectAdd = check_vector_add(R, VecSize);
    // std::cout << "Vector Addition for size: " << VecSize << "\n";
    // std::cout << "Correct? " << std::boolalpha << CorrectAdd << std::endl;

    sycl::free(A, Q);
    sycl::free(B, Q);
    sycl::free(R, Q);
    

    for (const TimingEvent &event : Events)
    {
        std::cout << event.Name << "," << event.ExecTime << "," << event.VectorSize << "," << DevName << "\n";
    }

    return 0;
}

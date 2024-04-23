#include <CL/sycl.hpp>
#include "../../sycl_utils.hpp"
#include "../../tasking/ArrayQueue.cpp"
#include "../va_profiler.cpp"

template<std::size_t WorkGroupSize>
void sk_multi_queue_add(sycl::queue &Q, const std::size_t VecSize)
{
    const size_t NumWorkGroups = VecSize / WorkGroupSize;
    if (NumWorkGroups % 2 != 0 && NumWorkGroups != 1)
    {
        std::cerr << "Number of work groups should divide Vector Size evenly!" << "\n";
        return;
    }

    auto StartTimePoint = std::chrono::high_resolution_clock::now();

    auto TaskQueues = sycl::malloc_device<SPMCArrayQueue<int, WorkGroupSize>>(NumWorkGroups, Q);
    // for (size_t i = 0; i < NumWorkGroups; i++)
    // {
    //     new (TaskQueues + i) SPMCArrayQueue<int, WorkGroupSize>();
    // }

    int *A = sycl::malloc_device<int>(VecSize, Q);
    int *B = sycl::malloc_device<int>(VecSize, Q);
    int *R = sycl::malloc_device<int>(VecSize, Q);

    Q.parallel_for(VecSize, [=](sycl::id<1> idx) {
        A[idx] = 1;
        B[idx] = 0;
        R[idx] = 0;
    });
    Q.wait();
    Q.single_task([=]()
        {
            for (size_t i = 0; i < NumWorkGroups; i++)
            {
                new (TaskQueues + i) SPMCArrayQueue<int, WorkGroupSize>();
            }
        }
    );
    Q.wait();

    auto MemorySetupTimePoint = std::chrono::high_resolution_clock::now();

    sycl::event AddEvent = Q.submit([&](sycl::handler &h)
    {
        h.parallel_for(sycl::nd_range<1>{sycl::range<1>{VecSize}, sycl::range<1>{WorkGroupSize}}, [=](sycl::nd_item<1> Item)
        {
            int QueueIdx = Item.get_global_id() / WorkGroupSize;
            auto &TargetQueue = TaskQueues[QueueIdx];
            sycl::group Group = Item.get_group();

            if (Item.get_local_id() == 0)
            {
                for (int i = 0; i < WorkGroupSize; i++)
                {
                    int ItemVal = i + QueueIdx * WorkGroupSize;
                    TargetQueue.push(ItemVal);
                }
            }

            sycl::group_barrier(Group);

            int ItemVal = TargetQueue.front(Item.get_local_id());
            R[ItemVal] = A[ItemVal] + B[ItemVal];

            sycl::group_barrier(Group);

            if (Item.get_local_id() == 0)
            {
                while (!TargetQueue.empty())
                {
                    TargetQueue.pop();
                }
            }
        });
    });
    Q.wait();

    auto EndTimePoint = std::chrono::high_resolution_clock::now(); 

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

    std::cout << "Total Exec Time" << "," << ExecTime.count() << "," << VecSize << "," << WorkGroupSize << "\n";
    std::cout << "Memory Setup Time" << "," << MemTime.count() << "," << VecSize << "," << WorkGroupSize << "\n";
    std::cout << "Kernel Exec Time" << "," << KernelProfileTime << "," << VecSize << "," << WorkGroupSize << "\n";

    // bool CorrectAdd = check_vector_add(R, VecSize);
    // std::cout << "Vector Addition for size: " << VecSize << "\n";
    // std::cout << "Correct? " << std::boolalpha << CorrectAdd << std::endl;

    sycl::free(TaskQueues, Q);
    sycl::free(A, Q);
    sycl::free(B, Q);
    sycl::free(R, Q);
}

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

    constexpr std::size_t WorkGroupSize = 32;

    auto MaxGroupSize = Device.get_info<sycl::info::device::max_work_group_size>();
    if (WorkGroupSize > MaxGroupSize)
    {
        std::cerr << "Work Group Size cannot exceed " << MaxGroupSize << " on this device!" << "\n";
        return 1;
    }

    
    sk_multi_queue_add<WorkGroupSize>(Q, VecSize);

    return 0;
}
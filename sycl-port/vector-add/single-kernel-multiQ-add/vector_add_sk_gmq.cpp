#include <CL/sycl.hpp>
#include "../../sycl_utils.hpp"
#include "../../ArrayQueue.cpp"
#include "../va_profiler.cpp"

template<std::size_t WorkGroupSize>
void sk_multi_queue_add(sycl::queue &Q, const std::size_t VecSize, std::vector<TimingEvent> &Events)
{
    const size_t NumWorkGroups = VecSize / WorkGroupSize;
    if (NumWorkGroups % 2 != 0 && NumWorkGroups != 1)
    {
        std::cerr << "Number of work groups should divide Vector Size evenly!" << "\n";
        return;
    }

    auto StartTimePoint = std::chrono::high_resolution_clock::now();

    auto TaskQueues = sycl::malloc_shared<SPMCArrayQueue<int, WorkGroupSize>>(NumWorkGroups, Q);
    for (size_t i = 0; i < NumWorkGroups; i++)
    {
        new (TaskQueues + i) SPMCArrayQueue<int, WorkGroupSize>();
    }

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

    sycl::free(TaskQueues, Q);
    sycl::free(A, Q);
    sycl::free(B, Q);
    sycl::free(R, Q);
}

int main(int argc, char **argv)
{
    sycl::default_selector device_selector;
    sycl::property_list props{sycl::property::queue::enable_profiling()}; // For measuring device execution times

    sycl::queue Q(device_selector, props);
    sycl::device Device = Q.get_device();

    auto DevName = Device.get_info<sycl::info::device::name>();

    constexpr std::size_t WorkGroupSize = 1024;

    auto MaxGroupSize = Device.get_info<sycl::info::device::max_work_group_size>();
    if (WorkGroupSize > MaxGroupSize)
    {
        std::cerr << "Work Group Size cannot exceed " << MaxGroupSize << " on this device!" << "\n";
        return 1;
    }

    // ------------------------
    // PROFILING
    // ------------------------ 
    
    std::vector<TimingEvent> Events;

    for (int i = 10; i < 30; i++)
    {
        std::size_t VecSize = std::pow(2, i);
        sk_multi_queue_add<WorkGroupSize>(Q, VecSize, Events);
    }

    for (const TimingEvent &event : Events)
    {
        std::cout << event.Name << "," << event.ExecTime << "," << event.VectorSize << "," << WorkGroupSize << "," << DevName << "\n";
    }

    return 0;
}
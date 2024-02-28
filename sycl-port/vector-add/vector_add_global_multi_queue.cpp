#include <CL/sycl.hpp>
#include "../sycl_utils.hpp"
#include "../ArrayQueue.cpp"
#include "va_profiler.cpp"

template<std::size_t WorkGroupSize>
void profile_multi_queue_va(sycl::queue &Q, const std::size_t VecSize, std::vector<VectorEventProfile> &EventProfiles, int ProfilingIters)
{
    eventList EnqueueEvents;
    eventList AddEvents;
    eventList ShutdownEvents;
    std::vector<double> TotalExecutionTimes;

    const size_t NumWorkGroups = VecSize / WorkGroupSize;
    if (NumWorkGroups % 2 != 0 && NumWorkGroups != 1)
    {
        std::cerr << "Number of work groups should divide Vector Size evenly!" << "\n";
        return;
    }

    std::cout << " ========== PROFILING ========== " << "\n"
    << "Vector Size: " << VecSize << "\n"
    << "# of Work Groups: " << NumWorkGroups << "\n";
    
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

    {

    }

    for (int i = 0; i < ProfilingIters; i++)
    {
        auto StartTime = std::chrono::high_resolution_clock::now();

        sycl::event EnqueueEvent = Q.submit([&](sycl::handler &h)
        {
            h.parallel_for(sycl::nd_range<1>{sycl::range<1>{VecSize}, sycl::range<1>{WorkGroupSize}}, [=](sycl::nd_item<1> Item)
            {
                if (Item.get_local_id() == 0)
                {
                    int QueueIdx = Item.get_global_id() / WorkGroupSize;
                    auto &TargetQueue = TaskQueues[QueueIdx];

                    for (int i = 0; i < WorkGroupSize; i++)
                    {
                        int ItemVal = i + QueueIdx * WorkGroupSize;
                        TargetQueue.push(ItemVal);
                    }
                }
            });
        });
        Q.wait();

        sycl::event AddEvent = Q.submit([&](sycl::handler &h)
        {
            h.parallel_for(sycl::nd_range<1>{sycl::range<1>{VecSize}, sycl::range<1>{WorkGroupSize}}, [=](sycl::nd_item<1> Item)
            {
                int QueueIdx = Item.get_global_id() / WorkGroupSize;
                auto &TargetQueue = TaskQueues[QueueIdx];

                int ItemVal = TargetQueue.front(Item.get_local_id());
                R[ItemVal] = A[ItemVal] + B[ItemVal];
            });
        });
        Q.wait();
        
        sycl::event ShutdownEvent = Q.submit([&](sycl::handler &h)
        {
            h.parallel_for(sycl::nd_range<1>{sycl::range<1>{VecSize}, sycl::range<1>{WorkGroupSize}}, [=](sycl::nd_item<1> Item)
            {
                if (Item.get_local_id() == 0)
                {
                    int QueueIdx = Item.get_global_id() / WorkGroupSize;
                    auto &TargetQueue = TaskQueues[QueueIdx];

                    while (!TargetQueue.empty())
                    {
                        TargetQueue.pop();
                    }
                }
            });
        });
        Q.wait();

        auto EndTime = std::chrono::high_resolution_clock::now(); 
        durationMiliSecs ExecTime = EndTime - StartTime;
        TotalExecutionTimes.push_back(ExecTime.count());

        EnqueueEvents.push_back(EnqueueEvent);
        AddEvents.push_back(AddEvent);
        ShutdownEvents.push_back(ShutdownEvent);
    }

    bool CorrectAdd = check_vector_add(A, B, R, VecSize);
    std::cout << "Correct? " << std::boolalpha << CorrectAdd << std::endl;

    sycl::free(TaskQueues, Q);
    sycl::free(A, Q);
    sycl::free(B, Q);
    sycl::free(R, Q);

    VectorEventProfile EnqueueProfile = profile_vec_events(EnqueueEvents, TotalExecutionTimes, "Enqueue", VecSize);
    VectorEventProfile AddProfile = profile_vec_events(AddEvents, TotalExecutionTimes, "Queue Add", VecSize);
    VectorEventProfile ShutdownProfile = profile_vec_events(ShutdownEvents, TotalExecutionTimes, "Shutdown", VecSize);    
    
    EventProfiles.push_back(EnqueueProfile);
    EventProfiles.push_back(AddProfile);
    EventProfiles.push_back(ShutdownProfile);

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

    // ------------------------
    // PROFILING
    // ------------------------
    
    std::vector<VectorEventProfile> EventProfiles;

    for (int i = 10; i < 30; i++)
    {
        std::size_t VecSize = std::pow(2, i);
        profile_multi_queue_va<WorkGroupSize>(Q, VecSize, EventProfiles, ProfilingIters);
    }

    if (WriteToFile)
    {
        std::ofstream OutFile("profiling-results/profiling_global_multi_queue.csv", std::ios::app);
    
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
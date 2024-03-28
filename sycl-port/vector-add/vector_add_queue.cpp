#include <fstream>
#include <vector>
#include <CL/sycl.hpp>
#include "../sycl_utils.hpp"
#include "../ArrayQueue.cpp"
#include "va_profiler.cpp"

template<int VecSize>
void profile_for_vec_size(sycl::queue &Q, std::vector<VectorEventProfile> &EventProfiles, int ProfilingIters)
{
    eventList EnqueueEvents;
    eventList AddEvents;
    eventList ShutdownEvents;
    std::vector<double> TotalExecutionTimes;

    auto MyQueue = sycl::malloc_shared<SPMCArrayQueue<int, VecSize>>(16, Q);
    new (MyQueue) SPMCArrayQueue<int, VecSize>();

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
        auto StartTime = std::chrono::high_resolution_clock::now();

        // Task Spawning Kernel
        auto EnqueueEvent = Q.submit([&](sycl::handler &h)
        {
            h.single_task([=]()
            {
                for (int i =0; i < VecSize; i++)
                {
                    MyQueue->push(i);
                }
            });
        });
        Q.wait();

        // Main Loop SPMC working
        auto AddEvent = Q.submit([&](sycl::handler &h)
        {
            h.parallel_for(VecSize, [=](sycl::id<1> idx)
            {
                // Prevent out-of-bounds access
                if (idx < MyQueue->size())
                {
                    int itemVal = MyQueue->front(idx);
                    R[itemVal] = A[itemVal] + B[itemVal];
                } 
            }); 
        });
        Q.wait();

        // Shutdown Kernel
        auto ShutdownEvent = Q.submit([&](sycl::handler &h)
        {
            h.single_task([=]()
            {
                while (!MyQueue->empty())
                {
                    MyQueue->pop();
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
    std::cout << "Vector Addition for size: " << VecSize << "\n";
    std::cout << "Correct? " << std::boolalpha << CorrectAdd << std::endl;


    sycl::free(MyQueue, Q);
    sycl::free(A, Q);
    sycl::free(B, Q);
    sycl::free(R, Q);

    VectorEventProfile EnqueueProfile = profile_vec_events(EnqueueEvents, TotalExecutionTimes, "Enqueue", VecSize);
    VectorEventProfile AddProfile = profile_vec_events(AddEvents, TotalExecutionTimes, "Queue Add", VecSize);
    VectorEventProfile ShutdownProfile = profile_vec_events(ShutdownEvents, TotalExecutionTimes, "Shutdown", VecSize);    
    
    EventProfiles.push_back(EnqueueProfile);
    EventProfiles.push_back(AddProfile);
    EventProfiles.push_back(ShutdownProfile);
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cout << "Usage: <exec> <profiling iterations> <write to file (1|0)> <GPU name>" << std::endl;
        return 1;
    }
    int ProfilingIters = std::stoi(argv[1]);
    int WriteToFile = std::stoi(argv[2]);
    auto GPU = argv[3];

    sycl::default_selector device_selector;
    sycl::property_list props{sycl::property::queue::enable_profiling()}; // For measuring device execution times

    sycl::queue Q(device_selector, props);
    sycl::device Device = Q.get_device();

    std::cout << "\nBegin Device Information ====================================" << "\n";
    std::cout << Device;
    std::cout << "End Device Information   ====================================" << "\n";


    // ------------------------
    // PROFILING
    // ------------------------

    std::vector<VectorEventProfile> EventProfiles;

    // profile_for_vec_size<128>(Q, EventProfiles, ProfilingIters);
    // profile_for_vec_size<256>(Q, EventProfiles, ProfilingIters);
    // profile_for_vec_size<512>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<1024>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<2048>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<4096>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<8192>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<16384>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<32768>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<65536>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<131072>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<262144>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<524288>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<1048576>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<2097152>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<4194304>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<8388608>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<16777216>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<33554432>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<67108864>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<134217728>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<268435456>(Q, EventProfiles, ProfilingIters);
    profile_for_vec_size<536870912>(Q, EventProfiles, ProfilingIters);

    if (WriteToFile)
    {
        std::ofstream OutFile("profiling-results/profiling_single_queue.csv", std::ios::app);
    
        bool WriteHeaders = OutFile.tellp() == 0;
        if (WriteHeaders) {
            OutFile << "Event,MeanCGSubmissionTime(ms),MeanKernelExecTime(ms),MeanTotalExecTime(ms),VecSize,GPU\n";
        }
        
        for (const auto &profile : EventProfiles)
        {
            OutFile << profile.name << "," << profile.profileData.cgSubmissionTime << "," 
            << profile.profileData.kernelExecTime << "," 
            << profile.profileData.totalExecTime << ","
            << profile.vecSize << ","
            << GPU << "\n";
        }

        OutFile.close();
    }

    return 0;
}

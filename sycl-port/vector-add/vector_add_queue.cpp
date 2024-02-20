#include <fstream>
#include <vector>
#include <CL/sycl.hpp>
#include "../ArrayQueue.cpp"
#include "va_profiler.cpp"
#include "../Helpers.cpp"

template<int vectorSize>
void profileForVecSize(sycl::queue &Q, std::vector<VectorEventProfile> &eventProfiles, int profilingIters)
{
    eventList enqueueEvents;
    eventList addEvents;
    eventList shutdownEvents;
    std::vector<double> totalExecutionTimes;

    auto myQueue = sycl::malloc_shared<SPMCArrayQueue<int, vectorSize>>(16, Q);
    new (myQueue) SPMCArrayQueue<int, vectorSize>();

    int *A = sycl::malloc_shared<int>(vectorSize, Q);
    int *B = sycl::malloc_shared<int>(vectorSize, Q);
    int *R = sycl::malloc_shared<int>(vectorSize, Q);

    for (int i = 0; i < vectorSize; i++)
    {
        A[i] = 1;
        B[i] = 0;
    }

    for (int i = 0; i < profilingIters; i++)
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        // Task Spawning Kernel
        auto enqueueEvent = Q.submit([&](sycl::handler &h)
        {
            h.single_task([=]()
            {
                for (int i =0; i < vectorSize; i++)
                {
                    myQueue->push(i);
                }
            });
        });
        Q.wait();

        // Main Loop SPMC working
        auto addEvent = Q.submit([&](sycl::handler &h)
        {
            h.parallel_for(vectorSize, [=](sycl::id<1> idx)
            {
                // Prevent out-of-bounds access
                if (idx < myQueue->size())
                {
                    int itemVal = myQueue->front(idx);
                    R[itemVal] = A[itemVal] + B[itemVal];
                } 
            }); 
        });
        Q.wait();

        // Shutdown Kernel
        auto shutdownEvent = Q.submit([&](sycl::handler &h)
        {
            h.single_task([=]()
            {
                while (!myQueue->empty())
                {
                    myQueue->pop();
                }
            }); 
        });
        Q.wait();

        enqueueEvents.push_back(enqueueEvent);
        addEvents.push_back(addEvent);
        shutdownEvents.push_back(shutdownEvent);

        auto endTime = std::chrono::high_resolution_clock::now(); 
        durationMiliSecs execTime = endTime - startTime;
        totalExecutionTimes.push_back(execTime.count());
    }

    bool res = checkAdd(A, B, R, vectorSize);
    if (!res) std::cout << "Addition failed for vector size: " << vectorSize << "\n";

    sycl::free(myQueue, Q);
    sycl::free(A, Q);
    sycl::free(B, Q);
    sycl::free(R, Q);

    VectorEventProfile enqueueProfile = profileVecEvents(enqueueEvents, totalExecutionTimes, "Enqueue", vectorSize);
    VectorEventProfile addProfile = profileVecEvents(addEvents, totalExecutionTimes, "Queue Add", vectorSize);
    VectorEventProfile shutdownProfile = profileVecEvents(shutdownEvents, totalExecutionTimes, "Shutdown", vectorSize);    
    
    eventProfiles.push_back(enqueueProfile);
    eventProfiles.push_back(addProfile);
    eventProfiles.push_back(shutdownProfile);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: <exec> <profiling iterations> <write to file (1|0)>" << std::endl;
        return 1;
    }
    int profilingIters = std::stoi(argv[1]);
    int writeToFile = std::stoi(argv[2]);

    sycl::default_selector device_selector;
    sycl::property_list props{sycl::property::queue::enable_profiling()}; // For measuring device execution times

    sycl::queue Q(device_selector, props);
    printDevInfo(Q);

    // ------------------------
    // PROFILING
    // ------------------------

    std::vector<VectorEventProfile> eventProfiles;

    profileForVecSize<10>(Q, eventProfiles, profilingIters);
    profileForVecSize<100>(Q, eventProfiles, profilingIters);
    profileForVecSize<1000>(Q, eventProfiles, profilingIters);
    profileForVecSize<10000>(Q, eventProfiles, profilingIters);
    profileForVecSize<100000>(Q, eventProfiles, profilingIters);
    profileForVecSize<1000000>(Q, eventProfiles, profilingIters);
    profileForVecSize<10000000>(Q, eventProfiles, profilingIters);
    profileForVecSize<100000000>(Q, eventProfiles, profilingIters);

    if (writeToFile)
    {
        std::ofstream outFile("profiling_queue_va_usm.csv", std::ios::app);
    
        bool writeHeaders = outFile.tellp() == 0;
        if (writeHeaders) {
            outFile << "Event,CGSubmissionTime(ms),KernelExecTime(ms),TotalExecTime(ms),VectorSize\n";
        }
        
        for (auto profile : eventProfiles)
        {
            outFile << profile.name << "," << profile.profileData.cgSubmissionTime << "," 
            << profile.profileData.kernelExecTime << "," << profile.profileData.totalExecTime << ","
            << profile.vecSize << "\n";
        }

        outFile.close();
    }

    return 0;
}

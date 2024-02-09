#include <fstream>
#include "va_profiler.cpp"
#include "../Helpers.cpp"

std::vector<int> vectorSizes = {10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};

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
    for (int vectorSize : vectorSizes)
    {
        eventList addEvents;
        std::vector<double> totalExecutionTimes;

        int *A = sycl::malloc_shared<int>(vectorSize, Q);
        int *B = sycl::malloc_shared<int>(vectorSize, Q);
        int *R = sycl::malloc_shared<int>(vectorSize, Q);

        for (int i = 0; i < vectorSize; i++)
        {
            A[i] = i;
            B[i] = i;
        }

        for (int i = 0; i < profilingIters; i++)
        {
            auto startTime = std::chrono::high_resolution_clock::now();

            auto addEvent = Q.submit([&](sycl::handler &h)
            {
                h.parallel_for(vectorSize, [=](sycl::id<1> idx)
                {
                    R[idx] = A[idx] + B[idx];
                });
            });
            Q.wait();

            addEvents.push_back(addEvent);

            auto endTime = std::chrono::high_resolution_clock::now(); 
            durationMiliSecs execTime = endTime - startTime;
            totalExecutionTimes.push_back(execTime.count());
        }

        bool res = checkAdd(A, B, R, vectorSize);
        if (!res) std::cout << "Addition failed for vector size: " << vectorSize << "\n";
        
        sycl::free(A, Q);
        sycl::free(B, Q);
        sycl::free(R, Q);

        VectorEventProfile addProfile = profileVecEvents(addEvents, totalExecutionTimes, "Simple Add", vectorSize);
        eventProfiles.push_back(addProfile);
    }

    if (writeToFile)
    {
        std::ofstream outFile("profiling_va_usm.csv", std::ios::app);

        bool writeHeaders = outFile.tellp() == 0;
        if (writeHeaders) {
            outFile << "Event,CGSubmissionTime(ms),KernelExecTime(ms),TotalExecTime(ms),VectorSize\n";
        }
        
        for (const auto &profile : eventProfiles)
        {
            outFile << profile.name << "," << profile.profileData.cgSubmissionTime << "," 
            << profile.profileData.kernelExecTime << "," << profile.profileData.totalExecTime << ","
            << profile.vecSize << "\n";
        }

        outFile.close();
    }

    return 0;
}

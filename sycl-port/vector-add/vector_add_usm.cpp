#include <cmath>
#include <fstream>
#include "../sycl_utils.hpp"
#include "../ArrayQueue.cpp"
#include "va_profiler.cpp"
  
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
    for (int i = 10; i < 30; i++)
    {
        size_t VecSize = std::pow(2, i);
        
        eventList AddEvents;
        std::vector<double> TotalExecutionTimes;

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

            auto AddEvent = Q.submit([&](sycl::handler &h)
            {
                h.parallel_for(VecSize, [=](sycl::id<1> idx)
                {
                    R[idx] = A[idx] + B[idx];
                });
            });
            Q.wait();

            AddEvents.push_back(AddEvent);

            auto EndTime = std::chrono::high_resolution_clock::now(); 
            durationMiliSecs ExecTime = EndTime - StartTime;
            std::cout << "Exec time for iter {" << i << "}, for vecsize: " << VecSize << " " << ExecTime.count() << "\n";
            TotalExecutionTimes.push_back(ExecTime.count());
        }

        bool CorrectAdd = check_vector_add(R, VecSize);
        std::cout << "Vector Addition for size: " << VecSize << "\n";
        std::cout << "Correct? " << std::boolalpha << CorrectAdd << std::endl;
        
        sycl::free(A, Q);
        sycl::free(B, Q);
        sycl::free(R, Q);

        VectorEventProfile addProfile = profile_vec_events(AddEvents, TotalExecutionTimes, "Simple Add", VecSize);
        EventProfiles.push_back(addProfile);
    }

    if (WriteToFile)
    {
        std::ofstream OutFile("profiling-results/profiling_va_usm_test.csv", std::ios::app);

        bool WriteHeaders = OutFile.tellp() == 0;
        if (WriteHeaders) {
            OutFile << "Event,MeanCGSubmissionTime(ms),MeanKernelExecTime(ms),StdKernelExecTime(ms),MeanTotalExecTime(ms),StdTotalExecTime(ms),VecSize,GPU\n";
        }
        
        for (const auto &profile : EventProfiles)
        {
            OutFile << profile.name << "," << profile.profileData.cgSubmissionTime << "," 
            << profile.profileData.kernelExecTime << ","
            << profile.profileData.kernelExecTimeStdDev << "," 
            << profile.profileData.totalExecTime << ","
            << profile.profileData.totalExecTimeStdDev << ","
            << profile.vecSize << ","
            << GPU << "\n";
        }

        OutFile.close();
    }

    return 0;
}

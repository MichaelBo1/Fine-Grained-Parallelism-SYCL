#include <chrono>
#include <fstream>
#include <vector>
#include <CL/sycl.hpp>

using u64 = uint64_t;
using eventList = std::vector<sycl::event>;
using durationMiliSecs = std::chrono::duration<double, std::milli>;

constexpr double nanoSecInMilisec = 1000000.0;

double toMili(u64 timeNanoSecs)
{
    return timeNanoSecs / nanoSecInMilisec;
}

// Miliseconds
struct ProfileData
{
    double cgSubmissionTime{0};
    double kernelExecTime{0};
    double totalExecTime{0};
};

struct VectorEventProfile
{
    ProfileData profileData;
    std::string name;
    int vecSize;
};

VectorEventProfile profileVecEvents(const eventList &events, const std::vector<double> &totalExecTimes, const std::string &eventName, int vecSize)
{
    VectorEventProfile vecEventProfile;
    ProfileData profileData;
    u64 cgSubmissionTime = 0;
    u64 kernelExecTime = 0;
    double runningExecTime = 0;

    std::size_t numEvents = events.size();
    for (std::size_t i = 0; i < numEvents; i++)
    {
        auto curEvent = events.at(i);

        auto cgSubmissionTimePoint = curEvent.get_profiling_info<sycl::info::event_profiling::command_submit>();
        auto startKernelExecTimePoint = curEvent.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto endKernelExecTimePoint = curEvent.get_profiling_info<sycl::info::event_profiling::command_end>();
        cgSubmissionTime += startKernelExecTimePoint - cgSubmissionTimePoint;
        kernelExecTime += endKernelExecTimePoint - startKernelExecTimePoint;
        runningExecTime += totalExecTimes.at(i);
    }

    profileData.cgSubmissionTime = toMili(cgSubmissionTime / numEvents);
    profileData.kernelExecTime = toMili(kernelExecTime / numEvents);
    profileData.totalExecTime = runningExecTime / numEvents;
    
    vecEventProfile.profileData = profileData;
    vecEventProfile.name = eventName;
    vecEventProfile.vecSize = vecSize;

    return vecEventProfile;
}

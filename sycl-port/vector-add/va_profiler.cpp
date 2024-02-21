#include <chrono>
#include <fstream>
#include <vector>
#include <CL/sycl.hpp>

using u64 = uint64_t;
using eventList = std::vector<sycl::event>;
using durationMiliSecs = std::chrono::duration<double, std::milli>;

constexpr double nanoSecInMilisec = 1000000.0;

double to_mili(u64 timeNanoSecs)
{
    return timeNanoSecs / nanoSecInMilisec;
}

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

VectorEventProfile profile_vec_events(const eventList &Events, const std::vector<double> &TotalExecTimes, const std::string &EventName, int VecSize)
{
    VectorEventProfile VecEventProfile;
    ProfileData ProfileData;

    u64 CgSubmissionTime = 0;
    u64 KernelExecTime = 0;
    double RunningExecTime = 0;

    std::size_t NumEvents = Events.size();
    for (int i = 0; i < NumEvents; i++)
    {
        auto CurEvent = Events.at(i);

        auto CgSubmissionTimePoint = CurEvent.get_profiling_info<sycl::info::event_profiling::command_submit>();
        auto StartKernelExecTimePoint = CurEvent.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto EndKernelExecTimePoint = CurEvent.get_profiling_info<sycl::info::event_profiling::command_end>();

        CgSubmissionTime += StartKernelExecTimePoint - CgSubmissionTimePoint;
        KernelExecTime += EndKernelExecTimePoint - StartKernelExecTimePoint;
        RunningExecTime += TotalExecTimes.at(i);
    }

    ProfileData.cgSubmissionTime = to_mili(CgSubmissionTime / NumEvents);
    ProfileData.kernelExecTime = to_mili(KernelExecTime / NumEvents);
    ProfileData.totalExecTime = RunningExecTime / NumEvents;
    
    VecEventProfile.profileData = ProfileData;
    VecEventProfile.name = EventName;
    VecEventProfile.vecSize = VecSize;

    return VecEventProfile;
}

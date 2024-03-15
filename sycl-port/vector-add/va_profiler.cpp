#include <chrono>
#include <fstream>
#include <vector>
#include <CL/sycl.hpp>

using u64 = uint64_t;
using durationMiliSecs = std::chrono::duration<double, std::milli>;

constexpr double NanoSecInMilisec = 1000000.0;

double to_mili(u64 TimeNanoSecs)
{
    return TimeNanoSecs / NanoSecInMilisec;
}

struct TimingEvent
{
    std::string Name;
    std::size_t VectorSize;
    double ExecTime;
};
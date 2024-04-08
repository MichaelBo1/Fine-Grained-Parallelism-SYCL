#include <CL/sycl.hpp>
#include "../../sycl_utils.hpp"
#include "../../tasking/ArrayQueue.cpp"
#include "../va_profiler.cpp"

template<std::size_t VecSize>
void single_queue_add(sycl::queue &Q, std::vector<TimingEvent> &Events)
{
    auto StartTimePoint = std::chrono::high_resolution_clock::now();

    auto MyQueue = sycl::malloc_shared<SPMCArrayQueue<int, VecSize>>(1, Q);
    new (MyQueue) SPMCArrayQueue<int, VecSize>();

    int *A = sycl::malloc_shared<int>(VecSize, Q);
    int *B = sycl::malloc_shared<int>(VecSize, Q);
    int *R = sycl::malloc_shared<int>(VecSize, Q);

    for (int i = 0; i < VecSize; i++)
    {
        A[i] = 1;
        B[i] = 0;
    }

    auto MemorySetupTimePoint = std::chrono::high_resolution_clock::now();

    sycl::event EnqueueEvent = Q.submit([&](sycl::handler &h)
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

    sycl::event AddEvent = Q.submit([&](sycl::handler &h)
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

    sycl::event ShutdownEvent = Q.submit([&](sycl::handler &h)
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

    auto EndTimePoint = std::chrono::high_resolution_clock::now(); 

    durationMiliSecs ExecTime = EndTimePoint - StartTimePoint;
    durationMiliSecs MemTime = MemorySetupTimePoint - StartTimePoint;

    auto StartEnqueueKernelExecTimePoint = EnqueueEvent.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto EndEnqueueKernelExecTimePoint = EnqueueEvent.get_profiling_info<sycl::info::event_profiling::command_end>();
    double EnqueueKernelProfileTime = to_mili(EndEnqueueKernelExecTimePoint - StartEnqueueKernelExecTimePoint);

    auto StartAddKernelExecTimePoint = AddEvent.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto EndAddKernelExecTimePoint = AddEvent.get_profiling_info<sycl::info::event_profiling::command_end>();
    double AddKernelProfileTime = to_mili(EndAddKernelExecTimePoint - StartAddKernelExecTimePoint);

    auto StartShutdownKernelExecTimePoint = ShutdownEvent.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto EndShutdownKernelExecTimePoint = ShutdownEvent.get_profiling_info<sycl::info::event_profiling::command_end>();
    double ShutdownKernelProfileTime = to_mili(EndShutdownKernelExecTimePoint - StartShutdownKernelExecTimePoint);

    Events.push_back({"Total Exec Time", VecSize, ExecTime.count()});
    Events.push_back({"Memory Setup Time", VecSize, MemTime.count()});
    Events.push_back({"Enqueue Kernel Exec Time", VecSize, EnqueueKernelProfileTime});
    Events.push_back({"Add Kernel Exec Time", VecSize, AddKernelProfileTime});
    Events.push_back({"Shutdown Kernel Exec Time", VecSize, ShutdownKernelProfileTime});

    // bool CorrectAdd = check_vector_add(R, VecSize);
    // std::cout << "Vector Addition for size: " << VecSize << "\n";
    // std::cout << "Correct? " << std::boolalpha << CorrectAdd << std::endl;

    sycl::free(MyQueue, Q);
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

    // ------------------------
    // PROFILING
    // ------------------------
    
    std::vector<TimingEvent> Events;

    single_queue_add<1024>(Q, Events);
    single_queue_add<2048>(Q, Events);
    single_queue_add<4096>(Q, Events);
    single_queue_add<8192>(Q, Events);
    single_queue_add<16384>(Q, Events);
    single_queue_add<32768>(Q, Events);
    single_queue_add<65536>(Q, Events);
    single_queue_add<131072>(Q, Events);
    single_queue_add<262144>(Q, Events);
    single_queue_add<524288>(Q, Events);
    single_queue_add<1048576>(Q, Events);
    single_queue_add<2097152>(Q, Events);
    single_queue_add<4194304>(Q, Events);
    single_queue_add<8388608>(Q, Events);
    single_queue_add<16777216>(Q, Events);
    single_queue_add<33554432>(Q, Events);
    single_queue_add<67108864>(Q, Events);
    single_queue_add<134217728>(Q, Events);
    single_queue_add<268435456>(Q, Events);
    single_queue_add<536870912>(Q, Events);

    for (const TimingEvent &event : Events)
    {
        std::cout << event.Name << "," << event.ExecTime << "," << event.VectorSize << "," << DevName << "\n";
    }

    return 0;
}
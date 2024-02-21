#include <CL/sycl.hpp>
#include "../sycl_utils.hpp"
#include "../ArrayQueue.cpp"

// void vector_add(sycl::queue &Q, const size_t MaxSize, const size_t VecSize)
// {
//     int *A = sycl::malloc_shared<int>(VecSize, Q);
//     int *B = sycl::malloc_shared<int>(VecSize, Q);
//     int *R = sycl::malloc_shared<int>(VecSize, Q);

//     for (size_t i = 0; i < VecSize; i++)
//     {
//         A[i] = 1;
//         B[i] = 0;
//     }
//     constexpr size_t WorkGroupSize = 64; // Number of work-items per work-group

//     Q.submit([&](sycl::handler &h) 
//     {
//         auto LocalQueue = sycl::accessor<SPMCArrayQueue<int, WorkGroupSize>, 1, sycl::access::mode::read_write, sycl::access::target::local>(sycl::range<1>(1), h);

//         // sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> LocalQueue(sycl::range<1>(WorkGroupSize), h);
//         h.parallel_for(sycl::nd_range<1>{sycl::range<1>{VecSize}, sycl::range<1>{WorkGroupSize}}, [=](sycl::nd_item<1> Item)
//         {
//             if (Item.get_local_id() == 0)
//             {
//                 new (LocalQueue.get_pointer()) SPMCArrayQueue<int, WorkGroupSize>();
//                 for (int i = 0; i < WorkGroupSize; i++)
//                 {
//                     LocalQueue[0].push(i);
//                 }
//             }

//             Item.barrier(sycl::access::fence_space::local_space);

//             auto idx = Item.get_global_id();
//             R[idx] = A[idx] + B[idx];
            


//         });

//     });
//     Q.wait();

//     bool CorrectAdd = check_vector_add(A, B, R, VecSize);
//     std::cout << "Vector Addition for size: " << VecSize << "\n";
//     std::cout << "Correct? " << std::boolalpha << CorrectAdd << std::endl;

//     sycl::free(A, Q);
//     sycl::free(B, Q);
//     sycl::free(R, Q);
// }

template<size_t VecSize>
void multi_queue_vector_add(sycl::queue &Q, const size_t WorkGroupSize)
{
    const size_t NumWorkGroups = VecSize / WorkGroupSize;
    if (NumWorkGroups % 2 != 0)
    {
        throw "Number of work groups should divide Vector Size evenly!";
    }

    auto TaskQueues = sycl::malloc_shared<SPMCArrayQueue<int, VecSize>>(NumWorkGroups, Q);
    for (size_t i = 0; i < NumWorkGroups; i++)
    {
        new (TaskQueues + i) SPMCArrayQueue<int, VecSize>();
    }

    int *A = sycl::malloc_shared<int>(VecSize, Q);
    int *B = sycl::malloc_shared<int>(VecSize, Q);
    int *R = sycl::malloc_shared<int>(VecSize, Q);

    for (int i = 0; i < VecSize; i++)
    {
        A[i] = 1;
        B[i] = 0;
    }

    Q.submit([&](sycl::handler &h)
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
            int QueueIdx = Item.get_global_id() / WorkGroupSize;
            auto &TargetQueue = TaskQueues[QueueIdx];

            while (!TargetQueue.empty())
            {
                TargetQueue.pop();
            }
        });
    });
    Q.wait();

    bool CorrectAdd = check_vector_add(A, B, R, VecSize);
    std::cout << "Vector Addition for size: " << VecSize << "\n";
    std::cout << "Correct? " << std::boolalpha << CorrectAdd << std::endl;

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

    std::cout << "\nBegin Device Information ====================================" << "\n";
    std::cout << Device;
    std::cout << "End Device Information   ====================================" << "\n";

    constexpr size_t VecSize = 32;
    // auto WorkGroupSize = std::min(Device.get_info<sycl::info::device::max_work_group_size>(), VecSize + (VecSize % 2));
    auto WorkGroupSize = 16;
    if (WorkGroupSize % 2 != 0)
    {
        throw "Work-group size needs to be even!";
    }
    // Checks for local-mem support
    auto HasLocalMem = Device.is_host() || Device.get_info<sycl::info::device::local_mem_type>() != sycl::info::local_mem_type::none;
    auto LocalMemSize = Device.get_info<sycl::info::device::local_mem_size>();
    if (!HasLocalMem || LocalMemSize < (WorkGroupSize * sizeof(int)))
    {
        throw "Device does not have enough local memory!";
    }

    // vector_add(Q, WorkGroupSize, VecSize);
    multi_queue_vector_add<VecSize>(Q, WorkGroupSize);
    return 0;
}
#include <CL/sycl.hpp>
#include "../sycl_utils.hpp"
#include "Mutex.hpp"

int main()
{
    sycl::default_selector device_selector;

    sycl::queue Q(device_selector);
    sycl::device Device = Q.get_device();

    std::cout << "Device Info:\n";
    std::cout << Device;


    auto SimpleMutex = sycl::malloc_shared<Mutex>(1, Q);
    new (SimpleMutex) Mutex();

    int *Val = sycl::malloc_shared<int>(1, Q);
    *Val = 0;

    // Q.submit([&](sycl::handler &h)
    // {
    //     sycl::stream out(1024, 256, h);

    //     h.parallel_for(16, [=](sycl::id<1> idx)
    //     {
    //         out << "Trying to process for idx: " << idx << "\n";
    //         SimpleMutex->lock();
    //         *Val += 1;
    //         SimpleMutex->unlock();
    //     });
    // });
    // Q.wait();
    Q.submit([&](sycl::handler &h)
    {
        sycl::stream out(1024, 256, h);
        // 4 groups of 4
        h.parallel_for(sycl::nd_range<1>{sycl::range<1>{16}, sycl::range<1>{4}}, [=](sycl::nd_item<1> Item)
        {
            bool IsMaster = Item.get_local_id() == 0;
            sycl::group Group = Item.get_group();

            if (IsMaster)
            {
                out << "Master in " << Item.get_global_id() << " acquiring lock \n";
                SimpleMutex->lock();
                *Val += 1;
            }

            sycl::group_barrier(Group);
            SimpleMutex->unlock();
            sycl::group_barrier(Group);


        });
    });
    Q.wait();

    std::cout << "Val: " << *Val << "\n";
    if (*Val == 4)
    {
        std::cout << "Mutex works as expected for parallel_for!\n";
    }
    else
    {
        std::cout << "Mutex does not work as expected for parallel_for!\n";
    }

    sycl::free(SimpleMutex, Q);
    sycl::free(Val, Q);
}
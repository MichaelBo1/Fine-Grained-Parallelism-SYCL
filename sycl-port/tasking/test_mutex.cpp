#include <CL/sycl.hpp>
#include "../sycl_utils.hpp"
#include "Mutex.hpp"

int main void()
{
    sycl::default_selector device_selector;

    sycl::queue Q(device_selector);
    sycl::device Device = Q.get_device();

    std::cout << "Device Info:\n";
    std::cout << Device;


    Mutex SimpleMutex = sycl::malloc_shared<Mutex>(1, Q);
    new (SimpleMutex) Mutex();

    int *Val = sycl::malloc_shared<int>(1, Q);
    *Val = 0;

    Q.submit([&](sycl::handler &h)
    {
        h.parallel_for(16, [=](sycl::id<1> idx)
        {
            SimpleMutex->lock();
            *Val += 1;
            SimpleMutex->unlock();
        });
    });
    Q.wait();

    std::cout << "Val: " << *Val << "\n";
    if (*Val == 16)
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
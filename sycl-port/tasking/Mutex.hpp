#ifndef __MUTEX_H__
#define __MUTEX_H__

#include <CL/sycl.hpp>

class Mutex
{
public:
    Mutex(){};
    ~Mutex(){};
    void lock()
    {
        sycl::atomic_ref<
            int,
            sycl::memory_order::acq_rel,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space> atomic_mutex(mutex);

        int expected = 0;
        int desired = 1;
        while (atomic_mutex.compare_exchange_weak(expected, desired, sycl::memory_order_acq_rel, sycl::memory_order_acquire) != true);
 
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    };
    void unlock()
    {
        sycl::atomic_ref<
            int,
            sycl::memory_order::acq_rel,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space> atomic_mutex(mutex);

        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
        atomic_mutex.exchange(0);
    };

private:
    int mutex = 0;
};
#endif

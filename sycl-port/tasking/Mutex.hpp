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
        atomic_ref<
            int,
            memory_order::acq_rel,
            memory_scope::device,
            access::address_space::global_space> atomic_mutex(mutex);

        while (atomic_mutex.compare_exchange_weak(0, 1) != true) // what is the default scope?
            ;
        // necessary since "atomic functions do not act as memory fences and do
        // not imply synchronization or ordering constraints for memory
        // operations" (cuda documentation)
        // thread does not work on variables within the critical section before mutex is set to 1
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device)
    };
    void unlock()
    {
        atomic_ref<
            int,
            memory_order::acq_rel,
            memory_scope::device,
            access::address_space::global_space> atomic_mutex(mutex);
        // thread does not work on variables within the critical section after the lock was released
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device)
        atomic_mutex.exchange(0);
    };

private:
    int mutex = 0;
};
#endif

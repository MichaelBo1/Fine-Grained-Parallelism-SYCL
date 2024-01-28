#include <CL/sycl.hpp>
#include "../ArrayQueue.cpp"

void device_alloc(sycl::queue &Q)
{
    auto myQueue = sycl::malloc_device<SPMCArrayQueue<int, 10>>(1, Q);

    Q.submit([&](sycl::handler &h)
    { 
        h.single_task([=]()
        {
            new (myQueue) SPMCArrayQueue<int, 10>();

            myQueue->push(6);
        });
    });
    Q.wait();

    SPMCArrayQueue<int, 10> hostQueueCopy;

    Q.submit([&](sycl::handler &h)
    {
        h.memcpy(&hostQueueCopy, myQueue, sizeof(SPMCArrayQueue<int, 10>));
    });
    Q.wait();
    
    std::cout << "Contents of the queue:" << std::endl;
    while (!hostQueueCopy.empty()) {
        std::cout << hostQueueCopy.front() << " ";
        hostQueueCopy.pop();
    }
    std::cout << std::endl;
    sycl::free(myQueue, Q);
}

void shared_alloc(sycl::queue &Q)
{
    auto myQueue = sycl::malloc_shared<SPMCArrayQueue<int, 10>>(1, Q);
    new (myQueue) SPMCArrayQueue<int, 10>();

    for (int i = 0; i < 3; i++)
    {
        myQueue->push(i);
    }
    Q.submit([&](sycl::handler &h)
    { 
        h.single_task([=]()
        {
            myQueue->pop();
            myQueue->push(6);
        });
    });
    Q.wait();

    std::cout << "Contents of the queue:" << std::endl;
    while (!myQueue->empty()) {
        std::cout << myQueue->front() << " ";
        myQueue->pop();
    }
    std::cout << std::endl;
    sycl::free(myQueue, Q);
}


int main(int argc, char **argv)
{

    sycl::default_selector device_selector;

    sycl::queue Q(device_selector);

     std::cout << "Running on "
    << Q.get_device().get_info<sycl::info::device::name>()
    << "\n";    

    // device_alloc(Q);
    shared_alloc(Q);
    return 0;
}


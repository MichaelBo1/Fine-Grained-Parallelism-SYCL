#include <CL/sycl.hpp>
#include "../ArrayQueue.cpp"

constexpr int vectorSize = 40;

void printDevInfo(const sycl::queue &Q)
{
    std::cout << "Running on "
    << Q.get_device().get_info<sycl::info::device::name>()
    << "\n"; 
}

void printVec(int *vec, const std::string &label)
{
    std::cout << label << ": ";

    for (int i = 0; i < vectorSize; i++)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{

    sycl::default_selector device_selector;
    sycl::queue Q(device_selector);
    printDevInfo(Q);
    std::size_t maxWorkItems = Q.get_device().get_info<sycl::info::device::max_work_group_size>();

    std::cout << "MAX WORK ITEMS: " << maxWorkItems << std::endl;
   
    auto myQueue = sycl::malloc_shared<SPMCArrayQueue<int, vectorSize>>(1, Q);
    new (myQueue) SPMCArrayQueue<int, vectorSize>();

    int *A = sycl::malloc_shared<int>(vectorSize, Q);
    int *B = sycl::malloc_shared<int>(vectorSize, Q);
    int *res = sycl::malloc_shared<int>(vectorSize, Q);

    for (int i = 0; i < vectorSize; i++)
    {
        A[i] = 1;
        B[i] = i+1;
        res[i] = 0;
    }
    printVec(A, "A");
    printVec(B, "B");

    // Task Spawning Kernel
    
    Q.submit([&](sycl::handler &h)
    { 
        h.single_task([=]()
        {
            for (int i = 0; i < vectorSize; i++)
            {
                myQueue->push(i);
            }
        });
    });
    Q.wait();

    // Main Loop SPMC working
    sycl::range<1> numItems{maxWorkItems};
    Q.submit([&](sycl::handler &h)
    {
        h.parallel_for(numItems, [=](sycl::id<1> idx)
        {
            // Prevent out-of-bounds access
            if (idx < myQueue->size())
            {
                int itemVal = myQueue->front(idx);
                res[itemVal] = A[itemVal] + B[itemVal];
            }
        });
    });
    Q.wait();

    printVec(res, "R");

    // Shutdown Kernel
    Q.submit([&](sycl::handler &h)
    { 
        h.single_task([=]()
        {
            while (!myQueue->empty())
            {
                myQueue->pop();
            }
        });
    });
    Q.wait();

    std::cout << "Is task queue empty? " << myQueue->empty() << std::endl;

    sycl::free(myQueue, Q);
    sycl:free(A, Q);
    sycl::free(B, Q);
    sycl::free(res, Q);
    // checks for successful allocation
    return 0;
}


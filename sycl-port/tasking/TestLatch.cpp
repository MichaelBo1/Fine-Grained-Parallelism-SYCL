struct device_latch
{
    device_latch(size_t num_groups): counter(0), expected(num_groups) {}

    template <int Dimensions>
    void arrive_and_wait(sycl::nd_item<Dimensions> &it)
    {
        it.barrier();

        if ((it.get_local_linear_id() == 0)) {
            atomic_ref<
                size_t,
                memory_order::acq_rel,
                memory_scope::device,
                access:address_space::global_space> atomic_counter(counter);

            atomic_counter++;

            while (atomic_counter.load() != expected) {}

        }
        it.barrier();

    }
    size_t counter;
    size_t expected;
}

// Use Case

void* ptr = sycl::malloc_shared(sizeof(device_latch), Q);
device_latch* latch = new (ptr) device_latch(num_groups);

Q.submit([&](handler& h) {
    h.parallel_for(R, [=](nd_item<1> it) {
        // Every work-item writes a 1 to its location
        data[it.get_global_linear_id()] = 1;
        // Every work-item waits for all writes
        latch->arrive_and_wait(it);
        // Every work-item sums the values it can see
        size_t sum = 0;
        for (int i = 0; i < num_groups * items_per_group; ++i) {
            sum += data[i]; 
        }
        sums[it.get_global_linear_id()] = sum;
    });
}).wait();
    free(ptr, Q);
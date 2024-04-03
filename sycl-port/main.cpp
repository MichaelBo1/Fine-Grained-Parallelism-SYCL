#include "tasking/ArrayQueue.cpp"
#include <CL/sycl.hpp>


template <typename TaskType>
void produce_and_consume(sycl::queue &Q)
{

}

/*
FOR NOW JUST CONSIDER HOW THE CODE COULD BE DONE
-- NEED TO BE IN A SYCL ND-RANGE KERNEL

>> Get correct task queue
>> Setup initial tasks if master work-item
group_barrier()

>> While the queue_size > 0 or the task is not done
group_barrier

auto ID = Item.get_local_id()

// Assumed to be in ND-range
sycl::group = Item.get_group()
if (ID == 0) # master
{
    initialize queue
    track queue size
}

sycl::group_barrier(Group)

while (queue_size > 0 or !Finished(TaskType))
{
    sycl::group_barrier(Group)
    
    if (queue_size > ID)
    {
        queue->front[ID].execute() # Task
    }

    sycl::group_barrier(Group)

    # master handles DR
    if (ID == 0)
    {
        # number of task done

        for (int task = 0; task < tasks_done; task++)
        {
            current_task_id = queue->front(task).get_id()
            SolveDependencies<TaskType>(current_task_id)
        }
        global_queue->pop(tasks_done)
        queue_size = global_queue->size();
    }

    sycl::group_barrier(Group)
}

*/
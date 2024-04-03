#ifndef __TASK_H__
#define __TASK_H__

#include "Processor.hpp"

class BaseTask
{
public:
    virtual void execute() const = 0;
    virtual ~BaseTask() = default;
};

template <typename ProcessorType, typename DataType>
class Task : public BaseTask
{
public:
    Task() : processor_(nullptr)
   {
   }

    Task(ProcessorType *proc, DataType data) : processor_(proc), data_(data)
    {
    }

    /**
      \brief Executes the stored Executor of this Task.
    */
    void execute() const override
    {
        processor_->execute(data_);
    }

    DataType getBoxId() const
    {
        return data_;
    }

private:
    /**
     * This needs to be a pointer because all abstraction is hidden in the inheritance of
     * Executors.
     * This avoids template parameters in this class (Task).
     */
    ProcessorType *processor_;
    DataType data_;
};

#endif

#ifndef __PROCESSOR_H__
#define __PROCESSOR_H__

#include "../accumulation/accumulation.hpp"
#include <tuple>

template <typename FMMHandle, typename MutexContainer, typename dependency_manager>
class TaskFactory;

template <
        template <typename...> class ConcreteProcessor,
        /*typename fmm_handle_type,
        typename mutex_container_type,*/
        typename fmm_handle_type,
        typename dependency_manager>
class AbstractProcessor
{
public:
    /*using FMMHandle = fmm_handle_type;
    using DependencyManagerType = dependency_manager;
    using MutexContainerType = mutex_container_type;
    using TaskFactoryType = TaskFactory<FMMHandle, MutexContainerType, DependencyManagerType>;

    using Processor = ConcreteProcessor<FMMHandle, MutexContainerType, DependencyManagerType>;*/
    using FMMHandleType = fmm_handle_type;
    using DependencyManagerType = dependency_manager;
    using Processor = ConcreteProcessor<FMMHandleType, DependencyManagerType>;

    /* AbstractProcessor(
             FMMHandle &fmm_handle,
             MutexContainerType &mutex_container,
             TaskFactoryType &task_factory,
             DependencyManagerType &dm)
             : fmm_handle_(fmm_handle),
               mutex_container_(mutex_container),
               task_factory_(task_factory),
               dm_(dm){}*/
    //ordering of parameters has to be the same as in acc()
    __device__ __host__ AbstractProcessor(DependencyManagerType &dm, FMMHandleType& fmm_handle) : fmm_handle_(fmm_handle), dm_(dm)
    {
    //this works, since AbstractProcessor is constructed locally in main.cu
    /*printf("Comp Tree in run_computation():\n");
    for (unsigned int i = 0; i < this->fmm_handle_.num_boxes(); i++)
    {
        printf("%ld\n", this->fmm_handle_.lin_element(i)[0]);
    }*/
    }

    template <typename DataType>
    __device__ __host__ void execute(DataType data)
    {
        // concrete_processor().pre_processing(tree_partition);
        concrete_processor().run_computation(data);
        // concrete_processor().post_processing(tree_partition);
    }

    /**
     * All method should be empty by default and hidden with an implementation in child
     * class/concrete processor if needed.
     */
    template <typename TreePartitionType>
    void pre_processing(TreePartitionType /*unused*/)
    {
    }
    template <typename TreePartitionType>
    void run_computation(TreePartitionType /*unused*/)
    {
    }
    template <typename TreePartitionType>
    void post_processing(TreePartitionType /*unused*/)
    {
    }

protected:
    FMMHandleType &fmm_handle_;
    /*MutexContainerType &mutex_container_;
    TaskFactoryType &task_factory_;*/
    DependencyManagerType &dm_;

    __device__ __host__ Processor &concrete_processor()
    {
        return *static_cast<Processor *>(this);
    }
};

template <typename... Args>
struct P2PProcessor : public AbstractProcessor<P2PProcessor, Args...>
{
    typedef AbstractProcessor<P2PProcessor, Args...> Base;  // define AbstractProcessor as Base
                                                            // Class
    using Base::AbstractProcessor;  // Inherit constructor of Base (Reason: Constructors are not
                                    // inherited by default!)
    template <typename DataType>
    __device__ void run_computation(DataType data)
    {
        accumulation::p2p(data, this->dm_, this->fmm_handle_);
    }
};

template <typename... Args>
struct P2MProcessor : public AbstractProcessor<P2MProcessor, Args...>
{
    typedef AbstractProcessor<P2MProcessor, Args...> Base;  // define AbstractProcessor as Base
                                                            // Class
    using Base::AbstractProcessor;  // Inherit constructor of Base (Reason: Constructors are not
                                    // inherited by default!)
    template <typename DataType>
    __device__ void run_computation(DataType data)
    {
        accumulation::p2m(data, this->dm_, this->fmm_handle_);
    }
};

template <typename... Args>
struct M2MProcessor : public AbstractProcessor<M2MProcessor, Args...>
{
    typedef AbstractProcessor<M2MProcessor, Args...> Base;  // define AbstractProcessor as Base
                                                            // Class
    using Base::AbstractProcessor;  // Inherit constructor of Base (Reason: Constructors are not
                                    // inherited by default!)
    template <typename DataType>
    __device__ void run_computation(DataType data)
    {
        accumulation::m2m(data, this->dm_, this->fmm_handle_);
    }
};

template <typename... Args>
struct L2LProcessor : public AbstractProcessor<L2LProcessor, Args...>
{
  typedef AbstractProcessor<L2LProcessor, Args...> Base;  // define AbstractProcessor as Base
                                                          // Class
  using Base::AbstractProcessor;  // Inherit constructor of Base (Reason: Constructors are not
                                  // inherited by default!)
  template <typename DataType>
  __device__ void run_computation(DataType data)
  {
    accumulation::l2l(data, this->dm_, this->fmm_handle_);
  }
};

template <typename... Args>
struct M2LProcessor : public AbstractProcessor<M2LProcessor, Args...>
{
  typedef AbstractProcessor<M2LProcessor, Args...> Base;  // define AbstractProcessor as Base
  // Class
  using Base::AbstractProcessor;  // Inherit constructor of Base (Reason: Constructors are not
                                  // inherited by default!)
  template <typename DataType>
  __device__ void run_computation(DataType data)
  {
    accumulation::m2l(data, this->dm_, this->fmm_handle_);
  }
};


template <typename... Args>
struct L2PProcessor : public AbstractProcessor<L2PProcessor, Args...>
{
    typedef AbstractProcessor<L2PProcessor, Args...> Base;  // define AbstractProcessor as Base
                                                            // Class
    using Base::AbstractProcessor;  // Inherit constructor of Base (Reason: Constructors are not
                                    // inherited by default!)
    template <typename DataType>
    __device__ void run_computation(DataType data)
    {
        accumulation::l2p(data, this->dm_, this->fmm_handle_);
    }
};

#endif

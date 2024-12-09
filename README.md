This repository contains the source code and experimental framework for porting the GPU-tasking framework GPU-Eventify from CUDA to SYCL. The primary objective is to extend GPU-Eventify's capabilities for heterogeneous device support while maintaining portability, performance, and correctness.

> This work represents a preliminary exploration, developed and tested on SLURM-managed jobs on a university compute cluster. The code is rough and not fully optimized, with findings being tentative. Notably, challenges were encountered, particularly regarding the lack of robust synchronization primitive support in SYCL.


GPU-Eventify is based on:
- Tasks: Encapsulate parallel operations on data.
- User-Defined Dependency Structures: Represent and manage task dependencies.
- Work-Sharing Task Queues: Manage ready-to-execute tasks using efficient spin locks.
#### SYCL Porting Highlights
The CUDA-to-SYCL porting process involved:
- Mapping memory and kernel models between CUDA and SYCL.
- Adapting task queues and synchronization mechanisms using SYCL primitives.
- Implementing Single-Producer Multi-Consumer (SPMC) queues for work-sharing.

#### Synchronisation Mechanisms
- SYCL primitives such as `atomic_ref` and `group_barrier` were utilized.
- Forward progress challenges and device-specific behavior required a refined approach to locks and barriers.


A parallel vector addition benchmark was implemented to evaluate the overhead introduced by SYCL tasking mechanisms:
- Basic SYCL Vector Addition: A single parallel_for kernel for baseline comparison.
- Single Global Queue: Managed task queuing with sequential task enqueuing, execution, and dequeuing kernels.
- Multiple Global Queues: Work was distributed across multiple queues, improving parallelism.

#### Metrics
- Profiling with SYCL event objects.
- Host-side runtime measurements using high_resolution_clock.

#### Challenges and Limitations
- Synchronization: Lack of device-wide synchronization mechanisms in SYCL introduced challenges for robust lock implementations.
- Performance: Tasking overhead was significant in fine-grained scenarios (e.g., vector addition), highlighting areas for optimization.
- Portability: Variability in forward progress guarantees across SYCL implementations may affect consistency.

#### Future Work
- Refine synchronization mechanisms for better performance across heterogeneous devices.
- Expand support for more complex tasking and dependency resolution schemes.
- Investigate further optimizations for SYCL's tasking primitives.

#### References
- SYCL Specification: https://www.khronos.org/sycl
- CUDA Programming Guide: https://docs.nvidia.com/cuda/

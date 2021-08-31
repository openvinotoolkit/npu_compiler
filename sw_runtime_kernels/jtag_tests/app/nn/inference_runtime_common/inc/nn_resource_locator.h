/*
* {% copyright %}
*/
#ifndef NN_RESOURCE_LOCATOR_H_
#define NN_RESOURCE_LOCATOR_H_

#include <nn_inference_runtime_types.h>
#include <nn_cmx_memory_map.h>
#include <nn_relocation.h>
#include <nn_resources.h>
#include <nn_math.h>
#include <array>

namespace nn
{
    namespace inference_runtime
    {
        template <typename Task>
        class TaskLocator
        {
        public:
            TaskLocator() :
                tasks_(nullptr),
                count_(0)
            {
            }

            TaskLocator(uint32_t addr, unsigned int size) : TaskLocator(reinterpret_cast<void*>(addr), size) {}

            TaskLocator(void *addr, unsigned int size) :
                tasks_(reinterpret_cast<Task *>(math::ptr_align_up<alignof(Task)>(addr))),
                count_(round_down_to_power_of_2((size - static_cast<unsigned int>(reinterpret_cast<char *>(tasks_) - reinterpret_cast<char *>(addr))) / sizeof(Task)))
            {
                nnLog(MVLOG_DEBUG, "%p, %u -> %p, %u", addr, size, tasks_, count_);
            }

            inline Task &task(unsigned int i) const { return tasks_[i & (count_ - 1)]; }
            inline Task *tasks() const { return tasks_; }
            inline unsigned int count() const { return count_; }

        private:
            Task *tasks_;
            unsigned int count_;

            static inline unsigned int round_down_to_power_of_2(unsigned int x)
            {
                if (x == 0)
                    return 0;

                return 1u << math::lastBitIndex(x);
            }
        };

        typedef TaskLocator<backend::DMATask> DMALocator;
        typedef TaskLocator<backend::DPUInvariantWrapper> InvariantLocator;
        typedef TaskLocator<backend::DPUVariantWrapper> VariantLocator;

        struct StaticMapping
        {
            StaticMapping() = default;
            StaticMapping(NNCmxMemoryMap *cmx);

            std::array<Buffer, MAX_CLUSTERS> workareas_;
            std::array<Buffer, MAX_CLUSTERS> metadataStorage_;
            std::array<Buffer, MAX_DMA_ENGINES> dmaStorage_;
        };

        struct RuntimeMapping
        {
            RuntimeMapping();
            RuntimeMapping(const StaticMapping &global, ClusterMapper::Config config);

            std::array<DMALocator, MAX_DMA_ENGINES> dma_;
            InvariantLocator inv_;
            VariantLocator var_;
            ClusterMapper::Config config_;
            std::array<unsigned char, MAX_CLUSTERS> fifos_;

        private:
            enum
            {
                INVARIANT_COUNT = 64,
                VARIANT_COUNT = 512,
            };
        };
    }
}

#endif // NN_RESOURCE_LOCATOR_H_

//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once
namespace nn {
namespace common_runtime {

enum : unsigned int {
    // For Std FW, only use 1 thread for 1/2 tile execution.
    // For standalone tests, use 2 threads to support parallel tile 0/1 execution

#ifndef CONFIG_VALIDATION_APP_ENABLED
    INF_THREADS_COUNT = 1,
#else
    INF_THREADS_COUNT = 2,
#endif
    IR_WORKER_COUNT = INF_THREADS_COUNT,

    // TODO: Change this once multi-substreamID architecture is defined
    // For now just use one worker for simplicity

    // variant, invariant, act invo
    NUM_COMPONENT_FEEDERS = 3,

    // Two DMA + components
    NUM_METADATA_FEEDERS = MAX_DMA_ENGINES + NUM_COMPONENT_FEEDERS,

    // Dual tile blob can use all 64 barriers. Single tile blob can only use 32
    // (since it may be running in parallel)
    MAX_BARRIERS_PER_INFERENCE = 64,

//    // See docs/ReadMe.md : Windowing and Memory Layout
//    ACT_CMX_WINDOW = 0x1F000000,
//    ACT_RT_CODE_WINDOW = 0x1C000000,
//    ACT_KERNEL_CODE_WINDOW = 0x1D000000,
//    ACT_KERNEL_DATA_WINDOW = 0x1E000000
};
// See docs/ReadMe.md : Windowing and Memory Layout
constexpr static uint32_t ACT_CMX_WINDOW{0x1F000000};
constexpr static uint32_t ACT_RT_CODE_WINDOW{0x1C000000};
constexpr static uint32_t ACT_KERNEL_CODE_WINDOW{0x1D000000};
constexpr static uint32_t ACT_KERNEL_DATA_WINDOW{0x1E000000};

// ShaveNN window layout
constexpr static uint32_t SNN_RT_CODE_WINDOW{0x1C000000};
constexpr static uint32_t SNN_RT_DATA_WINDOW{0x1D000000};
constexpr static uint32_t SNN_KERNEL_CODE_WINDOW{0x1E000000};
//constexpr static uint32_t SNN_KERNEL_DATA_WINDOW{0x1E000000};
constexpr static uint32_t SNN_CMX_WINDOW{0x1F000000};

constexpr static uint32_t SNN_DATA_SIZE{1_KB};
constexpr static uint32_t SNN_STACK_SIZE{1_KB};
constexpr static uint32_t ACTSHV_SCRATCH_SIZE{1_KB};
constexpr static uint32_t METADATA_SIZE{64_KB};
constexpr static uint32_t WORKSPACE_SIZE{1942_KB};
constexpr static uint32_t ACTSHV_STACK_SIZE{7_KB};
constexpr static uint32_t DMA_STORAGE_SIZE{32_KB};

// Shave windows must be aligned on a 1K boundary
constexpr static uint32_t SHV_WIN_ALIGN{0x400};

} // namespace common_runtime
} // namespace nn

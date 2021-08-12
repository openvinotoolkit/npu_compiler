//
// Copyright 2020 Intel Corporation.
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

#include "vpux/utils/IE/loop.hpp"

#include <ie_common.h>
#include <ie_parallel.hpp>

using namespace vpux;

StringLiteral vpux::stringifyEnum(LoopExecPolicy val) {
    switch (val) {
    case LoopExecPolicy::Sequential:
        return "Sequential";
    case LoopExecPolicy::Parallel:
        return "Parallel";
    default:
        return "<UNKNOWN>";
    }
}

void vpux::loop_1d(LoopExecPolicy policy, int64_t dim0, FuncRef<void(int64_t)> proc) {
    if (policy == LoopExecPolicy::Parallel) {
        InferenceEngine::parallel_for(dim0, proc);
    } else {
        for (int64_t d0 = 0; d0 < dim0; ++d0) {
            proc(d0);
        }
    }
}

// [Track number: E#17831]
// TODO Investigate performance and potential improvements
void vpux::blocked_loop_1d(LoopExecPolicy policy, size_t dim0, FuncRef<void(size_t, size_t)> proc, size_t blockSize) {
    if (!dim0) {
        IE_THROW() << "Blocked loop - zero size";
    }

    const int tbbThreads = parallel_get_max_threads();
    const size_t defBlocksAmount = tbbThreads > 0 ? tbbThreads : 8;
    if (!blockSize) {
        blockSize = dim0 >= defBlocksAmount ? dim0 / defBlocksAmount : 1;
    }

    const int64_t numBlocks = static_cast<int64_t>(dim0 >= blockSize ? dim0 / blockSize : 1);
    loop_1d(policy, numBlocks, [dim0, blockSize, numBlocks, proc](int64_t blockIndex) {
        const auto startIndex = static_cast<size_t>(blockIndex * blockSize);
        const auto endIndex = static_cast<size_t>(blockIndex < numBlocks - 1 ? startIndex + blockSize - 1 : dim0 - 1);
        proc(startIndex, endIndex);
    });
}

void vpux::loop_2d(LoopExecPolicy policy, int64_t dim0, int64_t dim1, FuncRef<void(int64_t, int64_t)> proc) {
    if (policy == LoopExecPolicy::Parallel) {
        InferenceEngine::parallel_for2d(dim0, dim1, proc);
    } else {
        for (int64_t d0 = 0; d0 < dim0; ++d0) {
            for (int64_t d1 = 0; d1 < dim1; ++d1) {
                proc(d0, d1);
            }
        }
    }
}
void vpux::loop_3d(LoopExecPolicy policy, int64_t dim0, int64_t dim1, int64_t dim2,
                   FuncRef<void(int64_t, int64_t, int64_t)> proc) {
    if (policy == LoopExecPolicy::Parallel) {
        InferenceEngine::parallel_for3d(dim0, dim1, dim2, proc);
    } else {
        for (int64_t d0 = 0; d0 < dim0; ++d0) {
            for (int64_t d1 = 0; d1 < dim1; ++d1) {
                for (int64_t d2 = 0; d2 < dim2; ++d2) {
                    proc(d0, d1, d2);
                }
            }
        }
    }
}

void vpux::loop_4d(LoopExecPolicy policy, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                   FuncRef<void(int64_t, int64_t, int64_t, int64_t)> proc) {
    if (policy == LoopExecPolicy::Parallel) {
        InferenceEngine::parallel_for4d(dim0, dim1, dim2, dim3, proc);
    } else {
        for (int64_t d0 = 0; d0 < dim0; ++d0) {
            for (int64_t d1 = 0; d1 < dim1; ++d1) {
                for (int64_t d2 = 0; d2 < dim2; ++d2) {
                    for (int64_t d3 = 0; d3 < dim3; ++d3) {
                        proc(d0, d1, d2, d3);
                    }
                }
            }
        }
    }
}

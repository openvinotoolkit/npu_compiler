//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

namespace vpux {
namespace VPUMI40XX {
using lcaCache = llvm::DenseMap<std::pair<uint32_t, uint32_t>, llvm::SmallVector<mlir::Value>>;

//
// AddEnqueue Utils
//

bool contains(const llvm::SmallVector<mlir::Value>& vec, const mlir::Value& element);

VPUMI40XX::ConfigureBarrierOp getBarrierOp(mlir::Operation* op);

size_t getBarrierIndex(mlir::Operation* op);

bool taskOpComparator(mlir::Operation* lhs, mlir::Operation* rhs);

void reindexEnqueueOps(llvm::SmallVector<VPURegMapped::EnqueueOp> enquOps);

void dfs(mlir::Value val, llvm::SetVector<mlir::Value>& visited, size_t indexMax);

llvm::SmallVector<mlir::Value> lca(mlir::Value lhs, mlir::Value rhs, lcaCache& cache, size_t indexMax);
llvm::SmallVector<mlir::Value> lca(llvm::SmallVector<mlir::Value>& lhs, mlir::Value rhs, lcaCache& cache,
                                   size_t indexMax);
VPURegMapped::TaskOpInterface getNextOp(VPURegMapped::TaskOpInterface op);

llvm::SmallVector<mlir::Value> getPreviousUsages(mlir::ValueRange barrs);

// TODO: need to figure out a clean way to get barriers purely from taskOpInterface
VPUMI40XX::ExecutableTaskOpInterface getBarrieredOp(VPURegMapped::TaskOpInterface primary,
                                                    VPURegMapped::TaskOpInterface secondary);

struct HwQueueType {
    VPURegMapped::TaskType type;
    uint32_t tile = 0;
    uint32_t index = 0;

    bool operator<(const HwQueueType& other) const {
        if (type == other.type) {
            if (tile == other.tile) {
                return index < other.index;
            }
            return tile < other.tile;
        }
        return type < other.type;
    }
    bool operator==(const HwQueueType& other) const {
        return type == other.type && tile == other.tile && index == other.index;
    }
    bool operator!=(const HwQueueType& other) const {
        return !(*this == other);
    }
};

//
// ConfigureBarrier Utils
//

void setBarrierIDs(mlir::MLIRContext* ctx, mlir::func::FuncOp funcOp);

//
// Log Fetch Tasks
//

struct FetchTaskDetails {
    size_t tileIndex;
    size_t taskIndex;
    size_t dmaWithBarriers;
    size_t barrierIdx;
    size_t primaryStart;
    size_t primaryEnd;
    size_t secondaryStart;
    size_t secondaryEnd;
    std::string taskType;
    size_t executionGroup;
};

VPUMI40XX::NNDMAOp getPreviousDMAWithBarriers(VPURegMapped::TaskOpInterface taskOpInterface);
void logFetchOpsDetails(mlir::func::FuncOp netFunc, Logger log);

struct EnqDmaInfo {
    int64_t startTaskIdx;
    int64_t endTaskIdx;
    VPUMI40XX::NNDMAOp enqDmaOp;
};

mlir::DenseMap<VPUMI40XX::HwQueueType, SmallVector<EnqDmaInfo>> getEnqueueDmaData(
        VPUMI40XX::NNDMAOp firstDmaTile0List0Op, Logger log);

}  // namespace VPUMI40XX
}  // namespace vpux

namespace llvm {
template <>
struct DenseMapInfo<vpux::VPUMI40XX::HwQueueType> {
    static vpux::VPUMI40XX::HwQueueType getEmptyKey() {
        return vpux::VPUMI40XX::HwQueueType{DenseMapInfo<vpux::VPURegMapped::TaskType>::getEmptyKey(), 0, 0};
    }

    static vpux::VPUMI40XX::HwQueueType getTombstoneKey() {
        return vpux::VPUMI40XX::HwQueueType{DenseMapInfo<vpux::VPURegMapped::TaskType>::getTombstoneKey(), 0, 0};
    }

    static unsigned getHashValue(vpux::VPUMI40XX::HwQueueType val) {
        auto h1 = hash_value(val.type);
        auto h2 = hash_value(val.tile);
        auto h3 = hash_value(val.index);

        return static_cast<unsigned>(hash_combine(h1, h2, h3));
    }

    static bool isEqual(vpux::VPUMI40XX::HwQueueType lhs, vpux::VPUMI40XX::HwQueueType rhs) {
        return rhs == lhs;
    }
};
}  // namespace llvm

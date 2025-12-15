//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include <mlir/IR/IRMapping.h>

#include <deque>
#include <iterator>
#include <string>

namespace vpux {

using SWTaskSignature = TaskSignature<VPUIP::SwKernelOp>;

template <class T>
unsigned countTasks(const SmallVector<std::pair<T, unsigned>>& vector) {
    return std::accumulate(vector.begin(), vector.end(), 0, [](const auto& a, const auto& b) {
        return a + b.second;
    });
}

mlir::IntegerType getActShaveProfilingElementType(mlir::MLIRContext* ctx);

class ActShaveProfiler {
public:
    using ProfilingResults = SmallVector<std::pair<mlir::Value, unsigned>>;

    static unsigned getNextBufferId();

    // In case of tests same class may be called several times, so counter will be reused. Not a problem for parser, but
    // for clarity better to reset
    static void resetBufferIdCounter();

private:
    // Return number of SwKernelRun tasks within this SwKernelOp
    VPUIP::DistributedBufferType getDistributedBufferType(unsigned totalElements);
    static unsigned getNumProfiledTasks(VPUIP::SwKernelOp swOp) {
        auto swKernelRunIt = swOp.getBody().getOps<VPUIP::SwKernelRun>();
        return checked_cast<unsigned int>(std::distance(swKernelRunIt.begin(), swKernelRunIt.end()));
    }

public:
    ActShaveProfiler(unsigned clustersNum, mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                     vpux::IndexedSymbolAttr memKindAttr, mlir::func::FuncOp netFunc, vpux::Logger& log,
                     std::shared_ptr<NameUniqifier> uniqifier);
    ~ActShaveProfiler() = default;

    // Create allocation operation representing profiling buffer instance in CMX. If such buffer is full
    // new one needs to be allocated. Type of this alloc is a memref
    mlir::Operation* createAllocationOp(unsigned totalSizeCMXElements, const std::string& location);

    // Insert DMA that will copy profiling buffer instance to proper offset in profiling output once
    // profiling buffer instance is full or there are no more tasks to profile
    mlir::Value copyToDdr(const ProfilingResults& profilingResults, mlir::Operation* cmxMemOp, size_t& currentDDROffset,
                          mlir::BlockArgument& profilingDdrResult);

    // Get a SubView of profiling buffer instance so that given ActShave task is given required chunk of it
    mlir::Value getViewToBuffer(mlir::Operation* currentProfilingBuffer, unsigned profilingSamplesInCMX,
                                int64_t numTasks);

    // Replace a Actshave task with new one that has profiling output set
    mlir::Value replaceOpWithProfiledOp(VPUIP::SwKernelOp origSwTask, mlir::Value profilingBuffer, mlir::Location loc,
                                        VPUIP::SwProfilingMetadataAttr profMeta);

    // Get amount of memory needed to store profiling data of all ActShave tasks in the model
    unsigned getRequiredDdrMemory() const;

    // Go over all SwKernelOps and store required information about those tasks like required size of
    // profiling buffer or size of profiling buffer instances
    void scheduleTask(VPUIP::SwKernelOp swOp);

    // Main function which goes through all identified ActShave ops and based on gathered data recreates
    // those operations to have profiling output with proper slot in profiling buffer instance. When profiling
    // buffer is full it also inserts CMX2DDR DMA and allocates new profiling buffer
    void addProfilingOps(mlir::BlockArgument& profilingDdrResult, SmallVector<mlir::Value>& clusterResults);

private:
    SWTaskSignature getTaskSignature(VPUIP::SwKernelOp swOp) const;

    mlir::Type getTimestampType(int64_t tasksAmount);

    unsigned _clustersNum;
    unsigned _profilingWorkloadSize;
    unsigned _profilingElementSize;
    std::deque<unsigned> _profilingBufferSizes;
    SmallVector<SWTaskSignature> _swTaskSignatures;
    mlir::OpBuilder& _builder;
    mlir::MLIRContext* _ctx;
    mlir::func::FuncOp _netFunc;
    vpux::IndexedSymbolAttr _memKindAttr;
    vpux::Logger& _log;
    std::shared_ptr<NameUniqifier> _uniqifier;
    static inline unsigned uniqBufferId = 0;
};

}  // namespace vpux

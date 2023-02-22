//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include "vpux/compiler/dialect/VPUIP/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <deque>
#include <iterator>

namespace vpux {

unsigned getClustersNumber(VPUIP::NCEClusterTaskOp nceClusterTaskOp);

using NCETaskSignature = TaskSignature<VPUIP::NCEClusterTaskOp>;

// Base class for profiling buffer schedulers
// Algorithm is same for any amount of cluster, difference only in used types/ops
// For numClusters != 1 will be used DistributedBufferType, for single cluster ops memref. See details in
// SingleClusterScheduler
class BaseClusterBufferScheduler {
private:
    mlir::Type getTimestampType(unsigned dpuTasksAmount);

public:
    BaseClusterBufferScheduler(unsigned clustersAmount, unsigned profilingWorkloadSize, mlir::OpBuilder& builder,
                               mlir::MLIRContext* ctx, vpux::VPU::MemoryKind memKind, mlir::FuncOp netFunc,
                               std::shared_ptr<NameUniqifier> uniqifier);

    virtual ~BaseClusterBufferScheduler() = default;

    // Return needed for storing profiling results in DDR memory size in bytes
    unsigned getRequiredDdrMemory() const;

    // Schedule next NCE Task
    void scheduleNceTask(VPUIP::NCEClusterTaskOp nceClusterTaskOp);

    // Add needed for profiling buffers/views/copies
    void addProfilingOps(unsigned& currentDDROffset, SmallVector<mlir::Value>& clusterResults,
                         mlir::BlockArgument& profilingResult, int& nceId);

protected:
    // Region of logic, which depends on amount of clusters. By default operates on distributed types

    virtual NCETaskSignature getTaskSignature(VPUIP::NCEClusterTaskOp nceClusterTaskOp) = 0;

    virtual mlir::Operation* createAllocationOp(unsigned totalSizeCMXElements, const std::string& location) = 0;

    virtual mlir::Value getViewToBuffer(mlir::Operation* currentProfilingBuffer, unsigned, SmallVector<int64_t>) = 0;

    virtual mlir::Value copyToDDR(mlir::BlockArgument& profilingResult, mlir::Operation*,
                                  SmallVector<mlir::Value>& dpuProfilingOutputs, unsigned numElements, unsigned offset,
                                  StringRef name) = 0;

protected:
    unsigned _clustersAmount;
    unsigned _profilingWorkloadSize;
    unsigned _profilingElementSize;
    std::deque<unsigned> _profilingBufferSizes;
    SmallVector<NCETaskSignature> _nceTaskSignatures{};
    mlir::OpBuilder& _builder;
    mlir::MLIRContext* _ctx;
    mlir::FuncOp _netFunc;
    vpux::IndexedSymbolAttr _memKindAttr;
    std::shared_ptr<NameUniqifier> _uniqifier;
};

class SingleClusterScheduler : public BaseClusterBufferScheduler {
public:
    SingleClusterScheduler(unsigned profilingWorkloadSize, mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                           vpux::VPU::MemoryKind memKind, mlir::FuncOp netFunc,
                           std::shared_ptr<NameUniqifier> uniqifier);

protected:
    virtual NCETaskSignature getTaskSignature(VPUIP::NCEClusterTaskOp nceClusterTaskOp) override;

    virtual mlir::Operation* createAllocationOp(unsigned totalSizeCMXElements, const std::string& location) override;

    virtual mlir::Value copyToDDR(mlir::BlockArgument& profilingResult, mlir::Operation* cmxMemOp,
                                  SmallVector<mlir::Value>& dpuProfilingOutputs, unsigned numElements, unsigned offset,
                                  StringRef name) override;

    virtual mlir::Value getViewToBuffer(mlir::Operation* currentProfilingBuffer, unsigned profilingSamplesInCMX,
                                        SmallVector<int64_t> sizes) override;
};

class MultiClusterScheduler : public BaseClusterBufferScheduler {
private:
    VPUIP::DistributedBufferType getDistributedBufferType(unsigned totalElements);

    mlir::Type getDistributedTimestampType(unsigned dpuTasksAmount);

public:
    using BaseClusterBufferScheduler::BaseClusterBufferScheduler;

protected:
    virtual NCETaskSignature getTaskSignature(VPUIP::NCEClusterTaskOp nceClusterTaskOp) override;

    virtual mlir::Operation* createAllocationOp(unsigned totalSizeCMXElements, const std::string& location) override;

    virtual mlir::Value copyToDDR(mlir::BlockArgument& profilingResult, mlir::Operation* cmxMemOp,
                                  SmallVector<mlir::Value>& dpuProfilingOutputs, unsigned numElements, unsigned offset,
                                  StringRef name) override;

    virtual mlir::Value getViewToBuffer(mlir::Operation* currentProfilingBuffer, unsigned profilingSamplesInCMX,
                                        SmallVector<int64_t> sizes) override;
};

}  // namespace vpux

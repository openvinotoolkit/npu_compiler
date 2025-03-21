//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {
namespace VPUIP {

using DmaFusionHandlerType = std::function<void(SmallVector<std::pair<VPUIP::NNDMAOp, bool>>)>;

//
// ClusterNCEBaseRewriter
//

class ClusterNCEBaseRewriter {
public:
    ClusterNCEBaseRewriter(mlir::MLIRContext* ctx, Logger log): _log(log), _ctx(ctx) {
        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    void matchAndRewrite(VPUIP::NCEClusterTaskOp nceTask, mlir::OpBuilder& builder) const;

protected:
    virtual void getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs, SmallVector<mlir::Value>& outputBuffs,
                                  SmallVector<mlir::Value>& parentOutputSparsityMap,
                                  SmallVector<mlir::Value>& outputSparsityMapBuffs,
                                  SmallVector<SmallVector<mlir::Value>>& outputItiBuffs, mlir::Location loc,
                                  VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                                  mlir::OpBuilder& builder) const = 0;

    virtual void getInputBuffers(SmallVector<mlir::Value>& parentInputBuffs, SmallVector<mlir::Value>& inputBuffs,
                                 SmallVector<mlir::Value>& parentInputSparsityMap,
                                 SmallVector<mlir::Value>& inputSparsityMapBuffs,
                                 SmallVector<mlir::Value>& parentInputSETable,
                                 SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
                                 VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                                 mlir::OpBuilder& builder) const = 0;

    virtual SmallVector<mlir::Value> getWeightsBuffers(mlir::Location loc, VPUIP::NCEClusterTaskOp nceTask,
                                                       const int64_t numClusters, mlir::OpBuilder& builder) const;

    virtual mlir::UnitAttr isSegmentedNCETask(VPUIP::DistributedBufferType inputType) const = 0;

private:
    SmallVector<mlir::IntegerAttr> getOutChannelOffsets(VPUIP::NCEClusterTaskOp nceTask,
                                                        VPUIP::DistributedBufferType inType,
                                                        VPUIP::DistributedBufferType outType) const;

protected:
    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

//
// ClusterPerElementDMABaseRewriter
//

class ClusterPerElementDMABaseRewriter {
public:
    ClusterPerElementDMABaseRewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount,
                                     std::optional<DmaFusionHandlerType> maybeFusionHandler, Logger log)
            : _log(log), _ctx(ctx), _dmaPortCount(dmaPortCount), _maybeFusionHandler(std::move(maybeFusionHandler)) {
        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    void matchAndRewrite(VPUIP::DMATypeOpInterface dmaOp, mlir::OpBuilder& builder,
                         bool isDataOverlapped = false) const;

protected:
    enum UnrollingType { FAILED, SEGMENTED, DUPLICATED };
    virtual bool isTargetOp(VPUIP::DMATypeOpInterface dmaOp) const = 0;
    virtual VPUIP::DMATypeOpInterface wrapIntoTaskOp(VPUIP::DMATypeOpInterface dmaOp, VPURT::TaskOp vpurtTask,
                                                     mlir::Location loc, mlir::Value input, mlir::Value output_buff,
                                                     int64_t port, mlir::OpBuilder& builder) const = 0;
    virtual UnrollingType getUnrollingType(VPU::DistributionMode inputMode, VPU::DistributionMode outputMode) const = 0;

private:
    void unrollSegmentedOrOverlapped(mlir::Location loc, VPURT::TaskOp vpurtTask, mlir::OpBuilder& builder,
                                     bool isDataOverlapped) const;
    void unrollDuplicated(mlir::Location loc, VPURT::TaskOp vpurtTask, mlir::OpBuilder& builder) const;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    int64_t _dmaPortCount;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
    std::optional<DmaFusionHandlerType> _maybeFusionHandler;
};

//
// ClusterDMARewriter
//

class ClusterDMARewriter final : public ClusterPerElementDMABaseRewriter {
public:
    ClusterDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount,
                       std::optional<DmaFusionHandlerType> maybeFusionHandler, Logger log)
            : ClusterPerElementDMABaseRewriter(ctx, dmaPortCount, std::move(maybeFusionHandler), log) {
    }

private:
    bool isTargetOp(VPUIP::DMATypeOpInterface dmaOp) const override;
    VPUIP::DMATypeOpInterface wrapIntoTaskOp(VPUIP::DMATypeOpInterface dmaOp, VPURT::TaskOp vpurtTask,
                                             mlir::Location loc, mlir::Value input, mlir::Value output_buff,
                                             int64_t port, mlir::OpBuilder& builder) const override;
    UnrollingType getUnrollingType(VPU::DistributionMode inputMode, VPU::DistributionMode outputMode) const override;
};

void unrollClusterTilingCommon40XXPlus(mlir::func::FuncOp funcOp,
                                       std::optional<DmaFusionHandlerType> maybeDmaFusionHandler, vpux::Logger log);

}  // namespace VPUIP
}  // namespace vpux

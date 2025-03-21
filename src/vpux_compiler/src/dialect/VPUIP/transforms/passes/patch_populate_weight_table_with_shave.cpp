//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_PATCHPOPULATEWEIGHTTABLEWITHSHAVE
#define GEN_PASS_DEF_PATCHPOPULATEWEIGHTTABLEWITHSHAVE
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// PatchPopulateWeightTableWithShavePass
//

class PatchPopulateWeightTableWithShavePass final :
        public VPUIP::impl::PatchPopulateWeightTableWithShaveBase<PatchPopulateWeightTableWithShavePass> {
public:
    explicit PatchPopulateWeightTableWithShavePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    uint64_t getPointer(mlir::Value value, uint64_t defaultValue);
    SmallVector<int32_t> getWeightsPerClusterPointers(VPUIP::NCEClusterTaskOp nceOp, int64_t numCluster);
    void patchShaveForPopulateWeightTable(VPUIP::SwKernelOp swKernelOp, VPUIP::NCEClusterTaskOp nceOp,
                                          VPURT::DeclareBufferOp wtDecBuf);
    mlir::Value getRootBuffer(mlir::Value val);
    int32_t getWeightsAddressOffsetForSubView(VPUIP::NCEClusterTaskOp nceOp);
    mlir::Operation* getShaveKernelOpForDstBuffer(mlir::Value prodBuffer, mlir::Value dstBuffer);
};

mlir::Value PatchPopulateWeightTableWithShavePass::getRootBuffer(mlir::Value val) {
    vpux::ValueSourceInfo aliasInfo(val);
    auto rootBuffers = aliasInfo.getRoots(val);
    VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value expected to have only one root. Got {0}", rootBuffers.size());
    return *rootBuffers.begin();
}

//
// safeRunOnFunc
//

void PatchPopulateWeightTableWithShavePass::safeRunOnFunc() {
    auto funcOp = getOperation();
    // For each nceOp.weight_table find related DeclareBufferOp. Next find dmaOp which
    // fills the buffer. DmaOp's input is expected to be Const::DeclareOp which
    // should be modified by adding relocateWeightTable transformation.
    funcOp.walk([this](vpux::VPUIP::NCEClusterTaskOp nceOp) {
        // Don't need to run weights table patching if operation has zeroOffset.
        // Patching is done earlier in compilation by RelocateWeightTableForReuse pass.
        if (nceOp.getIsZeroOffsetWeightsTable()) {
            return;
        }
        auto wTable = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.getWeightTable());
        if (wTable == nullptr) {
            return;
        }
        _log.trace("WeightTable identified for operation '{0}'", nceOp->getLoc());

        auto rootBuffer = getRootBuffer(wTable);
        auto wtDecBuf = rootBuffer.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(wtDecBuf != nullptr, "DeclareBufferOp expected as a weight_table parent");

        // Get Shave operation that populates weights table for NCE Task
        auto* cstLoadOp = getShaveKernelOpForDstBuffer(wtDecBuf.getResult(), wtDecBuf.getResult());
        if (cstLoadOp == nullptr) {
            return;
        }

        auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(cstLoadOp);
        VPUX_THROW_UNLESS(swKernelOp != nullptr, "Got invalid WT population op");

        // special condition for shave offset population, also consider multiple kernel runs
        _log.trace("Shave weight table population '{0}'", swKernelOp->getLoc());
        patchShaveForPopulateWeightTable(swKernelOp, nceOp, wtDecBuf);
        nceOp->setAttr(vpux::VPUIP::populateWeightTableWithShave, mlir::BoolAttr::get(nceOp.getContext(), true));
    });
}

// Find a Shave operation that writes data into a given buffer
mlir::Operation* PatchPopulateWeightTableWithShavePass::getShaveKernelOpForDstBuffer(mlir::Value prodBuffer,
                                                                                     mlir::Value dstBuffer) {
    for (auto* user : prodBuffer.getUsers()) {
        if (auto subView = mlir::dyn_cast<VPUIP::SubViewOp>(user)) {
            // case for tiled Shave on the same cluster
            return getShaveKernelOpForDstBuffer(subView.getResult(), dstBuffer);
        }
        // propagate through spill_write->spill_read chain
        if (auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(user)) {
            auto value = dmaOp.getInput();
            vpux::ValueSourceInfo aliasInfo(value);
            auto rootBuffers = aliasInfo.getRoots(value);
            VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value expected to have only one root. Got {1}",
                              rootBuffers.size());
            auto staticAlloc = *rootBuffers.begin();
            VPUX_THROW_WHEN(staticAlloc == nullptr, "Can't find dst buffer for spill write");
            auto spillWrites = to_vector(staticAlloc.getUsers());
            VPUX_THROW_UNLESS(spillWrites.size() > 0, "Must be non 0 users");
            auto writeDma = mlir::dyn_cast<VPUIP::NNDMAOp>(spillWrites.front());
            auto realSource = writeDma.getInput();
            for (auto bufferUser : realSource.getUsers()) {
                if (mlir::isa<VPUIP::SubViewOp>(bufferUser)) {
                    auto firstUser = to_vector(bufferUser->getUsers()).front();
                    if (auto shaveOp = mlir::dyn_cast<VPUIP::SwKernelOp>(firstUser)) {
                        return shaveOp;
                    }
                }
            }
            return nullptr;
        }

        auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user);
        if (nceClustOp != nullptr && mlir::isa<VPUIP::SwKernelOp>(nceClustOp.getInnerTaskOp())) {
            if (getRootBuffer(nceClustOp.getOutputs()[0]) == dstBuffer) {
                return nceClustOp.getOperation();
            }
        }

        if (auto swOp = mlir::dyn_cast<VPUIP::SwKernelOp>(user)) {
            if (getRootBuffer(swOp.getOutputBuffs()[0]) == dstBuffer) {
                return swOp.getOperation();
            }
        }
    }
    return nullptr;
}

void PatchPopulateWeightTableWithShavePass::patchShaveForPopulateWeightTable(VPUIP::SwKernelOp swKernelOp,
                                                                             VPUIP::NCEClusterTaskOp nceOp,
                                                                             VPURT::DeclareBufferOp wtDecBuf) {
    // find offsets due to Shave tiling and cluster tiling
    const auto weightAddressOffset = getWeightsAddressOffsetForSubView(nceOp);
    SmallVector<int64_t> swKernelRunOffsets;
    // find offsets for inner sw kernel runs
    for (auto outBuff : swKernelOp.getOutputBuffs()) {
        const auto outTopBuff = VPUIP::getTopBufferOfNCEClusterTiling(swKernelOp, outBuff);
        _log.trace("SW outTopBuff {0}", outTopBuff);
        if (auto subView = mlir::dyn_cast<VPUIP::SubViewOp>(outTopBuff.getDefiningOp())) {
            _log.trace("SV {0}", subView);
            const auto offsetsAttr = subView.getStaticOffsets();
            auto offsets = parseIntArrayAttr<int32_t>(offsetsAttr);
            _log.trace("offset {0}", offsets);
            swKernelRunOffsets.push_back(offsets[0] * weightAddressOffset);
        } else {
            swKernelRunOffsets.push_back(0);
        }
    }

    auto swKernelRun = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(
            checked_cast<size_t>(std::distance(swKernelRun.begin(), swKernelRun.end())) == swKernelRunOffsets.size(),
            "Failed to generate offset for every SwKernelRun");

    auto distributedType = wtDecBuf.getType().dyn_cast<VPUIP::DistributedBufferType>();
    int64_t numClusters = 1;
    if (distributedType != nullptr) {
        const auto distributionAttr = distributedType.getDistribution();
        numClusters = distributionAttr.getNumClusters().getInt();
    }
    auto weightsPerClusterPtrs = getWeightsPerClusterPointers(nceOp, numClusters);
    const auto baseOffset = weightsPerClusterPtrs[0];

    // patch base offset for each kernel run
    for (auto entry : swKernelRun | indexed) {
        auto attrs = entry.value().getAttrs().value();
        SmallVector<mlir::Attribute> newAttrs(attrs.begin(), attrs.end());
        // use base offset + kernel run offset with Shave tiling case
        const auto offsetAttr = getIntAttr(swKernelOp->getContext(), baseOffset + swKernelRunOffsets[entry.index()]);
        newAttrs[0] = offsetAttr;
        entry.value().setAttrsAttr(mlir::ArrayAttr::get(swKernelOp->getContext(), newAttrs));
        _log.trace("Updated base offset to {0}", offsetAttr);
    }

    // patch per cluster offsets during unrolling
    if (distributedType != nullptr) {
        // store offsets per cluster to add to base offset
        for (auto& offset : weightsPerClusterPtrs) {
            offset -= baseOffset;
        }
        const auto weightsPerClusterPtrsAttr = getIntArrayAttr(swKernelOp.getContext(), weightsPerClusterPtrs);
        swKernelOp->setAttr(vpux::VPUIP::weightsPtrsPerClusterAttr, weightsPerClusterPtrsAttr);
    }

    _log.trace("patched shave {0}", swKernelOp);
}

uint64_t PatchPopulateWeightTableWithShavePass::getPointer(mlir::Value value, uint64_t defaultValue) {
    if (value == nullptr) {
        return defaultValue;
    }
    vpux::ValueSourceInfo aliasInfo(value);
    auto rootBuffers = aliasInfo.getRoots(value);
    VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value expected to have only one root. Got {1}", rootBuffers.size());
    const auto rootBuffer = *rootBuffers.begin();

    auto valueDeclareBuffer = mlir::dyn_cast<VPURT::DeclareBufferOp>(rootBuffer.getDefiningOp());
    if (valueDeclareBuffer == nullptr) {
        return defaultValue;
    }
    return valueDeclareBuffer.getByteOffset();
}

SmallVector<int32_t> PatchPopulateWeightTableWithShavePass::getWeightsPerClusterPointers(VPUIP::NCEClusterTaskOp nceOp,
                                                                                         int64_t numCluster) {
    auto weights = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.getWeights());
    uint64_t weightsBasePointer = getPointer(weights, 0);

    if (weights == nullptr || !weights.getType().isa<VPUIP::DistributedBufferType>()) {
        return SmallVector<int32_t>(numCluster, static_cast<int32_t>(weightsBasePointer));
    }
    auto distributedType = weights.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto distributionAttr = distributedType.getDistribution();
    auto mode = distributionAttr.getMode().getValue();
    if (mode != (VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED)) {
        // For distribution modes except Duplicated|Segmented, the weights on every cluster is supposed to be symmetric.
        // So just return the base pointer for each weights
        return SmallVector<int32_t>(numCluster, static_cast<int32_t>(weightsBasePointer));
    }

    // For Duplicated|Segmented weights, weights should be non const for ops like MatMul. It's supposed to
    // be segmented on N, while the entire data is copied to all CMXs. So the weight's offset need to updated
    // accordingly since it's not symmetric.
    SmallVector<int32_t> weightsPerClusterPtrs;
    VPUX_THROW_UNLESS(nceOp.getWeightsSparsityMap() == nullptr,
                      "Weight sparisity map is found, weights is supposed to be non const at '{0}'", nceOp->getLoc());
    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
    VPUX_THROW_UNLESS(axis == Dims4D::Act::N.ind(), "Invalid Tile dim, get {0}, expect tiling on N.", axis);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    for (auto clusterOffsets : perClusterShapeOffsets) {
        auto perClusterOffset = Byte(clusterOffsets[Dim(axis)] * distributedType.getStrides()[Dim(axis)]).count();
        weightsPerClusterPtrs.push_back(static_cast<int32_t>(weightsBasePointer + perClusterOffset));
    }
    return weightsPerClusterPtrs;
}

int32_t PatchPopulateWeightTableWithShavePass::getWeightsAddressOffsetForSubView(VPUIP::NCEClusterTaskOp nceOp) {
    auto weights = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.getWeights());
    VPUX_THROW_UNLESS(weights != nullptr, "Failed to find weights");

    auto weightsType = weights.getType().dyn_cast<NDTypeInterface>();
    auto totalAllocSize = weightsType.getTotalAllocSize().count();
    auto outputChannels = weightsType.getShape()[Dims4D::Filter::OC];

    if (auto distWeightsType = weightsType.dyn_cast<VPUIP::DistributedBufferType>()) {
        outputChannels = distWeightsType.getPerClusterComputeShapes()[0][Dims4D::Filter::OC];
    }
    VPUX_THROW_UNLESS(totalAllocSize % outputChannels == 0, "Unequal division");
    return totalAllocSize / outputChannels;
}
}  // namespace

//
// createPatchPopulateWeightTableWithShavePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createPatchPopulateWeightTableWithShavePass(Logger log) {
    return std::make_unique<PatchPopulateWeightTableWithShavePass>(log);
}

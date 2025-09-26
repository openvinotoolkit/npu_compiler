//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/dense_map.hpp"

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_COMPUTEHALOREGIONFORDPUTASKOP
#define GEN_PASS_DEF_COMPUTEHALOREGIONFORDPUTASKOP
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;
using namespace VPUIP;

namespace {

bool isHaloInCurrentWorkload(ArrayRef<int64_t> haloStart, ArrayRef<int64_t> haloEnd, ArrayRef<int64_t> workloadStart,
                             ArrayRef<int64_t> workloadEnd) {
    const auto numDims = haloStart.size();

    for (size_t dim = 0; dim < numDims; dim++) {
        if (haloStart[dim] > workloadEnd[dim] || haloEnd[dim] < workloadStart[dim]) {
            return false;
        }
    }

    return true;
}

DenseMap<mlir::Attribute, NCEClusterTaskOp> getTargetNCEOpForEachOutwardHalo(VPUIP::ITIBufferType outputType,
                                                                             mlir::OperandRange outputItiBuffs) {
    DenseMap<mlir::Attribute, NCEClusterTaskOp> outwardHaloDstNCEOpMap;
    const auto outwardHalos = outputType.getOutwardHaloRegions();

    // Returns the NCE op that produces the value passed to it
    auto getProducerNCEOp = [](mlir::Value output) -> NCEClusterTaskOp {
        for (const auto& userOp : output.getUsers()) {
            auto nceOp = mlir::dyn_cast<NCEClusterTaskOp>(userOp);
            if (nceOp == nullptr) {
                continue;
            }

            if (nceOp.getOutputBuff() == output) {
                return nceOp;
            }
        }

        return nullptr;
    };

    for (const auto& outwardHalo : outwardHalos) {
        auto outputIti = llvm::find_if(outputItiBuffs, [&outwardHalo](mlir::Value itiOutput) {
            // An outward halo with multiple inward halos is used to describe the scenario where
            // the halo is broadcasted to multiple clusters. That means the parameters of the
            // halo region will all be the same for all the consumer clusters of the halo.
            // As such, we need to only find one of the ITI Buffs and NCE ops to compute the halo
            // region params.
            const auto firstInwardHalo =
                    mlir::cast<vpux::VPUIP::HaloRegionAttr>(*outwardHalo.getInwardHaloRegions().begin());
            auto itiType = mlir::cast<vpux::VPUIP::ITIBufferType>(itiOutput.getType());
            auto inwardHaloRegions = itiType.getInwardHaloRegions();
            return llvm::find(inwardHaloRegions, firstInwardHalo) != inwardHaloRegions.end();
        });

        VPUX_THROW_UNLESS(outputIti != outputItiBuffs.end(),
                          "Outward halo is not associated with any output iti buffer");

        const auto neighbourItiOp = getProducerNCEOp(*outputIti);
        VPUX_THROW_UNLESS(neighbourItiOp != nullptr,
                          "Could not find NCEClusterTaskOp producer for output_ITI buff: {0}", *outputIti);

        outwardHaloDstNCEOpMap.insert(std::make_pair(mlir::cast<mlir::Attribute>(outwardHalo), neighbourItiOp));
    }

    return outwardHaloDstNCEOpMap;
}

void computeHaloRegion(NCEClusterTaskOp nceOp, DPUTaskOp dpuTaskOp,
                       DenseMap<mlir::Attribute, NCEClusterTaskOp>& outwardHaloDstNCEOpMap, Logger log) {
    std::unordered_map<vpux::Dim, int64_t> workloadIndex = {{vpux::Dims4D::Act::W, 0},
                                                            {vpux::Dims4D::Act::H, 1},
                                                            {vpux::Dims4D::Act::C, 2}};

    mlir::OpBuilder builder(dpuTaskOp);
    auto ctx = builder.getContext();
    auto outputBuffer = nceOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();
    auto outputType = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(nceOp.getOutput().getType());

    auto srcLayout = mlir::cast<vpux::NDTypeInterface>(nceOp.getInput().getType()).getDimsOrder();
    auto dstLayout = mlir::cast<vpux::NDTypeInterface>(nceOp.getOutput().getType()).getDimsOrder();
    std::unordered_map<vpux::Dim, vpux::Dim> dimMapping;
    if (srcLayout != dstLayout) {
        for (size_t i = 0; i < dstLayout.numDims(); ++i) {
            dimMapping[srcLayout.dimAt(i)] = dstLayout.dimAt(i);
        }
    }

    auto getAdjustedDim = [&](vpux::Dim dim) {
        return dimMapping.empty() ? dim : dimMapping[dim];
    };

    int64_t srcSparseMapOffset = 0;
    if (nceOp.getOutputSparsityMapBuff() != nullptr) {
        auto outputSMBuffer = nceOp.getOutputSparsityMapBuff().getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(outputSMBuffer != nullptr, "Defining op of src sparsity map is not a DeclareBufferOp: {0}",
                          nceOp.getOutputSparsityMapBuff());

        srcSparseMapOffset = outputSMBuffer.getByteOffset();
    }

    SmallVector<mlir::Attribute> haloRegions;

    for (const auto& outwardHalo : outputType.getOutwardHaloRegions()) {
        auto dstNCEOp = outwardHaloDstNCEOpMap[mlir::cast<mlir::Attribute>(outwardHalo)];
        auto dstItiValue = dstNCEOp.getOutputBuff();
        auto dstItiBuffer = dstItiValue.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(dstItiBuffer != nullptr, "Defining op of dst NCE op is not a DeclareBufferOp: {0}",
                          dstItiValue);

        const auto outwardHaloShape = parseIntArrayAttr<int64_t>(outwardHalo.getShape());
        const auto outwardHaloOffset = parseIntArrayAttr<int64_t>(outwardHalo.getOffset());

        const auto currentWorkloadStart = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutStart());
        const auto currentWorkloadEnd = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutEnd());

        const auto haloStart =
                SmallVector<int64_t>{outwardHaloOffset[Dims4D::Act::W.ind()], outwardHaloOffset[Dims4D::Act::H.ind()],
                                     outwardHaloOffset[Dims4D::Act::C.ind()]};

        const auto haloEnd = SmallVector<int64_t>{
                haloStart[workloadIndex[vpux::Dims4D::Act::W]] + outwardHaloShape[Dims4D::Act::W.ind()] - 1,
                haloStart[workloadIndex[vpux::Dims4D::Act::H]] + outwardHaloShape[Dims4D::Act::H.ind()] - 1,
                haloStart[workloadIndex[vpux::Dims4D::Act::C]] + outwardHaloShape[Dims4D::Act::C.ind()] - 1};

        // If the DPUTasks split the full NCE op over the dimension that is being halo'd, then for H & W dims, we might
        // have some DPUTaks that do not need to produce a halo.
        // For SOK, we halo the whole output channels. Therefore, even if output volume is split in
        // multiple workloads, we'll need to send a halo from each DPUTask op, anyway.
        // E.g.
        // Current NCE op needs to send a halo of size 1 over height from height offset 31
        // output tensor computed in current cluster = 1xCx32xW
        // dpu tasks:
        //     DPUTask0 = {outStart = [0, 0, 0], outEnd = [W - 1, 15, C - 1]} => does not compute height 31, no halo
        //     needed
        //     DPUTask1 = {outStart = [0, 16, 0], outEnd = [W - 1, 31, C - 1]} => computes height 31, will need
        //     halo region defined
        if (!isHaloInCurrentWorkload(haloStart, haloEnd, currentWorkloadStart, currentWorkloadEnd)) {
            continue;
        }

        const auto& dstItiType = mlir::cast<vpux::VPUIP::ITIBufferType>(dstItiBuffer.getBuffer().getType());

        const auto firstInwardHalo =
                mlir::cast<vpux::VPUIP::HaloRegionAttr>(*outwardHalo.getInwardHaloRegions().begin());
        const auto dstItiOffset = parseIntArrayAttr<int64_t>(firstInwardHalo.getOffset());

        int64_t targetOffset = dstItiBuffer.getByteOffset() - outputBuffer.getByteOffset();

        int64_t sparsityOffset = 0;
        if (dstNCEOp.getOutputSparsityMapBuff() != nullptr) {
            auto dstOutSMBuffer = dstNCEOp.getOutputSparsityMapBuff().getDefiningOp<VPURT::DeclareBufferOp>();
            VPUX_THROW_UNLESS(dstOutSMBuffer != nullptr,
                              "Defining op of dst sparsity map is not a DeclareBufferOp: {0}",
                              dstNCEOp.getOutputSparsityMapBuff());

            VPUX_THROW_UNLESS(
                    nceOp.getOutputSparsityMapBuff() != nullptr,
                    "Target NCEOp has sparse output, but halo producer NCEOp does not, target: {0}, producer {1}",
                    dstNCEOp, nceOp);

            sparsityOffset = dstOutSMBuffer.getByteOffset() - srcSparseMapOffset;
        }

        const auto numBitsInByte = Byte(1).to<Bit>().count();

        // Need to apply offsets only for height/width
        const auto srcHaloHeightStart = outwardHaloOffset[getAdjustedDim(Dims4D::Act::H).ind()];
        const auto dstHaloHeightStart = dstItiOffset[getAdjustedDim(Dims4D::Act::H).ind()];
        const auto srcHaloWidthStart = outwardHaloOffset[getAdjustedDim(Dims4D::Act::W).ind()];
        const auto dstHaloWidthStart = dstItiOffset[getAdjustedDim(Dims4D::Act::W).ind()];
        const int64_t offset = dstHaloHeightStart * dstItiType.getStrides()[getAdjustedDim(Dims4D::Act::H)].count() +
                               dstHaloWidthStart * dstItiType.getStrides()[getAdjustedDim(Dims4D::Act::W)].count() -
                               srcHaloHeightStart * dstItiType.getStrides()[getAdjustedDim(Dims4D::Act::H)].count() -
                               srcHaloWidthStart * dstItiType.getStrides()[getAdjustedDim(Dims4D::Act::W)].count();

        targetOffset += offset / numBitsInByte;
        sparsityOffset += offset / outputType.getElemTypeSize().count() / numBitsInByte;

        SmallVector<int64_t> targetClustersVec;
        for (auto& inHaloAttr : outwardHalo.getInwardHaloRegions()) {
            const auto inwardHalo = mlir::cast<vpux::VPUIP::HaloRegionAttr>(inHaloAttr);
            targetClustersVec.push_back(inwardHalo.getClusterId().getInt());
        }

        const int64_t targetWidth = dstItiType.getStrides()[getAdjustedDim(Dims4D::Act::H)].count() /
                                    dstItiType.getStrides()[getAdjustedDim(Dims4D::Act::W)].count();

        const auto sparsityOffsetAttr =
                nceOp.getOutputSparsityMapBuff() != nullptr ? builder.getI64IntegerAttr(sparsityOffset) : nullptr;

        const auto dpuHaloRegion = VPUIP::DPUHaloRegionAttr::get(
                ctx, builder.getI64IntegerAttr(haloStart[workloadIndex[getAdjustedDim(vpux::Dims4D::Act::W)]]),
                builder.getI64IntegerAttr(haloEnd[workloadIndex[getAdjustedDim(vpux::Dims4D::Act::W)]]),
                builder.getI64IntegerAttr(haloStart[workloadIndex[getAdjustedDim(vpux::Dims4D::Act::H)]]),
                builder.getI64IntegerAttr(haloEnd[workloadIndex[getAdjustedDim(vpux::Dims4D::Act::H)]]),
                builder.getI64IntegerAttr(haloStart[workloadIndex[getAdjustedDim(vpux::Dims4D::Act::C)]]),
                builder.getI64IntegerAttr(haloEnd[workloadIndex[getAdjustedDim(vpux::Dims4D::Act::C)]]),
                builder.getI64IntegerAttr(targetOffset), getIntArrayAttr(ctx, targetClustersVec), sparsityOffsetAttr,
                builder.getI64IntegerAttr(targetWidth));
        haloRegions.push_back(dpuHaloRegion);
    }

    auto newDpuTask = builder.create<DPUTaskOp>(
            nceOp.getLoc(), dpuTaskOp.getOutStart(), dpuTaskOp.getOutEnd(), dpuTaskOp.getInStartAttr(),
            dpuTaskOp.getInEndAttr(), dpuTaskOp.getPad(), dpuTaskOp.getMpeMode(), dpuTaskOp.getClusterIdAttr(),
            builder.getArrayAttr(haloRegions), dpuTaskOp.getWorkloadIdAttr(), /*variant_primitive_id =*/nullptr);

    log.trace("Computed halo regions for DPUTaskOp '{0}': halo regions = {1}", newDpuTask.getLoc(), haloRegions);

    dpuTaskOp.erase();
}

//
// Compute Halo Region for each VPUIP.DPUTaskOps
//

class ComputeHaloRegionForDPUTaskOpPass final :
        public VPUIP::arch40xx::impl::ComputeHaloRegionForDPUTaskOpBase<ComputeHaloRegionForDPUTaskOpPass> {
public:
    explicit ComputeHaloRegionForDPUTaskOpPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ComputeHaloRegionForDPUTaskOpPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto tileOp = config::getTileExecutor(module);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE executor information");

    const auto tileCount = tileOp.getCount();
    if (tileCount == 1) {
        return;
    }

    func.walk([&](NCEClusterTaskOp nceOp) {
        auto outputType = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(nceOp.getOutput().getType());
        if (outputType == nullptr) {
            return;
        }

        auto outputItiBuffs = nceOp.getOutput_ITIBuff();

        auto outwardHaloDstNCEOpMap = getTargetNCEOpForEachOutwardHalo(outputType, outputItiBuffs);

        auto dpuTaskOps = nceOp.getVariants().getOps<DPUTaskOp>();
        for (auto dpuTaskOp : llvm::make_early_inc_range(dpuTaskOps)) {
            if (!dpuTaskOp.getHaloRegions().has_value()) {
                computeHaloRegion(nceOp, dpuTaskOp, outwardHaloDstNCEOpMap, _log);
            }
        }
    });
}

}  // namespace

//
// createComputeHaloRegionForDPUTaskOpPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createComputeHaloRegionForDPUTaskOpPass(Logger log) {
    return std::make_unique<ComputeHaloRegionForDPUTaskOpPass>(log);
}

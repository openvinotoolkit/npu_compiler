//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_SEGMENTHALOS
#define GEN_PASS_DEF_SEGMENTHALOS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
using namespace VPUIP;

using InwardHaloSubstitutesType = DenseMap<mlir::Attribute, SmallVector<HaloRegionAttr>>;
namespace {

struct SubstituteHalos {
    SmallVector<OutwardHaloRegionAttr> _outwardHalos = {};
    InwardHaloSubstitutesType _inwardHalos{};

    SubstituteHalos() = default;
    SubstituteHalos(ArrayRef<OutwardHaloRegionAttr> outwardHalos, const InwardHaloSubstitutesType& inwardHalos)
            : _outwardHalos(outwardHalos.begin(), outwardHalos.end()), _inwardHalos(inwardHalos){};
};

struct NceOpOutputs {
    mlir::Value _output;
    mlir::Value _outSparsityMap;

    NceOpOutputs() = default;
    NceOpOutputs(mlir::Value output, mlir::Value outSparsityMap): _output(output), _outSparsityMap(outSparsityMap){};
};

bool isHaloInCurrentWorkload(ArrayRef<int64_t> haloStart, ArrayRef<int64_t> haloShape, ArrayRef<int64_t> workloadStart,
                             ArrayRef<int64_t> workloadEnd) {
    const auto numDims = workloadStart.size();
    const SmallVector<int64_t> workloadDimToHaloDim = {Dims4D::Act::W.ind(), Dims4D::Act::H.ind(),
                                                       Dims4D::Act::C.ind()};

    for (size_t dim = 0; dim < numDims; dim++) {
        const auto haloDim = workloadDimToHaloDim[dim];
        const auto haloEnd = haloStart[haloDim] + haloShape[haloDim] - 1;
        if (haloStart[haloDim] > workloadEnd[dim] || haloEnd < workloadStart[dim]) {
            return false;
        }
    }

    return true;
}

void computeNewOutwardInwardHalos(mlir::OpBuilder builder, SubstituteHalos& newHalos, OutwardHaloRegionAttr outwardHalo,
                                  ArrayRef<int64_t> workloadStart, ArrayRef<int64_t> workloadEnd, Logger log) {
    const SmallVector<int64_t> workloadDimToHaloDim = {Dims4D::Act::W.ind(), Dims4D::Act::H.ind(),
                                                       Dims4D::Act::C.ind()};
    auto ctx = outwardHalo.getContext();

    const auto initHaloStart = parseIntArrayAttr<int64_t>(outwardHalo.getOffset());
    auto haloStart = initHaloStart;
    auto haloShape = parseIntArrayAttr<int64_t>(outwardHalo.getShape());

    for (size_t dim = 0; dim < workloadStart.size(); dim++) {
        const auto haloDim = workloadDimToHaloDim[dim];

        if (haloStart[haloDim] < workloadStart[dim]) {
            haloStart[haloDim] = workloadStart[dim];
        }

        const auto haloEnd = haloStart[haloDim] + haloShape[haloDim] - 1;
        if (haloEnd > workloadEnd[dim]) {
            haloShape[haloDim] = workloadEnd[dim] - haloStart[haloDim] + 1;
        }
    }

    auto haloShapeAttr = getIntArrayAttr(ctx, haloShape);
    auto haloOffsetAttr = getIntArrayAttr(ctx, haloStart);

    SmallVector<mlir::Attribute> newInwardHalosForOutwardHalo = {};
    for (auto inwardHalo : outwardHalo.getInwardHaloRegions()) {
        auto inwardHaloAttr = inwardHalo.cast<HaloRegionAttr>();
        auto inwardHaloOffset = parseIntArrayAttr<int64_t>(inwardHaloAttr.getOffset());

        for (size_t dim = 0; dim < inwardHaloOffset.size(); dim++) {
            inwardHaloOffset[dim] += haloStart[dim] - initHaloStart[dim];
        }

        auto inwardHaloOffsetAttr = getIntArrayAttr(ctx, inwardHaloOffset);
        auto newInwardHalo =
                HaloRegionAttr::get(ctx, haloShapeAttr, inwardHaloOffsetAttr, inwardHaloAttr.getClusterId());
        ;

        newInwardHalosForOutwardHalo.push_back(newInwardHalo.cast<mlir::Attribute>());
        if (newHalos._inwardHalos.count(inwardHalo) == 0) {
            newHalos._inwardHalos[inwardHalo] = SmallVector<HaloRegionAttr>{newInwardHalo};
        } else {
            newHalos._inwardHalos[inwardHalo].push_back(newInwardHalo);
        }
    }

    auto inwardHalosAttr = builder.getArrayAttr(newInwardHalosForOutwardHalo);
    auto newOutwardHalo =
            OutwardHaloRegionAttr::get(ctx, haloShapeAttr, haloOffsetAttr, outwardHalo.getClusterId(), inwardHalosAttr);
    newHalos._outwardHalos.push_back(newOutwardHalo);

    log.trace("Outward halo {0} to be replaced by '{1}'", outwardHalo, newOutwardHalo);
}

SubstituteHalos segmentHalos(NCEClusterTaskOp nceOp, Logger log) {
    auto dpuTaskOps = nceOp.getVariants().getOps<DPUTaskOp>();
    auto outputType = nceOp.getOutputBuff().getType().cast<ITIBufferType>();

    mlir::OpBuilder builder(nceOp);

    SubstituteHalos newHalos{};
    for (const auto& outwardHalo : outputType.getOutwardHaloRegions()) {
        const auto outwardHaloShape = parseIntArrayAttr<int64_t>(outwardHalo.getShape());
        const auto outwardHaloOffset = parseIntArrayAttr<int64_t>(outwardHalo.getOffset());

        for (auto dpuTask : dpuTaskOps) {
            const auto currentWorkloadStart = parseIntArrayAttr<int64_t>(dpuTask.getOutStart());
            const auto currentWorkloadEnd = parseIntArrayAttr<int64_t>(dpuTask.getOutEnd());
            if (!isHaloInCurrentWorkload(outwardHaloOffset, outwardHaloShape, currentWorkloadStart,
                                         currentWorkloadEnd)) {
                continue;
            }

            log.trace("Found outward halo {0} in dpuTask '{1}'", outwardHalo, dpuTask);

            computeNewOutwardInwardHalos(builder, newHalos, outwardHalo, currentWorkloadStart, currentWorkloadEnd, log);
        }
    }

    return newHalos;
}

// Get all the consumer NCEOps that current NCEOp will write a halo to
SmallVector<NCEClusterTaskOp> getTargetNCEOps(mlir::OperandRange outputItiBuffs) {
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

    SmallVector<NCEClusterTaskOp> neighbourNceOps = {};
    for (auto outputIti : outputItiBuffs) {
        const auto neighbourItiOp = getProducerNCEOp(outputIti);
        VPUX_THROW_UNLESS(neighbourItiOp != nullptr,
                          "Could not find NCEClusterTaskOp producer for output_ITI buff: {0}", outputIti);

        neighbourNceOps.push_back(neighbourItiOp);
    }

    return neighbourNceOps;
}

VPURT::DeclareBufferOp createBuffer(mlir::MLIRContext* ctx, ITIBufferType outputType, mlir::Value bufferOutput,
                                    ArrayRef<HaloRegionAttr> inwardHaloRegions,
                                    ArrayRef<OutwardHaloRegionAttr> outwardHaloRegions, Logger log) {
    auto outNdType = outputType.cast<NDTypeInterface>();
    auto bufferOp = bufferOutput.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(bufferOp != nullptr, "Defining op of buffer {0} is not a DeclareBufferOp: {1}", bufferOutput,
                      bufferOp);

    log.trace("Attempting to replace buffer '{0}'", bufferOp);
    auto newItiType = ITIBufferType::get(ctx, outNdType.getShape().raw(), outNdType.getElementType(),
                                         outputType.getLayout(), outNdType.getMemSpace(),
                                         outputType.getIduSegmentation(), inwardHaloRegions, outwardHaloRegions);

    mlir::OpBuilder builder(bufferOp);
    auto newBufferOp = builder.create<VPURT::DeclareBufferOp>(
            bufferOp.getLoc(), newItiType, bufferOp.getSectionAttr(), bufferOp.getSectionIndexAttr(),
            bufferOp.getByteOffsetAttr(), bufferOp.getSwizzlingKeyAttr());

    log.trace("Replaced with buffer '{0}'", newBufferOp);

    return newBufferOp;
}

void updateNceOps(NCEClusterTaskOp nceOp, DenseMap<NCEClusterTaskOp, NceOpOutputs>& newOutputs, Logger log) {
    auto outputItiBuffs = nceOp.getOutput_ITIBuff();
    auto dstNCEOps = getTargetNCEOps(outputItiBuffs);

    SmallVector<mlir::Value> newOutputItis = {};
    for (auto dstNceOp : dstNCEOps) {
        auto dstOutput = newOutputs[dstNceOp]._output;
        newOutputItis.push_back(dstOutput);
    }

    auto output = newOutputs[nceOp]._output;
    auto outSparsityMap = newOutputs[nceOp]._outSparsityMap;

    mlir::Type outSparsityMapType = outSparsityMap != nullptr ? outSparsityMap.getType() : nullptr;
    mlir::Type profilingOutputType = nceOp.getProfilingData() != nullptr ? nceOp.getProfilingData().getType() : nullptr;

    auto taskOp = nceOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(taskOp != nullptr, "Can't get VPURT task operation");

    mlir::OpBuilder builder(taskOp);

    auto updatedNceOp = VPURT::wrapIntoTaskOp<NCEClusterTaskOp>(
            builder, taskOp.getWaitBarriers(), taskOp.getUpdateBarriers(), nceOp.getLoc(), output.getType(),
            outSparsityMapType, profilingOutputType, nceOp.getInput(), nceOp.getInputSparsityMap(),
            nceOp.getInputStorageElementTable(), nceOp.getWeights(), nceOp.getWeightsSparsityMap(),
            nceOp.getWeightTable(), /*weight_table_data_ptr=*/nullptr, /*weight_table_sp_ptr=*/nullptr,
            /*weight_table_scale=*/nullptr, /*weight_table_bias=*/nullptr, /*weight_zero_points=*/nullptr,
            nceOp.getSprLookupTable(), nceOp.getPalletLookupTable(), nceOp.getParentInput(),
            nceOp.getParentInputSparsityMap(), nceOp.getParentInputStorageElementTable(), output, outSparsityMap,
            mlir::ValueRange(newOutputItis), output, outSparsityMap, nceOp.getProfilingData(), nceOp.getMaxPerXy(),
            nceOp.getMinPerXy(), nceOp.getMinMaxPerTensor(), nceOp.getTaskType(), nceOp.getKernelSizeAttr(),
            nceOp.getKernelStridesAttr(), nceOp.getKernelPaddingAttr(), nceOp.getIsContinuedAttr(),
            nceOp.getCmSpPatternAttr(),
            /*is_segmented*/ nullptr, nceOp.getOutChannelOffsetAttr(), nceOp.getInputChannelsCompressionAttr(),
            nceOp.getIsZeroOffsetWeightsTableAttr(), nceOp.getIsSuperdenseAttr(), nceOp.getIsInplaceAttr(),
            nceOp.getInputSeSizeAttr(), nceOp.getOutputSeSizeAttr(), nceOp.getIsPermuteQuantizeAttr(),
            nceOp.getIsSmallKernelOptimizedAttr(), nceOp.getMpeEngineAttr(), nceOp.getEltwiseTypeAttr());
    if (auto profMetadata = nceOp.getProfilingMetadataAttr()) {
        updatedNceOp.setProfilingMetadataAttr(profMetadata);
    }

    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(&updatedNceOp.getVariants().front());

        for (auto variant : nceOp.getVariants().getOps<VPUIP::DPUTaskOp>()) {
            builder.clone(*variant);
        }
    }

    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(&updatedNceOp.getPpe().front());

        for (auto& ppe : nceOp.getPpe().getOps()) {
            builder.clone(ppe);
        }
    }

    nceOp->replaceAllUsesWith(updatedNceOp);

    log.trace("Created new NCEClusterTaskOp '{0}', to replace {1}", updatedNceOp, nceOp);
}

//
// Segment Halo Regions so that each one fits inside one workload only
//

class SegmentHalosPass final : public VPUIP::impl::SegmentHalosBase<SegmentHalosPass> {
public:
    explicit SegmentHalosPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void SegmentHalosPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto ctx = module.getContext();

    auto tileOp = IE::getTileExecutor(module);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE executor information");

    if (tileOp.getCount() == 1) {
        return;
    }

    DenseMap<NCEClusterTaskOp, SubstituteHalos> nceOpNewHalos{};

    // Pre-compute the new inward/outward halos for each NCEOp.
    func.walk([&](NCEClusterTaskOp nceOp) {
        auto outputType = nceOp.getOutput().getType().dyn_cast<ITIBufferType>();
        if (outputType == nullptr) {
            return;
        }

        auto outputItiBuffs = nceOp.getOutput_ITIBuff();
        if (outputItiBuffs.empty()) {
            return;
        }

        auto dstNCEOps = getTargetNCEOps(outputItiBuffs);
        const auto substituteHalos = segmentHalos(nceOp, _log);

        // For each op add its pre-computed outward halos
        if (nceOpNewHalos.count(nceOp) == 0) {
            nceOpNewHalos[nceOp] = SubstituteHalos(substituteHalos._outwardHalos, InwardHaloSubstitutesType{});
        } else {
            nceOpNewHalos[nceOp]._outwardHalos = substituteHalos._outwardHalos;
        }

        // For ops that consume the newly segmented halo, ensure that the inward halos are updated
        for (auto& dstNceOp : dstNCEOps) {
            auto dstOutputBuff = dstNceOp.getOutputBuff();
            auto dstOutputType = dstOutputBuff.getType().cast<ITIBufferType>();

            const auto dstInwardRegions = dstOutputType.getInwardHaloRegions();
            InwardHaloSubstitutesType newInwardHalos{};

            // Iterate over the NCE consumer and check if it has an inward halo that was segmented
            for (auto dstInwardRegion : dstInwardRegions) {
                auto substituteInwardHalos = substituteHalos._inwardHalos.find(dstInwardRegion.cast<mlir::Attribute>());
                if (substituteInwardHalos != substituteHalos._inwardHalos.end()) {
                    newInwardHalos.insert(*substituteInwardHalos);
                }
            }

            // Update pre-computed inward halos for destination op
            if (nceOpNewHalos.count(dstNceOp) == 0) {
                nceOpNewHalos[dstNceOp] = SubstituteHalos(SmallVector<OutwardHaloRegionAttr>{}, newInwardHalos);
            } else {
                nceOpNewHalos[dstNceOp]._inwardHalos.insert(newInwardHalos.begin(), newInwardHalos.end());
            }
        }
    });

    DenseMap<NCEClusterTaskOp, NceOpOutputs> newOutputs{};

    // Make new ITI buffers from the segmented halos
    for (auto& opHalos : nceOpNewHalos) {
        auto nceOp = opHalos.first;
        const auto& newHalos = opHalos.second;

        auto outputType = nceOp.getOutput().getType().cast<ITIBufferType>();
        SmallVector<HaloRegionAttr> newInwardHalos = {};

        for (auto oldInwardHalo : outputType.getInwardHaloRegions()) {
            auto replacementHalos = newHalos._inwardHalos.find(oldInwardHalo);
            if (replacementHalos != newHalos._inwardHalos.end()) {
                newInwardHalos.append(replacementHalos->second.begin(), replacementHalos->second.end());
            } else {
                newInwardHalos.push_back(oldInwardHalo);
            }
        }

        auto newOutputBuffer =
                createBuffer(ctx, outputType, nceOp.getOutputBuff(), newInwardHalos, newHalos._outwardHalos, _log);

        mlir::Value newSparseMap = nullptr;
        if (auto outSparsityMap = nceOp.getOutputSparsityMapBuff()) {
            auto oldSparseMapItiType = outSparsityMap.getType().cast<ITIBufferType>();
            auto newSparseBufferOp = createBuffer(ctx, oldSparseMapItiType, outSparsityMap, newInwardHalos,
                                                  newHalos._outwardHalos, _log);

            newSparseMap = newSparseBufferOp.getBuffer();
        }

        newOutputs.insert(std::pair<NCEClusterTaskOp, NceOpOutputs>(
                nceOp, NceOpOutputs(newOutputBuffer.getBuffer(), newSparseMap)));
    }

    // Create new NCE ops with updated output buffers
    for (auto opToUpdate : newOutputs) {
        updateNceOps(opToUpdate.first, newOutputs, _log);
    }

    SmallVector<VPURT::DeclareBufferOp> bufferToErase = {};

    // Delete old NCE ops
    for (auto nceOpsToErase : llvm::make_early_inc_range(newOutputs)) {
        auto oldNceOp = nceOpsToErase.first;
        auto outputBuff = oldNceOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();
        auto sparseBuff = oldNceOp.getOutputSparsityMapBuff() != nullptr
                                  ? oldNceOp.getOutputSparsityMapBuff().getDefiningOp<VPURT::DeclareBufferOp>()
                                  : nullptr;

        bufferToErase.push_back(outputBuff);
        if (sparseBuff != nullptr) {
            bufferToErase.push_back(sparseBuff);
        }

        auto taskOp = oldNceOp->getParentOfType<VPURT::TaskOp>();
        taskOp->erase();
    }

    // Delete old buffers
    for (auto buffer : llvm::make_early_inc_range(bufferToErase)) {
        buffer->erase();
    }
}

}  // namespace

//
// createSegmentHalosPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSegmentHalosPass(Logger log) {
    return std::make_unique<SegmentHalosPass>(log);
}

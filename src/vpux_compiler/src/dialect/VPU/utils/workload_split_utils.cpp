//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/workload_split_utils.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/sparsity.hpp"

#include <vpu_cost_model.h>

using namespace vpux;
using namespace VPU;

// For workloads in sub tensors, offsets need to be from original full output tensor
void addSubTensorOffset(TileInfo& tileInfo, ShapeRef tensorOffset) {
    VPUX_THROW_WHEN(tileInfo.offsets.size() != tensorOffset.size(),
                    "Invalid size for TileInfo.offset {0} and sub tensor offset {1}", tileInfo.offsets.size(),
                    tensorOffset.size());

    for (auto d : irange(tileInfo.offsets.size())) {
        const auto dim = Dim(d);
        tileInfo.offsets[dim] += tensorOffset[dim];
    }
}

int64_t computeSplitCost(const VPUIP::WorkloadSplit& split, const VPUIP::WorkloadCostParams& params,
                         VPUNN::VPUCostModel& costModel, bool isAutopadODUEnabled, LogCb logCb) {
    VPUX_THROW_WHEN(params.arch < config::ArchKind::NPU37XX, "Unexpected architecture {0}", params.arch);
    std::vector<int64_t> workloadCost;
    workloadCost.reserve(split.size());

    std::string vpunnInputCheckInfo;

    // Correct invalid input channels for depthwise workload before passing to VPUNN
    // split to produce more small and valid workloads
    const SmallVector<int64_t> supportedChannelsDW = {64, 32, 16};
    auto correctDepthwiseWorkloadChannel = [=](const VPUIP::WorkloadTile& wl) -> std::vector<VPUIP::WorkloadTile> {
        auto wlChannel = std::get<0>(wl).shape[Dims4D::Act::C];

        // In case the autopadding feature is used, the output channels might not be aligned to be a multiple of 16
        // If this happens, the current output channel configuration can be considered a supported workload
        // configuration
        if (isAutopadODUEnabled && wlChannel < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) {
            return {wl};
        }

        SmallVector<int64_t> validWorkloadChannels;
        std::vector<VPUIP::WorkloadTile> newWorkloads;
        auto newWl = wl;
        validWorkloadChannels = splitWorkloadChannel(wlChannel, supportedChannelsDW);
        VPUX_THROW_WHEN(validWorkloadChannels.size() == 0,
                        "splitWorkloadChannel failed please check wlChannel - {0}, supportedChannelsDW - {1}",
                        wlChannel, supportedChannelsDW);
        for (auto validChannel : validWorkloadChannels) {
            std::get<0>(newWl).shape[Dims4D::Act::C] = validChannel;
            newWorkloads.push_back(newWl);
        }
        return newWorkloads;
    };

    std::vector<VPUIP::WorkloadTile> correctWls;
    for (const auto& wl : split) {
        correctWls.push_back(wl);
        // Split workload channel to satisfy HW limit for depthwise ops before passing to VPUNN
        if (params.nceTaskType == VPUIP::NCETaskType::DWCONV || params.nceTaskType == VPUIP::NCETaskType::MAXPOOL ||
            params.nceTaskType == VPUIP::NCETaskType::AVEPOOL) {
            auto wlChannel = std::get<0>(wl).shape[Dims4D::Act::C];
            if (std::find(supportedChannelsDW.begin(), supportedChannelsDW.end(), wlChannel) ==
                supportedChannelsDW.end()) {
                correctWls = correctDepthwiseWorkloadChannel(wl);
            }
        }

        for (const auto& correctWl : correctWls) {
            const auto vpunnWorkload = VPU::getDPUWorkload(params, correctWl);
            auto wlCost =
                    VPU::checkAndReturnCost(costModel.DPU(vpunnWorkload, vpunnInputCheckInfo), Logger::global(), true);
            if (wlCost >= VPU::INVALID_COST_BASE) {
                logCb(formatv("[VPUNN LOG] INVALID_COST is caught. Please check possible VPUNN debug info: {0}",
                              vpunnInputCheckInfo));
                VPU::printVPUNNWorkloadConfig(vpunnWorkload, logCb);
            }
            workloadCost.push_back(static_cast<int64_t>(wlCost));
        }

        correctWls.clear();
    }

    return VPUNN::dpu_schedule(checked_cast<unsigned int>(params.numDPU), workloadCost);
}

void generateWorkloads(mlir::OpBuilder& builder, VPU::NCEOpInterface origOp,
                       const VPUIP::WorkloadCostParams& costParams, VPU::MPEMode mpeMode,
                       ArrayRef<bool> isTileOverDimsSupported, VPUNN::VPUCostModel& costModel, Logger log,
                       mlir::IntegerAttr clusterId = nullptr, ShapeRef subTensorOffset = {}) {
    VPUIP::DpuTiler dpuTiler(costParams.outputShape, mpeMode);

    VPUIP::WorkloadSplitPool splitPoolSet;

    dpuTiler.tileOverH(costParams.numDPU, splitPoolSet);

    if (costParams.outputShape.size() == 5) {
        int64_t cluster = 0;
        if (clusterId != nullptr) {
            cluster = clusterId.getValue().getSExtValue();
        }
        // This logic assumes that each chunk starts right after the previous.
        // cluster 0: outOffsets [0, 0, 0, 0, 0]  outSizes [32, 1, 16, 16, 1]
        // cluster 1: outOffsets [32, 0, 0, 0, 0] outSizes [32, 1, 16, 16, 1]
        // cluster 2: outOffsets [64, 0, 0, 0, 0] outSizes [32, 1, 16, 16, 1]
        const Shape offsets = subTensorOffset.empty() ? Shape{0, 0, 0, 0, 0} : Shape(subTensorOffset);
        auto tilePad = VPU::getPaddingAttr(builder.getContext(), 0, 0, 0, 0);
        origOp.addWorkload(builder, origOp.getLoc(), offsets, costParams.outputShape, tilePad,
                           VPU::MPEMode::CUBOID_16x16, getIntAttr(origOp->getContext(), cluster));
        return;
    } else {
        dpuTiler.tileOverH(costParams.numDPU, splitPoolSet);
        // Invariants that produce sparse activations must have the same number of channels across the variants
        const auto requiresEqualZ =
                (mlir::dyn_cast<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()) != nullptr);
        const auto splitNumPool = dpuTiler.generateSplitNumberPool(costParams.numDPU, 1);

        for (const auto& splitNum : splitNumPool) {
            if (isTileOverDimsSupported[Dims4D::Act::W.ind()] == true &&
                isTileOverDimsSupported[Dims4D::Act::H.ind()] == true) {
                dpuTiler.tileOverHW(splitNum, VPUIP::SplitDimension::SPLIT_OVER_HW, splitPoolSet);
            } else if (isTileOverDimsSupported[Dims4D::Act::W.ind()] == true) {
                dpuTiler.tileOverHW(splitNum, VPUIP::SplitDimension::SPLIT_OVER_W, splitPoolSet);
            } else if (isTileOverDimsSupported[Dims4D::Act::H.ind()] == true) {
                dpuTiler.tileOverHW(splitNum, VPUIP::SplitDimension::SPLIT_OVER_H, splitPoolSet);
            }
            if (isTileOverDimsSupported[Dims4D::Act::C.ind()] == true) {
                dpuTiler.tileOverZ(splitNum, splitPoolSet, requiresEqualZ);
            }
        }
    }

    // select workload with minimum cost
    auto splitPool = to_std_vector(splitPoolSet);
    VPUX_THROW_WHEN(splitPool.empty(), "Workload split pool is empty");

    const auto isAutopadODUEnabled = hasAutoPaddingODU(getModuleOp(origOp));

    std::vector<int64_t> splitPoolCosts(splitPool.size(), 0);
    for (const auto ind : irange(splitPool.size())) {
        auto& curSplit = splitPool[ind];

        if (clusterId != nullptr) {
            for (auto& wl : curSplit) {
                auto& outTile = std::get<0>(wl);
                addSubTensorOffset(outTile, subTensorOffset);
            }
        }
        const auto logCb = [&](const formatv_object_base& msg) {
            log.trace("{0}", msg.str());
        };
        splitPoolCosts[ind] = computeSplitCost(curSplit, costParams, costModel, isAutopadODUEnabled, logCb);
    }

    const auto bestSplitInd = std::min_element(splitPoolCosts.begin(), splitPoolCosts.end()) - splitPoolCosts.begin();
    if (splitPoolCosts[bestSplitInd] >= VPU::INVALID_COST_BASE) {
        log.setName("GenerateWorkloads");
        log.debug("An INVALID_COST is caught for bestSplit when calling VPUNN. You can pass a logCb with LOG_ERROR "
                  "level to print debug info in `computeSplitCostByArch` function and report to E#83609 if necessary");
        log.nest().debug("bestSplit cost value: {0}", splitPoolCosts[bestSplitInd]);
    }
    const auto& bestSplit = splitPool[bestSplitInd];

    origOp->setAttr(DPUCost, getIntAttr(origOp->getContext(), splitPoolCosts[bestSplitInd]));

    const auto kernel = origOp.getKernelSizeVal();
    const auto strides = origOp.getStridesVal();

    for (const auto& wl : bestSplit) {
        const auto& outTile = std::get<0>(wl);
        const auto mpeMode = std::get<1>(wl);

        const auto padsTileConf =
                backInferPadsTile(outTile, costParams.fullInputShape, costParams.padInfo, kernel, strides);
        auto tilePad = VPU::getPaddingAttr(builder.getContext(), padsTileConf);

        origOp.addWorkload(builder, origOp.getLoc(), outTile.offsets, outTile.shape, tilePad, mpeMode, clusterId);
    }
}

VPU::DistributedTensorType vpux::VPU::getDistributedTensor(const mlir::Value value) {
    if (auto sparseTensor = mlir::dyn_cast<vpux::VPU::SparseTensorType>(value.getType())) {
        return mlir::dyn_cast<vpux::VPU::DistributedTensorType>(sparseTensor.getData());
    }
    return mlir::dyn_cast<vpux::VPU::DistributedTensorType>(value.getType());
}

void splitOntoWorkloads(mlir::OpBuilder& builder, VPU::NCEOpInterface origOp, VPUIP::WorkloadCostParams& costParams,
                        VPU::MPEMode mpeMode, ArrayRef<bool> isTileOverDimsSupported, VPUNN::VPUCostModel& costModel,
                        Logger log) {
    auto distributedIf = mlir::dyn_cast<VPU::DistributedTypeInterface>(origOp->getResult(0).getType());
    if ((distributedIf != nullptr) && (distributedIf.containsDistributedTypes())) {
        const auto outputs = origOp->getResults();
        VPUX_THROW_UNLESS(outputs.size() == 1, "Wrong outputs size: {0}", outputs.size());

        const auto output = *outputs.begin();

        auto distributedOutputType = getDistributedTensor(output);
        VPUX_THROW_WHEN(distributedOutputType == nullptr, "Wrong output type {0} for distributed operation",
                        output.getType());

        const auto outputSubTensorShapes = distributedOutputType.getPerClusterComputeShapes();
        auto outputSubTensorOffsets = distributedOutputType.getPerClusterComputeShapeOffsets();
        VPUX_THROW_WHEN(outputSubTensorShapes.size() != outputSubTensorOffsets.size(),
                        "sub tensor size:{0} not equal to offset size:{1}", outputSubTensorShapes.size(),
                        outputSubTensorOffsets.size());

        const auto inputs = origOp->getOperands();
        VPUX_THROW_UNLESS(inputs.size() >= 1, "Wrong inputs size: {0}", inputs.size());

        const auto input = *inputs.begin();
        auto distributedInputType = getDistributedTensor(input);
        VPUX_THROW_WHEN(distributedInputType == nullptr, "Wrong input type {0} for distributed operation",
                        input.getType());

        // @todo When halos supported in VPUNN, we need use computeShape instead of memory shape
        // See E#87028
        const auto inputSubTensorShapes = distributedInputType.getPerClusterMemoryShapes();
        VPUX_THROW_WHEN(outputSubTensorShapes.size() != inputSubTensorShapes.size(),
                        "output tensor size:{0} not equal to input tensor size:{1}", outputSubTensorShapes.size(),
                        inputSubTensorShapes.size());

        const auto distributionAttr = distributedOutputType.getDistribution();
        if (isSegmentedOverC(distributionAttr)) {
            // Here we keep the output offset for SOC NCEPermute to keep the logic be aligned
            // with SOH because it will be lowered to SOH NCEEltwise
            if (mlir::isa<VPU::NCEPermuteOp>(origOp.getOperation())) {
                // Correct layer strategy to the real strategy after being lowered to Eltwise
                costParams.layerStrategy = VPU::MultiClusterStrategy::SplitOverHeight;
            } else {
                // In the case of an non broadcasted SOK, outputSubTensorOffsets don't need to be applied
                for (auto& shapeOffset : outputSubTensorOffsets) {
                    std::fill(shapeOffset.begin(), shapeOffset.end(), 0);
                }
            }
        }

        for (size_t clusterId = 0; clusterId < outputSubTensorShapes.size(); clusterId++) {
            auto clusterIdAttr = getIntAttr(origOp->getContext(), clusterId);
            // Update workload params for per tile
            costParams.inputShape = inputSubTensorShapes[clusterId];
            costParams.outputShape = outputSubTensorShapes[clusterId];
            costParams.numTiles = distributionAttr.getNumClusters().getInt();
            // #E129156 once with the update of VPUNN to provide MPE mode explicitly
            if (costParams.arch != config::ArchKind::NPU40XX &&
                mlir::isa<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEInterpolateOp>(origOp)) {
                mpeMode = origOp.getMpeMode(nullptr, nullptr, outputSubTensorShapes[clusterId]);
            }
            generateWorkloads(builder, origOp, costParams, mpeMode, isTileOverDimsSupported, costModel, log,
                              clusterIdAttr, outputSubTensorOffsets[clusterId]);
        }
    } else {
        generateWorkloads(builder, origOp, costParams, mpeMode, isTileOverDimsSupported, costModel, log);
    }
}

VPU::MPEMode getNCEHeuristicMPEMode(VPU::NCEOpInterface nceOp) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getOperand(0).getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());

    const auto inElemType = inputType.getElementType();
    const auto outElemType = outputType.getElementType();

    const auto outputShape = outputType.getShape();

    return nceOp.getMpeMode(inElemType, outElemType, outputShape);
}

SmallVector<bool> getSupportedWorkloadSplitDim(VPU::NCEOpInterface nceOp, vpux::VPU::MPEMode mpeMode) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getOperand(0).getType());
    SmallVector<bool> isTileOverDimsSupported = {false, mpeMode == VPU::MPEMode::VECTOR, true, true};
    if (mlir::isa<VPU::NCEConvolutionOp>(nceOp.getOperation())) {
        const auto inOrder = inputType.getDimsOrder();
        const auto isCMajor = inOrder == DimsOrder::NCHW;
        isTileOverDimsSupported[Dims4D::Act::C.ind()] |= !isCMajor;
    } else if (mlir::isa<VPU::NCEEltwiseOp>(nceOp.getOperation())) {
        isTileOverDimsSupported[Dims4D::Act::C.ind()] = false;
    } else if (mlir::isa<VPU::NCEPermuteOp>(nceOp.getOperation())) {
        // For NCE Permute operation tileOverHK is needed : See E#91637
        isTileOverDimsSupported[Dims4D::Act::W.ind()] = false;
    }
    return isTileOverDimsSupported;
}

mlir::LogicalResult vpux::VPU::genericNCEWorkloadSplit(VPU::NCEOpInterface nceOp, mlir::PatternRewriter& rewriter,
                                                       config::ArchKind arch, int64_t numDPU,
                                                       std::shared_ptr<VPUNN::VPUCostModel> costModel, Logger log) {
    const auto mpeMode = getNCEHeuristicMPEMode(nceOp);
    auto params = VPU::getWorkloadCostParam(nceOp, arch, numDPU);
    auto isTileOverDimsSupported = getSupportedWorkloadSplitDim(nceOp, mpeMode);
    rewriter.modifyOpInPlace(nceOp, [&]() {
        splitOntoWorkloads(rewriter, nceOp, params, mpeMode, ArrayRef(isTileOverDimsSupported), *costModel, log);
    });
    return mlir::success();
}

/*
 * Correct output compute shape for NCE.Permute workloads
 * VPUNN only accepts NHWC input layout, so compiler converts the NCEPermuteOp from NCHW->NHWC to NHWC->NWCH
 * Need to cast the shape from VPUNN representation NHWC to VPU layer NCHW
 * The OC was converted to IC in correctParamsForNcePermute to pass VPUNN check, change it back
 */
void mapNCEPermuteShape(Shape& vpunnShape, int64_t OC) {
    vpunnShape[Dims4D::Act::H] = vpunnShape[Dims4D::Act::W];
    vpunnShape[Dims4D::Act::W] = vpunnShape[Dims4D::Act::C];
    vpunnShape[Dims4D::Act::C] = OC;
}

void vpux::VPU::splitWorkloadsWithInfo(VPU::NCEOpInterface nceOp, mlir::OpBuilder& builder,
                                       const VPUNN::LayerSplitInfo& splitInfo, Logger log) {
    log.trace("Splitting nce op into {0} clusters", splitInfo.size());

    auto ctx = builder.getContext();
    auto distributedType = getDistributedTensor(nceOp->getResult(0));
    VPUX_THROW_WHEN(distributedType == nullptr, "not distributed type");
    auto perClusterOffsets = distributedType.getPerClusterComputeShapeOffsets();
    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    auto combineOffsets = [](Shape offsets1, Shape offsets2) {
        VPUX_THROW_UNLESS(offsets1.size() == offsets2.size(), "can't combine offset {0} and {1}", offsets1, offsets2);
        auto combinedShape = Shape(offsets1.size(), 0);
        for (auto ind : irange(combinedShape.size())) {
            combinedShape[Dim(ind)] = offsets1[Dim(ind)] + offsets2[Dim(ind)];
        }
        return combinedShape;
    };

    for (auto clusterId : irange(splitInfo.size())) {
        auto clusterIdAttr = getIntAttr(ctx, clusterId);
        auto intraTileSplit = splitInfo[clusterId].best_intra_tile_split;

        for (auto workload : intraTileSplit.second) {
            auto mpeMode = VPU::getMPEMode(workload.execution_order);
            auto shapeArray = workload.outputs[0].get_shape();
            SmallVector<int64_t> shape(shapeArray.size());
            std::reverse_copy(shapeArray.begin(), shapeArray.end(), shape.begin());
            const auto padding = workload.padding;
            auto offsetsArray = workload.offsets;
            SmallVector<int64_t> offsets(offsetsArray.size());
            std::reverse_copy(offsetsArray.begin(), offsetsArray.end(), offsets.begin());

            // perClusterOffsets indicates offsets for each cluster
            // VPUNN workload offsets indicates the offsets for each variant on single cluster
            // add the two offsets to represent the absolute offsets for each workload
            auto realOffsets = combineOffsets(perClusterOffsets[clusterId], Shape(offsets));
            const auto distributionAttr = distributedType.getDistribution();
            if (!mlir::isa<VPU::NCEPermuteOp>(nceOp.getOperation()) && isSegmentedOverC(distributionAttr)) {
                // VPUNN only accepts NHWC input layout, so compiler converts the NCEPermuteOp from NCHW->NHWC to
                // NHWC->NWCH The offsets returned from VPUNN are based on SOH split
                realOffsets = Shape(offsets);
            }
            // VPUNN padding order is [top, bottom, left, right]
            // Compiler padding attr order is [left, right, top, bottom]
            const auto paddingAttr = VPU::getPaddingAttr(ctx, PadInfo(padding[2], padding[3], padding[0], padding[1]));
            auto realShape = Shape(shape);
            if (mlir::isa<VPU::NCEPermuteOp>(nceOp.getOperation())) {
                mapNCEPermuteShape(realShape, perClusterShapes[clusterId][Dims4D::Act::C]);
            }

            nceOp.addWorkload(builder, nceOp->getLoc(), realOffsets, realShape, paddingAttr, mpeMode, clusterIdAttr);
        }
    }
}

bool vpux::VPU::isSupportedPreSplitNCEOp(VPU::NCEOpInterface nceOp) {
    if (mlir::isa<VPU::NCECompressConvolutionOp>(nceOp)) {
        // Track [E#160091]
        // CompressedConv VPUNN MPE mode causes performance regression
        return false;
    }
    auto hasSparseOperands = llvm::any_of(nceOp->getOperands(), [](auto operand) {
        return mlir::isa<VPU::SparseTensorType>(operand.getType());
    });
    auto isSEPInterpolate = mlir::isa<VPU::NCEInterpolateOp>(nceOp.getOperation()) && hasSparseOperands;
    if (isSEPInterpolate) {
        // Track E#158943, to provide correct parameters for SEP Interpolate op
        return false;
    }
    const auto outputShape = getShape(nceOp->getResult(0));
    if (outputShape.size() != 4) {
        return false;
    }
    if (isActSparseOp(nceOp)) {
        // Track E#160972. Activation sparse op accuracy issue
        return false;
    }
    const auto isDistributedOp = getDistributedTensor(nceOp->getResult(0)) != nullptr &&
                                 getDistributedTensor(nceOp->getOperand(0)) != nullptr;
    if (!isDistributedOp) {
        return false;
    }
    return true;
}

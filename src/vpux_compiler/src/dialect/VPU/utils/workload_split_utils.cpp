//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/workload_split_utils.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/hash.hpp"
#include "vpux/compiler/utils/sparsity.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

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

int64_t computeSplitCost(mlir::MLIRContext* ctx, const VPUIP::WorkloadSplit& split,
                         const VPUIP::WorkloadCostParams& params, VPUNN::VPUCostModel& costModel,
                         bool isAutopadODUEnabled, LogCb logCb) {
    VPUX_THROW_WHEN(params.arch < config::ArchKind::NPU37XX, "Unexpected architecture {0}", params.arch);
    std::vector<int64_t> workloadCost;
    workloadCost.reserve(split.size());
    std::string vpunnInputCheckInfo;

    // Correct invalid input channels for depthwise workload before passing to VPUNN
    // split to produce more small and valid workloads
    const auto& strategyFactory = VPU::getVPUStrategyFactory(ctx);
    const auto supportedChannelsDW = strategyFactory->getSupportedChannelsDW();
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

    auto getDPUWorkloadCost = [](const VPUNN::VPUCostModel& costModel, const VPUNN::DPUWorkload& vpunnWorkload,
                                 std::string& vpunnInputCheckInfo) -> size_t {
        // Enable a cache for DPU workload costs because the sanity check step within the VPU cost model is
        // computationally expensive. This sanity check may be invoked multiple times for the same workload when it is
        // split to fit hardware constraints.
        // TODO: This cache can be removed once the sanity check cost is sufficiently optimized.
        auto& cache = VPU::getGlobalOpTilingCache();
        const auto useCache = cache.isCacheSupported();
        llvm::hash_code wlHash;
        if (useCache) {
            wlHash = llvm::hash_combine(static_cast<const void*>(&costModel), vpunnWorkload.hash());
            auto cachedCost = cache.getDPUWorkloadCost(wlHash);
            if (cachedCost.has_value()) {
                return cachedCost.value();
            }
        }

        auto wlCost =
                VPU::checkAndReturnCost(costModel.DPU(vpunnWorkload, vpunnInputCheckInfo), Logger::global(), true);

        if (useCache) {
            cache.updateDPUWorkloadCost(wlHash, static_cast<size_t>(wlCost));
        }
        return static_cast<size_t>(wlCost);
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
            auto wlCost = getDPUWorkloadCost(costModel, vpunnWorkload, vpunnInputCheckInfo);
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
    auto ctx = origOp.getContext();
    // Tile in pre-ODU space so that workload offsets/shapes match the coordinate system the DPU
    // hardware operates in.  For ops without an active ODU transform,
    // preODUShape equals outputShape, so behaviour is identical.
    VPUIP::DpuTiler dpuTiler(costParams.preODUShape, mpeMode);
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
                           VPU::MPEMode::CUBOID_16x16, getIntAttr(ctx, cluster));
        return;
    } else {
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

    const auto isAutopadODUEnabled = config::hasAutoPaddingODU(getModuleOp(origOp));

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
        splitPoolCosts[ind] = computeSplitCost(ctx, curSplit, costParams, costModel, isAutopadODUEnabled, logCb);
    }

    const auto bestSplitInd = std::min_element(splitPoolCosts.begin(), splitPoolCosts.end()) - splitPoolCosts.begin();
    if (splitPoolCosts[bestSplitInd] >= VPU::INVALID_COST_BASE) {
        log.setName("GenerateWorkloads");
        log.debug("An INVALID_COST is caught for bestSplit when calling VPUNN. You can pass a logCb with LOG_ERROR "
                  "level to print debug info in `computeSplitCostByArch` function and report to E#83609 if necessary");
        log.nest().debug("bestSplit cost value: {0}", splitPoolCosts[bestSplitInd]);
    }
    const auto& bestSplit = splitPool[bestSplitInd];

    if (mlir::dyn_cast<VPU::NCEConvolutionOp>(origOp.getOperation()) && bestSplit.size() > 1) {
        VPUX_THROW("NCE Convolution best split can't contain multiple variants per invariant (cluster). Only one "
                   "variant per invariant is allowed.");
    }
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

        // ODU scales are op-level (same for all clusters); read once before the loop.
        const auto oduScales = VPU::getODUScaling(origOp.getOperation());

        for (size_t clusterId = 0; clusterId < outputSubTensorShapes.size(); clusterId++) {
            auto clusterIdAttr = getIntAttr(origOp->getContext(), clusterId);
            // Update workload params for per tile
            costParams.inputShape = inputSubTensorShapes[clusterId];
            costParams.outputShape = outputSubTensorShapes[clusterId];
            // Map the per-cluster post-ODU output shape back to pre-ODU space so DpuTiler
            // sees the coordinate system the DPU hardware operates in.  When no transform is
            // active, invertODUScaling is a no-op and preODUShape equals outputShape.
            const auto preODUShapeResult = VPU::invertODUScaling(
                    oduScales, SmallVector<int64_t>(costParams.outputShape.raw()), origOp->getLoc());
            VPUX_THROW_UNLESS(mlir::succeeded(preODUShapeResult),
                              "Failed to invert ODU scaling for workload split at cluster {0}", clusterId);
            costParams.preODUShape = Shape(preODUShapeResult.value());

            // Per-cluster distributed offsets are defined in post-ODU output space.
            // Workload generation uses pre-ODU coordinates, so map offsets back as well.
            auto preODUOffset = SmallVector<int64_t>(outputSubTensorOffsets[clusterId].raw());
            if (!oduScales.empty()) {
                const auto preODUOffsetResult = VPU::invertODUScaling(oduScales, preODUOffset, origOp->getLoc());
                VPUX_THROW_UNLESS(mlir::succeeded(preODUOffsetResult),
                                  "Failed to invert ODU output offset for workload split at cluster {0}", clusterId);
                preODUOffset = preODUOffsetResult.value();
            }

            costParams.numTiles = distributionAttr.getNumClusters().getInt();
            // #E129156 once with the update of VPUNN to provide MPE mode explicitly
            // avoid using below logic for newer architectures
            if (costParams.arch != config::ArchKind::NPU40XX &&
                mlir::isa<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEInterpolateOp>(origOp)) {
                mpeMode = origOp.getMpeMode(nullptr, nullptr, outputSubTensorShapes[clusterId]);
            }
            generateWorkloads(builder, origOp, costParams, mpeMode, isTileOverDimsSupported, costModel, log,
                              clusterIdAttr, ShapeRef(preODUOffset));
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

//
// WorkloadDescriptor-based API
//

namespace {

// Generate workload descriptors for a single cluster (or non-distributed case).
// This mirrors generateWorkloads() but returns WorkloadDescriptors instead of creating ops.
SmallVector<VPU::WorkloadDescriptor, 4> generateWorkloadDescriptors(
        VPU::NCEOpInterface origOp, const VPUIP::WorkloadCostParams& costParams, VPU::MPEMode mpeMode,
        ArrayRef<bool> isTileOverDimsSupported, VPUNN::VPUCostModel& costModel, Logger log,
        std::optional<int64_t> clusterId = std::nullopt, ShapeRef subTensorOffset = {}) {
    auto ctx = origOp.getContext();
    SmallVector<VPU::WorkloadDescriptor, 4> result;

    VPUIP::DpuTiler dpuTiler(costParams.preODUShape, mpeMode);
    VPUIP::WorkloadSplitPool splitPoolSet;
    dpuTiler.tileOverH(costParams.numDPU, splitPoolSet);

    if (costParams.outputShape.size() == 5) {
        int64_t cluster = clusterId.value_or(0);
        const Shape offsets = subTensorOffset.empty() ? Shape{0, 0, 0, 0, 0} : Shape(subTensorOffset);
        result.push_back(VPU::WorkloadDescriptor{offsets,
                                                 costParams.outputShape,
                                                 PadInfo(0, 0, 0, 0),
                                                 VPU::MPEMode::CUBOID_16x16,
                                                 cluster,
                                                 {},
                                                 {}});
        return result;
    }

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

    auto splitPool = to_std_vector(splitPoolSet);
    VPUX_THROW_WHEN(splitPool.empty(), "Workload split pool is empty");

    const auto isAutopadODUEnabled = config::hasAutoPaddingODU(getModuleOp(origOp));

    std::vector<int64_t> splitPoolCosts(splitPool.size(), 0);
    for (const auto ind : irange(splitPool.size())) {
        auto& curSplit = splitPool[ind];

        if (clusterId.has_value()) {
            for (auto& wl : curSplit) {
                auto& outTile = std::get<0>(wl);
                addSubTensorOffset(outTile, subTensorOffset);
            }
        }
        const auto logCb = [&](const formatv_object_base& msg) {
            log.trace("{0}", msg.str());
        };
        splitPoolCosts[ind] = computeSplitCost(ctx, curSplit, costParams, costModel, isAutopadODUEnabled, logCb);
    }

    const auto bestSplitInd = std::min_element(splitPoolCosts.begin(), splitPoolCosts.end()) - splitPoolCosts.begin();
    if (splitPoolCosts[bestSplitInd] >= VPU::INVALID_COST_BASE) {
        log.setName("GenerateWorkloads");
        log.debug("An INVALID_COST is caught for bestSplit when calling VPUNN. You can pass a logCb with LOG_ERROR "
                  "level to print debug info in `computeSplitCostByArch` function and report to E#83609 if necessary");
        log.nest().debug("bestSplit cost value: {0}", splitPoolCosts[bestSplitInd]);
    }
    const auto& bestSplit = splitPool[bestSplitInd];

    if (mlir::dyn_cast<VPU::NCEConvolutionOp>(origOp.getOperation()) && bestSplit.size() > 1) {
        VPUX_THROW("NCE Convolution best split can't contain multiple variants per invariant (cluster). Only one "
                   "variant per invariant is allowed.");
    }
    origOp->setAttr(DPUCost, getIntAttr(origOp->getContext(), splitPoolCosts[bestSplitInd]));

    const auto kernel = origOp.getKernelSizeVal();
    const auto strides = origOp.getStridesVal();

    for (const auto& wl : bestSplit) {
        const auto& outTile = std::get<0>(wl);
        const auto wlMpeMode = std::get<1>(wl);

        const auto padsTileConf =
                backInferPadsTile(outTile, costParams.fullInputShape, costParams.padInfo, kernel, strides);

        result.push_back(
                VPU::WorkloadDescriptor{outTile.offsets, outTile.shape, padsTileConf, wlMpeMode, clusterId, {}, {}});
    }

    return result;
}

// Compute workload descriptors via heuristic path.
// Used by the SCF pass where distributed tensor types are not present
// (multi-clustering is represented by scf.forall instead).
SmallVector<VPU::WorkloadDescriptor, 4> computeDescriptorsHeuristic(VPU::NCEOpInterface origOp,
                                                                    VPUIP::WorkloadCostParams& costParams,
                                                                    VPU::MPEMode mpeMode,
                                                                    ArrayRef<bool> isTileOverDimsSupported,
                                                                    VPUNN::VPUCostModel& costModel, Logger log) {
    return generateWorkloadDescriptors(origOp, costParams, mpeMode, isTileOverDimsSupported, costModel, log);
}

}  // namespace

SmallVector<VPU::WorkloadDescriptor, 4> vpux::VPU::computeWorkloadDescriptors(VPU::NCEOpInterface nceOp,
                                                                              config::ArchKind arch, int64_t numDPU,
                                                                              VPUNN::VPUCostModel& costModel,
                                                                              int64_t& dpuCost, Logger log) {
    const auto mpeMode = getNCEHeuristicMPEMode(nceOp);
    auto params = VPU::getWorkloadCostParam(nceOp, arch, numDPU);
    auto isTileOverDimsSupported = getSupportedWorkloadSplitDim(nceOp, mpeMode);

    auto descriptors =
            computeDescriptorsHeuristic(nceOp, params, mpeMode, ArrayRef(isTileOverDimsSupported), costModel, log);

    // DPUCost was set on the op by generateWorkloadDescriptors
    if (auto costAttr = nceOp->getAttrOfType<mlir::IntegerAttr>(DPUCost)) {
        dpuCost = costAttr.getInt();
    }

    return descriptors;
}

SmallVector<VPU::WorkloadDescriptor, 4> vpux::VPU::computeWorkloadDescriptorsForSCF(
        VPU::NCEOpInterface nceOp, VPUIP::WorkloadCostParams& costParams, ShapeRef outputShape,
        VPUNN::VPUCostModel& costModel, int64_t& dpuCost, Logger log) {
    // Compute MPE mode from the provided output shape (not from the op's type).
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getOperand(0).getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
    const auto mpeMode = nceOp.getMpeMode(inputType.getElementType(), outputType.getElementType(), outputShape);

    auto isTileOverDimsSupported = getSupportedWorkloadSplitDim(nceOp, mpeMode);

    auto descriptors =
            computeDescriptorsHeuristic(nceOp, costParams, mpeMode, ArrayRef(isTileOverDimsSupported), costModel, log);

    if (auto costAttr = nceOp->getAttrOfType<mlir::IntegerAttr>(DPUCost)) {
        dpuCost = costAttr.getInt();
    }

    return descriptors;
}

SmallVector<VPU::DPUWorkloadOp> vpux::VPU::collectAllWorkloads(VPU::NCEOpInterface nceOp) {
    SmallVector<VPU::DPUWorkloadOp> result;
    nceOp.getWorkloads().walk([&](VPU::DPUWorkloadOp wl) {
        result.push_back(wl);
    });
    return result;
}

void vpux::VPU::materializeWorkloads(VPU::NCEOpInterface nceOp, mlir::OpBuilder& builder,
                                     ArrayRef<WorkloadDescriptor> workloads) {
    auto ctx = builder.getContext();
    auto& workloadRegion = nceOp.getWorkloads();
    if (workloadRegion.empty()) {
        workloadRegion.emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&workloadRegion.front());

    for (const auto& wl : workloads) {
        mlir::IntegerAttr clusterIdAttr = wl.clusterId.has_value() ? getIntAttr(ctx, wl.clusterId.value()) : nullptr;
        auto padAttr = VPU::getPaddingAttr(ctx, wl.padding);

        if (!wl.inOffsets.empty() && !wl.inSizes.empty()) {
            const auto outOffsetsAttr = mlir::DenseI64ArrayAttr::get(ctx, wl.outOffsets.raw());
            const auto outSizesAttr = mlir::DenseI64ArrayAttr::get(ctx, wl.outSizes.raw());
            const auto inOffsetsAttr = mlir::DenseI64ArrayAttr::get(ctx, wl.inOffsets.raw());
            const auto inSizesAttr = mlir::DenseI64ArrayAttr::get(ctx, wl.inSizes.raw());
            builder.create<DPUWorkloadOp>(nceOp->getLoc(), outOffsetsAttr, outSizesAttr, inOffsetsAttr, inSizesAttr,
                                          padAttr, VPU::MPEModeAttr::get(ctx, wl.mpeMode), clusterIdAttr);
        } else {
            nceOp.addWorkload(builder, nceOp->getLoc(), wl.outOffsets, wl.outSizes, padAttr, wl.mpeMode, clusterIdAttr);
        }
    }
}

void vpux::VPU::materializeWorkloadsDynamic(VPU::NCEOpInterface nceOp, mlir::OpBuilder& builder,
                                            ArrayRef<WorkloadDescriptor> baseDescriptors, mlir::Value dynChannelCount,
                                            ArrayRef<int64_t> supportedChannels, Logger log) {
    VPUX_THROW_WHEN(baseDescriptors.empty(), "materializeWorkloadsDynamic: no base descriptors");
    VPUX_THROW_WHEN(supportedChannels.empty(), "materializeWorkloadsDynamic: no supported channels");

    auto loc = nceOp->getLoc();
    auto ctx = builder.getContext();
    auto& workloadRegion = nceOp.getWorkloads();
    if (workloadRegion.empty()) {
        workloadRegion.emplaceBlock();
    }

    // Use the first base descriptor to determine MPE mode and non-channel dimensions.
    // For depthwise ops, all workloads share the same spatial dims and MPE mode.
    const auto& baseWl = baseDescriptors[0];
    const auto mpeMode = VPU::MPEModeAttr::get(ctx, baseWl.mpeMode);

    // Non-channel dimensions from the base descriptor (computed for max-bounds shape)
    const auto outN = baseWl.outSizes[Dims4D::Act::N];
    const auto outH = baseWl.outSizes[Dims4D::Act::H];
    const auto outW = baseWl.outSizes[Dims4D::Act::W];
    const auto padding = baseWl.padding;

    // Determine if back-infer is needed (if base descriptors have input workloads)
    const bool hasInputWorkloads = !baseWl.inOffsets.empty() && !baseWl.inSizes.empty();

    // Non-channel input dims (if applicable)
    int64_t inN = 0, inH = 0, inW = 0;
    if (hasInputWorkloads) {
        inN = baseWl.inSizes[Dims4D::Act::N];
        inH = baseWl.inSizes[Dims4D::Act::H];
        inW = baseWl.inSizes[Dims4D::Act::W];
    }

    mlir::IntegerAttr clusterIdAttr =
            baseWl.clusterId.has_value() ? getIntAttr(ctx, baseWl.clusterId.value()) : nullptr;

    // === Emit arith ops to compute loop trip counts ===
    // supportedChannels is in descending order, e.g. {64, 32, 16}.
    // The greedy split is equivalent to: for each supported channel size (largest first),
    // take as many as possible, then move to the next smaller size.
    //
    // For a single scf.for loop, we track:
    //   - n_i = number of workloads with supportedChannels[i]
    //   - total workloads = sum(n_i)
    //   - per-iteration: determine which "phase" we're in based on cumulative counts
    //
    // Phase boundaries: phase_end[0] = n_0, phase_end[1] = n_0 + n_1, ...
    // In iteration %j: if %j < phase_end[0] → size = supportedChannels[0], etc.

    // Insert arith ops BEFORE the NCE op, so they dominate the workload region
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(nceOp);

    auto constIndex = [&](int64_t val) -> mlir::Value {
        return builder.create<mlir::arith::ConstantIndexOp>(loc, val);
    };

    // Normalize supportedChannels and ensure the minimum HW channel alignment is available.
    // Note: kernel-optimization filtering may return only {32} on some archs, while runtime channel counts
    // can still be any multiple of VPU_CHANNEL_ALIGNMENT. Keep 16 as a fallback to avoid dropping remainder channels.
    SmallVector<int64_t> sortedChannels(supportedChannels.begin(), supportedChannels.end());
    sortedChannels.push_back(VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT);
    llvm::sort(sortedChannels, std::greater<int64_t>());
    sortedChannels.erase(llvm::unique(sortedChannels), sortedChannels.end());

    const int64_t alignment = VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    VPUX_THROW_WHEN(alignment <= 0, "materializeWorkloadsDynamic: invalid alignment {0}", alignment);

    // Keep only channel sizes that are multiples of the alignment.
    SmallVector<int64_t> validChannels;
    for (auto ch : sortedChannels) {
        if (ch % alignment == 0) {
            validChannels.push_back(ch);
        }
    }
    VPUX_THROW_WHEN(validChannels.empty(), "materializeWorkloadsDynamic: no valid channel sizes after filtering");

    // Compute: k = dynChannelCount / alignment (where alignment = smallest supported channel)
    // Then for each supported channel (in units of alignment):
    //   count_i = remaining / (supportedChannels[i] / alignment)
    //   remaining = remaining % (supportedChannels[i] / alignment)
    // Invariant: dynChannelCount is always a multiple of alignment (VPU_CHANNEL_ALIGNMENT = 16)
    // because NCE ops require aligned channel counts. The division is exact; no remainder is dropped.
    mlir::Value kVal = builder.create<mlir::arith::DivUIOp>(loc, dynChannelCount, constIndex(alignment));

    SmallVector<mlir::Value> phaseEnds;  // cumulative sum
    mlir::Value remaining = kVal;
    mlir::Value cumulativeCount = constIndex(0);

    for (auto channelSize : validChannels) {
        int64_t units = channelSize / alignment;
        auto unitsVal = constIndex(units);
        auto count = builder.create<mlir::arith::DivUIOp>(loc, remaining, unitsVal);
        remaining = builder.create<mlir::arith::RemUIOp>(loc, remaining, unitsVal);
        cumulativeCount = builder.create<mlir::arith::AddIOp>(loc, cumulativeCount, count);
        phaseEnds.push_back(cumulativeCount);
    }

    mlir::Value totalWorkloads = cumulativeCount;  // = phaseEnds.back()

    log.trace("materializeWorkloadsDynamic: validChannels={0}, emitting scf.for loop", validChannels);

    // Pre-create constants that will be used inside the scf.for body.
    // These are created before the NCE op so they dominate both the workload region and
    // the scf.for body (NCE workload region is NOT IsolatedFromAbove).
    mlir::Value c0 = constIndex(0);
    mlir::Value c1 = constIndex(1);

    // Constants for each supported channel size (used in arith.select inside the loop)
    SmallVector<mlir::Value> channelSizeConstants;
    for (auto channelSize : validChannels) {
        channelSizeConstants.push_back(constIndex(channelSize));
    }

    // === Emit scf.for inside the workload region ===
    builder.setInsertionPointToEnd(&workloadRegion.front());

    // scf.for %i = 0 to %totalWorkloads step 1 iter_args(%chOffset = 0) -> index
    auto forOp = builder.create<mlir::scf::ForOp>(
            loc, c0, totalWorkloads, c1, mlir::ValueRange{c0},
            [&](mlir::OpBuilder& bodyBuilder, mlir::Location bodyLoc, mlir::Value iv, mlir::ValueRange iterArgs) {
                mlir::Value chOffset = iterArgs[0];

                // Determine channel size for this iteration based on which phase we're in.
                // Start from the last (smallest) channel and work up via nested selects:
                //   size = supportedChannels[last]
                //   if iv < phaseEnds[last-1]: size = supportedChannels[last-1]
                //   ...
                mlir::Value chSize = channelSizeConstants.back();
                for (int64_t p = static_cast<int64_t>(validChannels.size()) - 2; p >= 0; --p) {
                    auto cmp = bodyBuilder.create<mlir::arith::CmpIOp>(bodyLoc, mlir::arith::CmpIPredicate::ult, iv,
                                                                       phaseEnds[p]);
                    chSize = bodyBuilder.create<mlir::arith::SelectOp>(bodyLoc, cmp, channelSizeConstants[p], chSize);
                }

                // Build OpFoldResult arrays for the DPU.Workload
                // outOffsets: [0, %chOffset, 0, 0], outSizes: [N, %chSize, H, W]
                SmallVector<mlir::OpFoldResult, 4> outOffsets = {
                        bodyBuilder.getIndexAttr(0), mlir::OpFoldResult(chOffset), bodyBuilder.getIndexAttr(0),
                        bodyBuilder.getIndexAttr(0)};
                SmallVector<mlir::OpFoldResult, 4> outSizes = {
                        bodyBuilder.getIndexAttr(outN), mlir::OpFoldResult(chSize), bodyBuilder.getIndexAttr(outH),
                        bodyBuilder.getIndexAttr(outW)};
                SmallVector<mlir::OpFoldResult, 4> padOFR = {
                        bodyBuilder.getIndexAttr(padding.left), bodyBuilder.getIndexAttr(padding.right),
                        bodyBuilder.getIndexAttr(padding.top), bodyBuilder.getIndexAttr(padding.bottom)};

                if (hasInputWorkloads) {
                    // For depthwise ops: input channel offset/size = output channel offset/size
                    SmallVector<mlir::OpFoldResult, 4> inOffsets = {
                            bodyBuilder.getIndexAttr(0), mlir::OpFoldResult(chOffset),
                            bodyBuilder.getIndexAttr(baseWl.inOffsets[Dims4D::Act::H]),
                            bodyBuilder.getIndexAttr(baseWl.inOffsets[Dims4D::Act::W])};
                    SmallVector<mlir::OpFoldResult, 4> inSizes = {
                            bodyBuilder.getIndexAttr(inN), mlir::OpFoldResult(chSize), bodyBuilder.getIndexAttr(inH),
                            bodyBuilder.getIndexAttr(inW)};

                    bodyBuilder.create<DPUWorkloadOp>(bodyLoc, outOffsets, outSizes, inOffsets, inSizes, padOFR,
                                                      mpeMode, clusterIdAttr);
                } else {
                    bodyBuilder.create<DPUWorkloadOp>(bodyLoc, outOffsets, outSizes, padOFR, mpeMode, clusterIdAttr);
                }

                auto nextOffset = bodyBuilder.create<mlir::arith::AddIOp>(bodyLoc, chOffset, chSize);
                bodyBuilder.create<mlir::scf::YieldOp>(bodyLoc, mlir::ValueRange{nextOffset.getResult()});
            });

    // CollectLoops / FullUnrollSCFLoopPass skips it —
    // the trip count is dynamic (computed from tensor.dim/arith.divui).
    forOp->setAttr("vpux.workload_loop", builder.getUnitAttr());
}

namespace {

// Split a single workload descriptor along channels according to supportedChannels.
// Returns the replacement descriptors (may be 1 if already valid, or multiple if split needed).
SmallVector<VPU::WorkloadDescriptor, 4> splitDescriptorChannels(const VPU::WorkloadDescriptor& wl,
                                                                ArrayRef<int64_t> supportedChannels) {
    auto wlChannels = wl.outSizes[Dims4D::Act::C];
    if (llvm::find(supportedChannels, wlChannels) != supportedChannels.end()) {
        return {wl};
    }

    auto newChannelSizes = splitWorkloadChannel(wlChannels, supportedChannels);
    VPUX_THROW_WHEN(newChannelSizes.empty(), "splitWorkloadChannel failed: wlChannel={0}, supportedChannels={1}",
                    wlChannels, supportedChannels);

    SmallVector<VPU::WorkloadDescriptor, 4> result;
    auto channelOffset = wl.outOffsets[Dims4D::Act::C];

    for (auto channelSize : newChannelSizes) {
        VPU::WorkloadDescriptor newWl = wl;
        newWl.outSizes[Dims4D::Act::C] = channelSize;
        newWl.outOffsets[Dims4D::Act::C] = channelOffset;
        channelOffset += channelSize;
        result.push_back(std::move(newWl));
    }

    return result;
}

}  // namespace

void vpux::VPU::correctWorkloadDescriptors(VPU::NCEOpInterface nceOp, SmallVector<WorkloadDescriptor, 4>& workloads,
                                           mlir::MLIRContext* ctx, Logger log,
                                           SmallVector<int64_t>* outSupportedChannels) {
    auto* op = nceOp.getOperation();
    const auto& strategyFactory = VPU::getVPUStrategyFactory(ctx);

    // NCEPermute corrections are intentionally omitted
    const bool isDepthwise = isDepthwiseOp(op);

    // Build supported channels list
    auto supportedChannelsDW = strategyFactory->getSupportedChannelsDW();
    SmallVector<int64_t> supportedChannels;

    if (isDepthwise) {
        supportedChannels.assign(supportedChannelsDW.begin(), supportedChannelsDW.end());
    }

    // Small kernel optimization filtering
    if (NCEInvariant::doesOpSupportSmallKernelOptimization(op)) {
        SmallVector<int64_t> wlChannels;
        for (const auto& wl : workloads) {
            wlChannels.push_back(wl.outSizes[Dims4D::Act::C]);
        }
        const auto maxSlotsSum = VPUIP::getBarrierMaxSlotCount(nceOp);
        const auto optimizedChannels = strategyFactory->getChannelsSupportedByKernelOptimization(
                wlChannels, static_cast<int64_t>(maxSlotsSum));
        if (!optimizedChannels.empty()) {
            supportedChannels.assign(optimizedChannels.begin(), optimizedChannels.end());
        }
    }

    // Small spatial compute DW ops errata
    auto workloadSizeConstraint = strategyFactory->getWorkloadSizeConstraint();
    if (workloadSizeConstraint.doesDWOperationNeedWorkloadSplit(nceOp)) {
        supportedChannels = workloadSizeConstraint.getChannelsSupportedBySmallSpatialComputeDwOp(supportedChannels);
    }

    // ODU autopad: add the output channel count as a supported channel
    if (VPU::canAutopadOutput(op)) {
        const auto outputChannels = mlir::cast<NDTypeInterface>(op->getResult(0).getType()).getShape()[Dims4D::Act::C];
        if (outputChannels != mlir::ShapedType::kDynamic && outputChannels > 0) {
            supportedChannels.push_back(outputChannels);
        }
    }

    // Early exit if no corrections needed
    if (supportedChannels.empty()) {
        if (outSupportedChannels) {
            outSupportedChannels->clear();
        }
        return;
    }

    // Apply corrections
    SmallVector<WorkloadDescriptor, 4> corrected;
    corrected.reserve(workloads.size());

    for (auto& wl : workloads) {
        // Channel split correction (depthwise, small kernel, etc.)
        auto splitWls = splitDescriptorChannels(wl, supportedChannels);
        corrected.append(splitWls);
    }

    workloads = std::move(corrected);
    if (outSupportedChannels) {
        *outSupportedChannels = std::move(supportedChannels);
    }
    log.trace("Corrected workloads for op '{0}' at '{1}': {2} workloads", op->getName(), op->getLoc(),
              workloads.size());
}

namespace {

int64_t getInputWorkloadStartCh(VPU::NCEOpInterface nceOp, int64_t outputStartCh) {
    return llvm::TypeSwitch<mlir::Operation*, int64_t>(nceOp.getOperation())
            .Case<NCEConvolutionOp, NCECompressConvolutionOp, NCEInterpolateOp, NCEReduceOp, NCEMatMulOp>(
                    [&](mlir::Operation*) {
                        return int64_t(0);
                    })
            .Case<NCEEltwiseOp, NCEDepthConvolutionOp, NCEMaxPoolOp, NCEAveragePoolOp, NCEPermuteOp>(
                    [&](mlir::Operation*) {
                        return outputStartCh;
                    })
            .Default([&](mlir::Operation*) -> int64_t {
                VPUX_THROW("Unsupported operation type for input workload: {0}", nceOp);
            });
}

int64_t getInputWorkloadSizeCh(VPU::NCEOpInterface nceOp, int64_t outputSizeCh, int64_t outputStartCh,
                               int64_t fullInputChannels) {
    return llvm::TypeSwitch<mlir::Operation*, int64_t>(nceOp.getOperation())
            .Case<NCEConvolutionOp, NCEInterpolateOp, NCEReduceOp, NCEMatMulOp>([&](mlir::Operation*) {
                return fullInputChannels;
            })
            .Case<NCEEltwiseOp>([&](mlir::Operation*) {
                // Eltwise has IC == OC element-wise: back-infer the workload's input channel size
                // directly from its output channel size. Using the bounded fullInputChannels here
                // would overshoot when ODU autopad expands the output.
                return outputSizeCh;
            })
            .Case<NCEDepthConvolutionOp, NCEMaxPoolOp, NCEAveragePoolOp>([&](mlir::Operation* op) {
                // Depthwise / pooling ops: back-inferred input channel size for this workload is
                // the workload's outputSizeCh, aligned up to the input channel alignment.
                if (auto alignedOp = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
                    return vpux::alignValUp(outputSizeCh, alignedOp.getInputChannelAlignment());
                }
                return outputSizeCh;
            })
            .Case<NCECompressConvolutionOp>([&](mlir::Operation*) {
                return VPU::NCEInvariant::getAlignment(
                        mlir::cast<vpux::NDTypeInterface>(nceOp.getWeightsOperand().getType()).getElementType());
            })
            .Case<NCEPermuteOp>([&](mlir::Operation*) {
                if ((outputStartCh + outputSizeCh) > fullInputChannels) {
                    return fullInputChannels - outputStartCh;
                }
                return outputSizeCh;
            })
            .Default([&](mlir::Operation*) -> int64_t {
                VPUX_THROW("Unsupported operation type for input workload: {0}", nceOp);
            });
}

void backInferWorkload(VPU::NCEOpInterface nceOp, VPU::WorkloadDescriptor& wl, ShapeRef fullInputShape) {
    const bool is5D = wl.outOffsets.size() == DimsGroups5D::Act::numDims;
    const auto dimC = is5D ? DimsGroups5D::Act::C : Dims4D::Act::C;
    const auto dimH = is5D ? DimsGroups5D::Act::H : Dims4D::Act::H;
    const auto dimW = is5D ? DimsGroups5D::Act::W : Dims4D::Act::W;
    const auto kYind = (is5D ? DimsGroups5D::Kernel::Y : Dims4D::Kernel::Y).ind();
    const auto kXind = (is5D ? DimsGroups5D::Kernel::X : Dims4D::Kernel::X).ind();

    const auto fullInputChannels = fullInputShape[dimC];
    const auto fullInputHeight = fullInputShape[dimH];
    const auto fullInputWidth = fullInputShape[dimW];

    const auto kernelSz = nceOp.getKernelSizeVal();
    const auto strides = nceOp.getStridesVal();

    // Resolve kernel padding. In the SCF flow tensor.pad on the activation carries the actual
    // kernel padding (the op's pad attribute is zeroed out by SCF tiling). Prefer tensor.pad
    // when present and statically known; otherwise fall back to the op's pad attribute.
    int64_t kernelPadTop = 0, kernelPadLeft = 0, kernelPadBottom = 0, kernelPadRight = 0;
    mlir::Value act = nceOp->getOperand(0);
    while (auto castOp = act.getDefiningOp<mlir::tensor::CastOp>()) {
        act = castOp.getSource();
    }
    auto padOp = act.getDefiningOp<mlir::tensor::PadOp>();
    const auto isStatic = [](mlir::ArrayRef<int64_t> arr) {
        return llvm::none_of(arr, [](int64_t v) {
            return mlir::ShapedType::isDynamic(v);
        });
    };
    if (padOp) {
        const auto lowPad = padOp.getStaticLow();
        const auto highPad = padOp.getStaticHigh();
        if (isStatic(lowPad) && isStatic(highPad)) {
            kernelPadTop = lowPad[dimH.ind()];
            kernelPadLeft = lowPad[dimW.ind()];
            kernelPadBottom = highPad[dimH.ind()];
            kernelPadRight = highPad[dimW.ind()];
        } else {
            // Dynamic pads (position-dependent in the SCF flow): try to evaluate each
            // dimension individually. Border slices will have non-zero padding on the
            // corresponding edge while middle slices will have zero.
            auto mixedLow = padOp.getMixedLowPad();
            auto mixedHigh = padOp.getMixedHighPad();
            auto evalPad = [](mlir::OpFoldResult ofr) -> int64_t {
                if (auto attr = mlir::dyn_cast<mlir::Attribute>(ofr)) {
                    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
                        return intAttr.getInt();
                    }
                }
                if (auto val = mlir::dyn_cast<mlir::Value>(ofr)) {
                    if (auto constOp = val.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
                        return constOp.value();
                    }
                    if (auto constOp = val.getDefiningOp<mlir::arith::ConstantIntOp>()) {
                        return constOp.value();
                    }
                }
                // Truly dynamic and non-foldable: workloads are generated for the bounded
                // (max) shape which corresponds to an interior tile where padding is zero.
                return 0;
            };
            kernelPadTop = evalPad(mixedLow[dimH.ind()]);
            kernelPadLeft = evalPad(mixedLow[dimW.ind()]);
            kernelPadBottom = evalPad(mixedHigh[dimH.ind()]);
            kernelPadRight = evalPad(mixedHigh[dimW.ind()]);
        }
    } else {
        const auto padding = nceOp.getPad();
        kernelPadTop = padding.getTop().getInt();
        kernelPadLeft = padding.getLeft().getInt();
        kernelPadBottom = padding.getBottom().getInt();
        kernelPadRight = padding.getRight().getInt();
    }

    // Spatial back-inference using inputForOutputDim.
    // Note on bounded sizes: in the SCF flow `fullInputHeight`/`fullInputWidth` are the bounded
    // (max-iteration) input dims taken from the op's operand type. The output workloads
    // (wl.outOffsets/outSizes) are also generated against the bounded output shape, so both sides
    // of inputForOutputDim live in the same max-bounds frame and the clamped input range is
    // consistent. Only the channel dimension is iteration-dynamic in this flow (patched up by
    // materializeWorkloadsDynamic); spatial dims are fixed at max-bounds in the descriptor.
    const DimRange outHeightTile(wl.outOffsets[dimH], wl.outOffsets[dimH] + wl.outSizes[dimH]);
    auto [inHeightTile, ignoreH1, ignoreH2] = vpux::inputForOutputDim(
            outHeightTile, kernelSz[kYind], strides[kYind], {0, fullInputHeight}, kernelPadTop, kernelPadBottom);

    const DimRange outWidthTile(wl.outOffsets[dimW], wl.outOffsets[dimW] + wl.outSizes[dimW]);
    auto [inWidthTile, ignoreW1, ignoreW2] = vpux::inputForOutputDim(
            outWidthTile, kernelSz[kXind], strides[kXind], {0, fullInputWidth}, kernelPadLeft, kernelPadRight);

    const auto inStartC = getInputWorkloadStartCh(nceOp, wl.outOffsets[dimC]);
    const auto inSizeC = getInputWorkloadSizeCh(nceOp, wl.outSizes[dimC], wl.outOffsets[dimC], fullInputChannels);
    const auto inSizeH = inHeightTile.end - inHeightTile.begin;
    const auto inSizeW = inWidthTile.end - inWidthTile.begin;

    VPUX_THROW_WHEN(
            mlir::isa<VPU::NCEPermuteOp>(nceOp.getOperation()) && inWidthTile.begin != 0 && inSizeW != fullInputWidth,
            "HW Permute does not support workload segmentation over W. Input workload start = {0}, "
            "Input workload size = {1}",
            inWidthTile.begin, inSizeW);

    if (is5D) {
        wl.inOffsets = Shape{wl.outOffsets[DimsGroups5D::Act::G], wl.outOffsets[DimsGroups5D::Act::N], inStartC,
                             inHeightTile.begin, inWidthTile.begin};
        wl.inSizes =
                Shape{wl.outSizes[DimsGroups5D::Act::G], wl.outSizes[DimsGroups5D::Act::N], inSizeC, inSizeH, inSizeW};
    } else {
        wl.inOffsets = Shape{wl.outOffsets[Dims4D::Act::N], inStartC, inHeightTile.begin, inWidthTile.begin};
        wl.inSizes = Shape{wl.outSizes[Dims4D::Act::N], inSizeC, inSizeH, inSizeW};
    }
}

}  // namespace

void vpux::VPU::backInferInputWorkloads(VPU::NCEOpInterface nceOp, SmallVector<WorkloadDescriptor, 4>& workloads,
                                        Logger log) {
    auto* op = nceOp.getOperation();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
    const auto fullInputShape = inputType.getShape();

    for (auto& wl : workloads) {
        backInferWorkload(nceOp, wl, fullInputShape);
    }

    log.trace("Back-inferred input workloads for op '{0}' at '{1}': {2} workloads", op->getName(), op->getLoc(),
              workloads.size());
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

        if (mlir::dyn_cast<VPU::NCEConvolutionOp>(nceOp.getOperation()) && intraTileSplit.second.size() > 1) {
            VPUX_THROW("NCE Convolution best split can't contain multiple variants per invariant (cluster). Only one "
                       "variant per invariant is allowed.");
        }

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
    // VPUNN pre-split generates workloads in the op's output coordinate space, which for ops
    // with an active ODU transform (D2S/S2D) is post-ODU. The DPU hardware requires workload
    // offsets/shapes in pre-ODU space. This will require further adjusting in different parts of the code and will be
    // handled with E#221433.
    if (!VPU::getODUScaling(nceOp.getOperation()).empty()) {
        return false;
    }
    const auto isDistributedOp = getDistributedTensor(nceOp->getResult(0)) != nullptr &&
                                 getDistributedTensor(nceOp->getOperand(0)) != nullptr;
    if (!isDistributedOp) {
        return false;
    }
    return true;
}

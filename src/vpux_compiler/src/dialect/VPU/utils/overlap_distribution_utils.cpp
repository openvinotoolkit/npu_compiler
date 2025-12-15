//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <stack>

using namespace vpux;
using namespace VPU;

namespace {
SmallVector<Shape> getPerClusterEndOffset(ArrayRef<Shape> startOffset, ArrayRef<Shape> size) {
    const auto numDims = startOffset[0].size();
    const auto numClusters = startOffset.size();
    auto endOffset = SmallVector<Shape>(numClusters, Shape(numDims, 0));

    for (size_t cluster = 0; cluster < numClusters; cluster++) {
        for (size_t dim = 0; dim < numDims; dim++) {
            endOffset[cluster][Dim(dim)] = startOffset[cluster][Dim(dim)] + size[cluster][Dim(dim)] - 1;
        }
    }

    return endOffset;
}

bool isDistributedTensorSOH(VPU::DistributedTensorType distributedTensorType) {
    auto distribution = distributedTensorType.getDistribution();
    const auto mode = distribution.getMode().getValue();

    const bool isSegOverlappedMode =
            mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED;
    if (!isSegOverlappedMode) {
        return false;
    }

    const auto numTiles = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
    return numTiles[Dims4D::Act::H.ind()] != 1;
}

bool isValidCandidateForCMXConcat(mlir::Operation* maybeConcat) {
    auto concat = mlir::dyn_cast_or_null<VPU::ConcatOp>(maybeConcat);
    if (concat == nullptr) {
        return false;
    }

    // Minimal memory requirement for cmx concat - the concat's size should fit into CMX
    auto concatOutType = mlir::cast<vpux::NDTypeInterface>(concat.getOutput().getType());
    const auto concatSize = concatOutType.getTotalAllocSize().count();
    const auto totalCMX = getTotalCMXSize(maybeConcat).count();
    if (concatSize > totalCMX) {
        return false;
    }

    bool isProducerOrConsumerSOH = false;
    for (const auto& producerConcat : concat->getOperands()) {
        if (!mlir::isa_and_nonnull<VPU::NCEOpInterface>(producerConcat.getDefiningOp())) {
            return false;
        }

        if (auto copyOp = mlir::dyn_cast<VPU::CopyOp>(producerConcat.getDefiningOp())) {
            auto nceOp = copyOp->getOperand(0).getDefiningOp<VPU::NCEOpInterface>();
            if (!nceOp) {
                return false;
            }

            auto distributedTensorType =
                    mlir::dyn_cast<vpux::VPU::DistributedTensorType>(nceOp->getResult(0).getType());
            if (distributedTensorType == nullptr) {
                return false;
            }

            isProducerOrConsumerSOH = isDistributedTensorSOH(distributedTensorType);
        } else if (auto nceOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(producerConcat.getDefiningOp())) {
            if (nceOp.getMultiClusterStrategy().has_value()) {
                if (nceOp.getMultiClusterStrategy().value() == VPU::MultiClusterStrategy::SplitOverHeight) {
                    isProducerOrConsumerSOH = true;
                }
            }
        }
    }
    for (const auto& consumerConcat : concat->getUsers()) {
        if (mlir::isa_and_nonnull<mlir::func::ReturnOp>(consumerConcat)) {
            continue;
        }

        if (!mlir::isa_and_nonnull<VPU::NCEOpInterface>(consumerConcat)) {
            return false;
        }

        if (auto copyOp = mlir::dyn_cast<VPU::CopyOp>(consumerConcat)) {
            if (!copyOp->hasOneUse()) {
                return false;
            }
            auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(*copyOp->user_begin());
            if (!nceOp) {
                return false;
            }

            auto distributedTensorType =
                    mlir::dyn_cast<vpux::VPU::DistributedTensorType>(nceOp->getResult(0).getType());
            if (distributedTensorType == nullptr) {
                return false;
            }

            isProducerOrConsumerSOH = isDistributedTensorSOH(distributedTensorType);
        } else if (auto nceOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerConcat)) {
            if (nceOp.getMultiClusterStrategy().has_value()) {
                if (nceOp.getMultiClusterStrategy().value() == VPU::MultiClusterStrategy::SplitOverHeight) {
                    isProducerOrConsumerSOH = true;
                }
            }
        }
    }

    auto isOffsetOnH = [](mlir::ArrayAttr offset) {
        auto offsetVector = Shape(parseIntArrayAttr<int64_t>(offset));
        return offsetVector[Dims4D::Act::H] != 0;
    };

    bool isConcatOverH = false;
    if (concat.getStaticOffsets().has_value()) {
        const auto concatDims = concat.getStaticOffsetsAttr().getAsRange<mlir::ArrayAttr>();
        isConcatOverH = llvm::any_of(concatDims, isOffsetOnH);
    } else if (concat.getPerAxis().has_value()) {
        const auto concatAxis = concat.getPerAxis().value().getAxis().getValue().getSExtValue();
        isConcatOverH = concatAxis == Dims4D::Act::H.ind();
    }

    return !(isProducerOrConsumerSOH && isConcatOverH);
}

int64_t extractKernelTileAxis(ArrayRef<int64_t> numTiles) {
    VPUX_THROW_UNLESS(numTiles[Dims4D::Act::H.ind()] == 1 || numTiles[Dims4D::Act::W.ind()] == 1,
                      "Multidimension cluster tiling across H and W is not yet supported.");
    if (numTiles[Dims4D::Act::W.ind()] > 1) {
        return Dims4D::Kernel::X.ind();
    }
    return Dims4D::Kernel::Y.ind();
}

// Currently supported "passthrough" ops where Overlapped params are concerned
// Concat not on clustering dim, Eltwise with in-place attr and QuantizeCast act as sort of "passthrough" ops.
// For example, given the following subgraph:
// PASS_OP is (Eltwise in place) || (valid CMXConcat) || QuantizeCast
//
//     DPU0           DPU1
//    /    \         /   |
// DPU3      PASS_OP    DPU4
//          /     |
//      DPU2     DPU5
//
// Sibling ops will be DPU2, DPU3, DPU4 and DPU5 i.e. the siblings of the PASS_OP & its consumers

bool isBranchingPassthroughOp(mlir::Operation* op) {
    if (isValidCandidateForCMXConcat(op)) {
        return true;
    }

    if (auto eltwise = mlir::dyn_cast_or_null<VPU::NCEEltwiseOp>(op)) {
        if (eltwise.getIsInplace().value_or(false)) {
            return true;
        }
    }

    return false;
}

bool isTransparentOp(mlir::Operation* op) {
    return mlir::isa<VPU::QuantizeCastOp, VPU::GroupSparseTensorOp>(op);
}

using SiblingSearchStackFrame = std::pair<mlir::Value, int64_t>;

void findSiblings(mlir::Value operand, std::set<VPU::ClusteredOpInterface>& siblings,
                  std::set<llvm::hash_code>& visited, const int64_t level = 2) {
    std::stack<SiblingSearchStackFrame> stack;
    stack.emplace(operand, level);

    while (!stack.empty()) {
        auto currentStackFrame = stack.top();
        auto& currentOperand = std::get<0>(currentStackFrame);
        auto& currentLevel = std::get<1>(currentStackFrame);
        stack.pop();

        const auto operandHash = mlir::hash_value(currentOperand);
        bool notVisited = visited.insert(operandHash).second;
        if (!notVisited) {
            continue;
        }

        auto findSiblingsForPassthroughOrMultiInputOp = [&](mlir::Operation* user) {
            if (auto clusteredUser = mlir::dyn_cast<VPU::ClusteredOpInterface>(user)) {
                // if op is a sibling already, it's consumers/producers were already iterated through
                bool canInsert = siblings.insert(clusteredUser).second;
                if (!canInsert) {
                    return;
                }
            }

            if (currentLevel < 0) {
                return;
            }

            if (user->getResult(0) != currentOperand && isPassthroughOp(user)) {
                stack.emplace(user->getResult(0), currentLevel - 1);
            }

            for (const auto siblingOperand : user->getOperands()) {
                if (siblingOperand == currentOperand) {
                    continue;
                }
                stack.emplace(siblingOperand, currentLevel - 1);
                if (isTransparentOp(user)) {
                    for (const auto siblingUser : siblingOperand.getUsers()) {
                        if (isTransparentOp(siblingUser)) {
                            stack.emplace(siblingUser->getResult(0), currentLevel - 1);
                        }
                    }
                }
            }
        };

        if (currentOperand.getDefiningOp() != nullptr && isPassthroughOp(currentOperand.getDefiningOp())) {
            findSiblingsForPassthroughOrMultiInputOp(currentOperand.getDefiningOp());
        }

        for (const auto& sibling : currentOperand.getUsers()) {
            if (isPassthroughOp(sibling)) {
                findSiblingsForPassthroughOrMultiInputOp(sibling);
            } else if (mlir::isa_and_nonnull<VPU::NCEEltwiseOp>(sibling)) {
                findSiblingsForPassthroughOrMultiInputOp(sibling);
            } else if (auto clusteredUser = mlir::dyn_cast<VPU::ClusteredOpInterface>(sibling)) {
                siblings.insert(clusteredUser);
            }
        }
    }
}

}  // namespace

bool vpux::VPU::isPassthroughOp(mlir::Operation* op) {
    // Given the following subgraph:
    //             Producer
    //         /             |
    // [QuantizeCast]  [QuantizeCast]
    //       |               |
    //     NceOp0          NceOp1
    // NceOp0 and NceOp1 should be aware of each other as siblings to be able to properly set their input distributions
    // TODO: 104112 avoid spilling due to other view ops besides of QuantizeCast
    if (isTransparentOp(op)) {
        return true;
    }

    return isBranchingPassthroughOp(op);
}

OverlapDistributionParams vpux::VPU::getOverlappedDistributionParameters(ArrayRef<VPU::ClusteredOpInterface> opSubgraph,
                                                                         int64_t kernelDistributionAxis,
                                                                         bool equalComputeAndMemoryView) {
    auto kernel = SmallVector<int64_t>{1, 1};
    auto pads = VPU::Padding(0, 0, 0, 0);
    auto strides = SmallVector<int64_t>{1, 1};

    SmallVector<VPU::NCEOpInterface> nceOpCandidates;
    for (auto clusteredOp : opSubgraph) {
        // clusteredOp with SOHO strategy satisfy below SOH condition too so won't be dropped
        if (mlir::isa<VPU::NCEOpInterface>(clusteredOp.getOperation()) &&
            clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
            nceOpCandidates.push_back(mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation()));
        }
    }

    if (nceOpCandidates.empty()) {
        return OverlapDistributionParams(kernel, pads, strides);
    }

    // For now just take the highest kernel
    // As we have better representation in distributedBuffer, switch to computing the
    // actual shapes per clusters

    auto largestKernel = 0;
    auto largestIndex = 0;
    for (auto it : nceOpCandidates | indexed) {
        auto kernelSize = it.value().getKernelSizeVal()[kernelDistributionAxis];
        if (kernelSize > largestKernel) {
            largestKernel = kernelSize;
            largestIndex = it.index();
        }
    }

    kernel = nceOpCandidates[largestIndex].getKernelSizeVal();
    pads = VPU::Padding::getClassFromAttr(nceOpCandidates[largestIndex].getPad());
    strides = nceOpCandidates[largestIndex].getStridesVal();

    return OverlapDistributionParams(kernel, pads, strides, equalComputeAndMemoryView);
}

std::pair<Shape, Shape> getActualDataMemOffsetsAndShapes(VPU::SparseTensorType sparseNceOpInputType,
                                                         ShapeRef effectiveMemOffsets, ShapeRef effectiveMemShape) {
    auto dataType = mlir::dyn_cast_or_null<NDTypeInterface>(sparseNceOpInputType.getData());
    const auto dataShape = getBoundedShape(dataType);

    auto memOffsets = Shape(dataShape.size());
    auto memShapes = Shape(dataShape.size());
    sparseNceOpInputType.getSeAttr().extractTile(effectiveMemOffsets, effectiveMemShape, dataShape, memOffsets,
                                                 memShapes);

    return {memOffsets, memShapes};
}

// Version with extension of halo
OverlapDistributionParams vpux::VPU::getOverlappedDistributionParameters(
        NDTypeInterface tensorType, ArrayRef<VPU::ClusteredOpInterface> consumerSubgraph, const int64_t numClusters,
        ArrayRef<int64_t> numTiles, const bool uniformDistributedSegments, const vpux::TileInfo& tileInfo) {
    VPUX_THROW_WHEN(tensorType == nullptr, "getOverlappedDistributionParameters: tensorType cannot be nullptr");

    auto neutralKernel = SmallVector<int64_t>{1, 1};
    auto neutralPads = VPU::Padding(0, 0, 0, 0);
    auto neutralStrides = SmallVector<int64_t>{1, 1};

    const auto distributionMode = VPU::DistributionMode::OVERLAPPED;
    auto neutralDistribution =
            VPU::DistributionInfo(distributionMode, numTiles, neutralKernel, neutralStrides, neutralPads, numClusters,
                                  {}, uniformDistributedSegments, {}, {}, {}, {}, {});

    const auto shape = getBoundedShape(tensorType);
    const auto computeShapes = getPerClusterComputeShapes(shape, neutralDistribution);
    const auto computeOffsets = getPerClusterComputeShapeOffsets(shape, neutralDistribution);
    auto memoryShapes = computeShapes;
    auto memoryOffsets = computeOffsets;

    SmallVector<VPU::NCEOpInterface> nceOpCandidates;
    for (auto clusteredOp : consumerSubgraph) {
        if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation())) {
            nceOpCandidates.push_back(nceOp);
        }
    }

    if (nceOpCandidates.empty()) {
        return OverlapDistributionParams(arrayOfArrayFromShape(memoryShapes), arrayOfArrayFromShape(memoryOffsets),
                                         arrayOfArrayFromShape(computeShapes), arrayOfArrayFromShape(computeOffsets));
    }

    const auto kernelTileAxis = extractKernelTileAxis(numTiles);
    const auto clusteringAxis = VPU::getDistributedTilingAxis(numTiles);
    const auto tileOffset = !tileInfo.offsets.empty() ? tileInfo.offsets[Dim(clusteringAxis)] : 0;

    // TODO: E#112803 Add support for extended memory view for sparse types with SETable; remove throw after
    // implementation is done.
    const auto sparseTensor = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tensorType);
    const bool origTensorHasSETable = (sparseTensor != nullptr) && (sparseTensor.getSeAttr() != nullptr);
    VPUX_THROW_WHEN(origTensorHasSETable && nceOpCandidates.size() != 1,
                    "Equalizing memory view between a SEP op and another NCE op is not supported, yet.");

    for (auto nceOp : nceOpCandidates) {
        auto kernelSize = nceOp.getKernelSizeVal();
        auto stridesSize = nceOp.getStridesVal();
        auto padsSize = VPU::Padding::getClassFromAttr(nceOp.getPad());

        // WA to get correct pad start when in a tiling scenario
        // TODO: Proper fix - E#112801
        if (tileOffset != 0) {
            if (kernelTileAxis == Dims4D::Kernel::Y.ind()) {
                padsSize =
                        VPU::Padding(padsSize.getLeftPad(), padsSize.getRightPad(), /*top*/ 0, padsSize.getBottomPad());
            } else {
                padsSize =
                        VPU::Padding(/*left*/ 0, padsSize.getRightPad(), padsSize.getTopPad(), padsSize.getBottomPad());
            }
        }

        auto consumerDistr = VPU::DistributionInfo(distributionMode, numTiles, kernelSize, stridesSize, padsSize,
                                                   numClusters, {}, uniformDistributedSegments, {}, {}, {}, {}, {});

        auto tensorShape = shape.toValues();
        const auto sparseNceOpInputType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(nceOp->getOperand(0).getType());
        const bool currentNceOpHasSETable =
                sparseNceOpInputType != nullptr && sparseNceOpInputType.getSeAttr() != nullptr;

        if (currentNceOpHasSETable) {
            if (nceOpCandidates.size() != 1) {
                // ignore consumers with SETable if there are other consumers
                continue;
            }

            // we decide memory view for producer op with only SEPOp as consumer
            if (!origTensorHasSETable) {
                tensorShape = sparseNceOpInputType.getSeAttr().inferOutputShape(shape);
            }
        }

        auto consumerMemoryShapesOpt = getPerClusterMemoryShapes(tensorShape, consumerDistr);
        if (!consumerMemoryShapesOpt.has_value()) {
            // op is not SOH/SOW Compatible
            continue;
        }

        auto consumerMemoryShapes = consumerMemoryShapesOpt.value();
        auto consumerMemoryOffsets = getPerClusterMemoryShapeOffsets(tensorShape, consumerDistr);

        if (origTensorHasSETable) {
            memoryShapes = std::move(consumerMemoryShapes);
            memoryOffsets = std::move(consumerMemoryOffsets);
            continue;
        }

        const bool opHasConsumerWithSETable = currentNceOpHasSETable && !origTensorHasSETable;
        // E#112803 support for SEP should be added
        // E#173578 restricting to child op DepthConvolution.
        // For other ops, it is possible to cause WLM rollback, caused by scheduler.
        auto isAffectingWLM = !mlir::isa_and_nonnull<VPU::NCEDepthConvolutionOp>(nceOp);

        auto arch = config::getArch(nceOp.getOperation());
        // E#173578 WLM rollback happens only for NPU4
        if (arch >= config::ArchKind::NPU50XX) {
            isAffectingWLM = false;
        }

        if (opHasConsumerWithSETable && (nceOpCandidates.size() != 1 || isAffectingWLM)) {
            continue;
        }

        auto initDistribution =
                OverlapDistributionParams(arrayOfArrayFromShape(memoryShapes), arrayOfArrayFromShape(memoryOffsets),
                                          arrayOfArrayFromShape(computeShapes), arrayOfArrayFromShape(computeOffsets));

        for (int64_t cluster = 0; cluster < numClusters; ++cluster) {
            auto& crtClusterMemoryOffsets = memoryOffsets[cluster];
            auto& crtClusterMemoryShapes = memoryShapes[cluster];

            auto crtClusterConsMemOffsets = consumerMemoryOffsets[cluster];
            auto crtClusterConsMemShapes = consumerMemoryShapes[cluster];
            if (opHasConsumerWithSETable) {
                std::tie(crtClusterConsMemOffsets, crtClusterConsMemShapes) = getActualDataMemOffsetsAndShapes(
                        sparseNceOpInputType, crtClusterConsMemOffsets, crtClusterConsMemShapes);
            }

            auto endOffset =
                    crtClusterMemoryOffsets[Dim(clusteringAxis)] + crtClusterMemoryShapes[Dim(clusteringAxis)] - 1;
            const auto candidateEndOffset =
                    crtClusterConsMemOffsets[Dim(clusteringAxis)] + crtClusterConsMemShapes[Dim(clusteringAxis)] - 1;

            auto startOffset = crtClusterMemoryOffsets[Dim(clusteringAxis)];
            const auto candidateStartOffset = crtClusterConsMemOffsets[Dim(clusteringAxis)];

            if (endOffset < candidateEndOffset) {
                endOffset = candidateEndOffset;
            } else if (opHasConsumerWithSETable && endOffset > candidateEndOffset) {
                return initDistribution;
            }

            if (startOffset > candidateStartOffset) {
                startOffset = candidateStartOffset;
            } else if (opHasConsumerWithSETable && startOffset < candidateStartOffset) {
                return initDistribution;
            }

            crtClusterMemoryOffsets[Dim(clusteringAxis)] = startOffset;
            crtClusterMemoryShapes[Dim(clusteringAxis)] = endOffset - startOffset + 1;

            // checking this in case the second operation is tiled. Because it is possible to have endOffset > shape.
            if (crtClusterMemoryOffsets[Dim(clusteringAxis)] + crtClusterMemoryShapes[Dim(clusteringAxis)] >
                shape[Dim(clusteringAxis)]) {
                return initDistribution;
            }
        }
    }

    return OverlapDistributionParams(arrayOfArrayFromShape(memoryShapes), arrayOfArrayFromShape(memoryOffsets),
                                     arrayOfArrayFromShape(computeShapes), arrayOfArrayFromShape(computeOffsets));
}

OverlapDistributionParams vpux::VPU::getActivationOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                                   ArrayRef<int64_t> activationTensorNumTiles,
                                                                   vpux::NDTypeInterface inType) {
    const auto kernelTileAxis = extractKernelTileAxis(activationTensorNumTiles);
    const auto localOverlappedParams =
            getOverlappedDistributionParameters(SmallVector<VPU::ClusteredOpInterface>({clusteredOp}), kernelTileAxis);

    // For 37XX, we do not set input workloads explicitly and therefore
    // OVERLAPPED should only represent the current op's input needs w/o
    // the sibling requirements
    auto archKind = config::getArch(clusteredOp.getOperation());
    if (archKind == config::ArchKind::NPU37XX) {
        return localOverlappedParams;
    }

    SmallVector<VPU::ClusteredOpInterface> siblingSubgraph;

    // TODO: use common interference graph util
    const size_t numberOperands = mlir::isa<VPU::NCEEltwiseOp>(clusteredOp.getOperation()) ? 2 : 1;

    // propagate operations which have the same input and output overlapped params
    const auto isBypassOp = [&](mlir::Operation* op) {
        if (mlir::isa_and_nonnull<VPU::QuantizeCastOp>(op)) {
            return true;
        }

        if (auto eltwiseOp = mlir::dyn_cast_or_null<VPU::NCEEltwiseOp>(op)) {
            if (eltwiseOp.getIsInplace().value_or(false)) {
                return true;
            }
        }

        return false;
    };

    const auto findSibling = [&](mlir::Value operand) -> void {
        for (const auto& sibling : operand.getUsers()) {
            SmallVector<mlir::Operation*> userOps;
            if (isBypassOp(sibling)) {
                userOps = to_small_vector(sibling->getUsers());
            } else {
                userOps.push_back(sibling);
            }

            for (const auto& user : userOps) {
                if (auto clusteredUser = mlir::dyn_cast<VPU::ClusteredOpInterface>(user)) {
                    siblingSubgraph.push_back(clusteredUser);
                }
            }
        }
    };

    // Given the following subgraph:
    //             Producer
    //         /             |
    // [QuantizeCast]  [QuantizeCast]
    //       |               |
    //     NceOp0          NceOp1
    // NceOp0 and NceOp1 should be aware of each other as siblings to be able to properly set their input distributions
    for (size_t opIdx = 0; opIdx < numberOperands; ++opIdx) {
        auto operand = clusteredOp->getOperand(opIdx);
        findSibling(operand);

        // TODO: 104112 avoid spilling due to other view ops besides of QuantizeCast
        if (auto quantizeCastOp = mlir::dyn_cast_or_null<VPU::QuantizeCastOp>(operand.getDefiningOp())) {
            operand = quantizeCastOp->getOperand(0);
            findSibling(operand);
        }
    }

    const auto candidateOverlappedParams = getOverlappedDistributionParameters(siblingSubgraph, kernelTileAxis);

    // Check candidateOverlappedParams if valid
    // For example,
    // Conv 1 with Input_Y = 8
    // Conv 1 with [[kernel_Y 1, stride_Y 1, pad_top_bottom 0, 0]]
    // Conv 2 with [[kernel_Y 8, stride_Y 8, pad_top_bottom 0, 0]]
    //
    // candidateOverlappedParams will follow Conv 2
    // Conv 1 will have Output_Y = 1 when inferring output shape
    // => will fail to split over numTiles (usually >=2)
    // So we need have this check
    const auto numTilesPerDim = (kernelTileAxis == Dims4D::Kernel::Y.ind())
                                        ? activationTensorNumTiles[Dims4D::Act::H.ind()]
                                        : activationTensorNumTiles[Dims4D::Act::W.ind()];
    const auto inputShape = getBoundedShape(inType);
    auto isOverlappedParamsValidForSplit = [&]() {
        const std::pair<int64_t, int64_t> inputHW = {inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W]};
        auto pads = candidateOverlappedParams.getPads();
        VPUX_THROW_WHEN(pads.has_value(), "Pads shouldn't be nullopt");
        const auto padInfo = PadInfo(pads->getLeftPad(), pads->getRightPad(), pads->getTopPad(), pads->getBottomPad());
        const auto getOutputHW = vpux::spatialOutputForInputWindowSize(inputHW, candidateOverlappedParams.getKernel(),
                                                                       candidateOverlappedParams.getStride(), padInfo);
        if (!getOutputHW.has_value()) {
            return false;
        }

        const auto outputHW = getOutputHW.value();
        return (kernelTileAxis == Dims4D::Kernel::Y.ind()) ? outputHW.first >= numTilesPerDim
                                                           : outputHW.second >= numTilesPerDim;
    };
    if (!isOverlappedParamsValidForSplit()) {
        return localOverlappedParams;
    }

    // Lacking a way specifying explicit per cluster shapes and offsets in the distributed
    // datatype, we are forced to pick a single sibling configuration.
    // Need to check if the picked configuration satisfies the current consumer
    // data requirements.
    //
    // For example two convolutions each with input height of 100 across 3 clusters
    // Conv 1 with [kernel_Y 1, stride_Y 1, pad_top_bottom 0, 0]
    // Conv 2 with [kernel_Y 5, stride_Y 4, pad_top_bottom 2, 2]
    //
    // Conv 1 needs [34, 33, 33] input size for each cluster
    // Conv 2 needs [35, 33, 31] input size for each cluster
    //
    // We can't pick a single sibling configuration to satisfy both operations.
    // Thus if the initial choice of config is not supporting, default to own config.
    //
    // This will change once we start using a more explicit representation for shapes per cluster
    // and no longer rely on the current distributed attribute config.

    const auto distributionMode = VPU::DistributionMode::OVERLAPPED;

    const auto candidateDistribution = VPU::DistributionInfo(
            distributionMode, activationTensorNumTiles, candidateOverlappedParams.getKernel(),
            candidateOverlappedParams.getStride(), candidateOverlappedParams.getPads(), numTilesPerDim, {},
            /*hasUniformDistributedSegments*/ true, {}, {}, {}, {}, {});

    const auto localDistributedAttr = VPU::DistributionInfo(
            distributionMode, activationTensorNumTiles, localOverlappedParams.getKernel(),
            localOverlappedParams.getStride(), localOverlappedParams.getPads(), numTilesPerDim, {},
            /*hasUniformDistributedSegments*/ true, {}, {}, {}, {}, {});

    const auto candidateOffsets = getPerClusterMemoryShapeOffsets(inputShape, candidateDistribution);
    const auto optionalCandidateShapes = getPerClusterMemoryShapes(inputShape, candidateDistribution);
    VPUX_THROW_UNLESS(optionalCandidateShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}", candidateDistribution);
    const auto& candidateShapes = optionalCandidateShapes.value();
    const auto localOffsets = getPerClusterMemoryShapeOffsets(inputShape, localDistributedAttr);
    const auto optionalLocalShapes = getPerClusterMemoryShapes(inputShape, localDistributedAttr);
    VPUX_THROW_UNLESS(optionalLocalShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}", localDistributedAttr);
    const auto& localShapes = optionalLocalShapes.value();

    for (auto offsetPerClusterZip : zip(candidateOffsets, localOffsets)) {
        for (auto dimZip : zip(std::get<0>(offsetPerClusterZip), std::get<1>(offsetPerClusterZip))) {
            if (std::get<0>(dimZip) > std::get<1>(dimZip)) {
                // candidate offset does not satisfy local op
                return localOverlappedParams;
            }
        }
    }

    const auto candidateEndOffset = getPerClusterEndOffset(candidateOffsets, candidateShapes);
    const auto localEndOffset = getPerClusterEndOffset(localOffsets, localShapes);

    for (auto endOffsetsPerClusterZip : zip(candidateEndOffset, localEndOffset)) {
        for (auto dimZip : zip(std::get<0>(endOffsetsPerClusterZip), std::get<1>(endOffsetsPerClusterZip))) {
            if (std::get<0>(dimZip) < std::get<1>(dimZip)) {
                // candidate shape does not satisfy local op
                return localOverlappedParams;
            }
        }
    }

    return candidateOverlappedParams;
}

std::set<VPU::ClusteredOpInterface> vpux::VPU::getSiblingOps(mlir::Operation* op) {
    if (!mlir::isa<VPU::ClusteredOpInterface>(op) && !isPassthroughOp(op)) {
        return {};
    }

    const bool isMultiInputOp = mlir::isa<VPU::NCEEltwiseOp, VPU::ConcatOp>(op);

    std::set<VPU::ClusteredOpInterface> siblingSubgraph = {};

    std::set<llvm::hash_code> visited = {};
    for (auto operand : op->getOperands()) {
        findSiblings(operand, siblingSubgraph, visited);

        if (!isMultiInputOp) {
            break;
        }
    }

    return siblingSubgraph;
}

bool vpux::VPU::outputOverlappedParamsIsHaloSupported(mlir::Operation* op) {
    return config::getArch(op) >= config::ArchKind::NPU40XX;
}

OverlapDistributionParams vpux::VPU::getActivationOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                                   ArrayRef<int64_t> activationTensorNumTiles,
                                                                   const bool uniformDistributedSegments,
                                                                   SiblingOpsAnalysis& siblingsAnalysis,
                                                                   vpux::NDTypeInterface inputType,
                                                                   const vpux::TileInfo& tileInfo) {
    // For 37XX, we do not set input workloads explicitly and therefore
    // OVERLAPPED should only represent the current op's input needs w/o
    // the sibling requirements
    // E#106872 to remove arch check
    if (!outputOverlappedParamsIsHaloSupported(clusteredOp.getOperation())) {
        const auto kernelTileAxis = extractKernelTileAxis(activationTensorNumTiles);
        return getOverlappedDistributionParameters(SmallVector<VPU::ClusteredOpInterface>({clusteredOp}),
                                                   kernelTileAxis);
    }

    // TODO: E#112803 Add support for extended memory view for sparse types with SETable
    auto sparseType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(clusteredOp->getOperand(0).getType());
    const bool isSparseTypeInputWithSeTable = (sparseType != nullptr) && (sparseType.getSeAttr() != nullptr);
    SmallVector<VPU::ClusteredOpInterface> siblingSubgraph{};
    if (isSparseTypeInputWithSeTable) {
        siblingSubgraph.push_back(clusteredOp);
    } else {
        auto siblings = siblingsAnalysis.getSiblings(clusteredOp);
        siblingSubgraph.append(siblings.begin(), siblings.end());
    }

    const auto clusteringAxis = VPU::getDistributedTilingAxis(activationTensorNumTiles);
    return getOverlappedDistributionParameters(
            (inputType != nullptr) ? inputType
                                   : mlir::cast<vpux::NDTypeInterface>(clusteredOp->getOperand(0).getType()),
            siblingSubgraph, activationTensorNumTiles[clusteringAxis], activationTensorNumTiles,
            uniformDistributedSegments, tileInfo);
}

OverlapDistributionParams vpux::VPU::getActivationOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                                   ArrayRef<int64_t> activationTensorNumTiles,
                                                                   const bool uniformDistributedSegments,
                                                                   vpux::NDTypeInterface inputType,
                                                                   const vpux::TileInfo& tileInfo) {
    // For 37XX, we do not set input workloads explicitly and therefore
    // OVERLAPPED should only represent the current op's input needs w/o
    // the sibling requirements
    // E#106872 to remove arch check
    if (!outputOverlappedParamsIsHaloSupported(clusteredOp.getOperation())) {
        const auto kernelTileAxis = extractKernelTileAxis(activationTensorNumTiles);
        return getOverlappedDistributionParameters(SmallVector<VPU::ClusteredOpInterface>({clusteredOp}),
                                                   kernelTileAxis);
    }

    // TODO: E#112803 Add support for extended memory view for sparse types with SETable
    auto sparseType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(clusteredOp->getOperand(0).getType());
    const bool isSparseTypeInputWithSeTable = (sparseType != nullptr) && (sparseType.getSeAttr() != nullptr);
    std::set<VPU::ClusteredOpInterface> siblingSubgraph = isSparseTypeInputWithSeTable
                                                                  ? std::set<VPU::ClusteredOpInterface>{clusteredOp}
                                                                  : getSiblingOps(clusteredOp.getOperation());

    const auto clusteringAxis = VPU::getDistributedTilingAxis(activationTensorNumTiles);
    return getOverlappedDistributionParameters(
            (inputType != nullptr) ? inputType
                                   : mlir::cast<vpux::NDTypeInterface>(clusteredOp->getOperand(0).getType()),
            SmallVector<VPU::ClusteredOpInterface>(siblingSubgraph.begin(), siblingSubgraph.end()),
            activationTensorNumTiles[clusteringAxis], activationTensorNumTiles, uniformDistributedSegments, tileInfo);
}

OverlapDistributionParams vpux::VPU::getOutputOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                               ArrayRef<int64_t> outputTensorNumTiles,
                                                               vpux::NDTypeInterface outputType,
                                                               ArrayRef<int64_t> activationTensorNumTiles) {
    SmallVector<VPU::ClusteredOpInterface> consumerSubgraph;
    auto archKind = config::getArch(clusteredOp.getOperation());
    const auto equalComputeAndMemoryView = archKind <= config::ArchKind::NPU37XX;

    if (auto eltwise = mlir::dyn_cast<VPU::NCEEltwiseOp>(clusteredOp.getOperation())) {
        if (eltwise.getIsInplace().value_or(false)) {
            // inplace eltwise should infer from input
            return getActivationOverlappedParams(clusteredOp, activationTensorNumTiles, outputType);
        }
    }

    for (const auto& result : clusteredOp->getResults()) {
        for (const auto& consumer : result.getUsers()) {
            if (auto clusteredConsumer = mlir::dyn_cast<VPU::ClusteredOpInterface>(consumer)) {
                consumerSubgraph.push_back(clusteredConsumer);
            }

            // Given the following subgraph:
            //     NCEProducer
            //         |
            //       Concat
            //      /    |
            // NceOp0  NceOp1
            // NCEProducer's effective consumers should be: {NceOp0, NceOp1}
            // NCEProducer should produce the largest halo possible to cover both NceOp0 & NceOp1 to allow the Concat
            // to be done implicitly in CMX.
            if (isValidCandidateForCMXConcat(consumer)) {
                for (const auto& consumerConcat : consumer->getUsers()) {
                    if (auto clusteredConsumerConcat = mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerConcat)) {
                        consumerSubgraph.push_back(clusteredConsumerConcat);
                    }
                }
            }

            // Given the following subgraph:
            //     NCEProducer
            //         |
            //     QuantizeCast
            //         |
            //       NceOp
            // NCEProducer's effective consumer should be NceOp
            // TODO: 104112 avoid spilling due to other view ops besides of QuantizeCast
            if (auto quantizeCastConsumer = mlir::dyn_cast_or_null<VPU::QuantizeCastOp>(consumer)) {
                for (const auto& consumerQuantizeCast : quantizeCastConsumer->getUsers()) {
                    if (auto clusteredConsumerQuantizeCast =
                                mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerQuantizeCast)) {
                        consumerSubgraph.push_back(clusteredConsumerQuantizeCast);
                    }
                }
            }

            // In-place Eltwise output overlapped params are inferred from input
            // propagate in-place Eltwise to consider consumers
            if (auto eltwiseConsumer = mlir::dyn_cast_or_null<VPU::NCEEltwiseOp>(consumer)) {
                if (eltwiseConsumer.getIsInplace().value_or(false)) {
                    for (const auto& consumerEltwise : eltwiseConsumer->getUsers()) {
                        if (auto clusteredConsumerEltwise =
                                    mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerEltwise)) {
                            consumerSubgraph.push_back(clusteredConsumerEltwise);
                        }
                    }
                }
            }
        }
    }
    const auto kernelTileAxis = extractKernelTileAxis(outputTensorNumTiles);
    const auto candidateOverlappedParams =
            getOverlappedDistributionParameters(consumerSubgraph, kernelTileAxis, equalComputeAndMemoryView);

    // Lacking a way specifying explicit per cluster shapes and offsets in the distributed
    // datatype, we are forced to pick a configuration where compute view is within the boundaries
    // of the memory view.
    // We represent input workload start & end through compute offset and size, while the total
    // amount of data in cluster is represented through memory shape. In cases where
    // compute start < memory start or compute end > memory_end, the size of data in cluster should be
    // max(compute end, memory_end) - min(compute start, memory start) + 1, but we currently have no way of
    // representing that. Therefore, we ensure that such a case will not happen by setting overlapped params k1x1,
    // s1x1, pad0x0x0x0 if the consumer distribution does not satisfy the requirements.

    const auto numTilesPerDim = (kernelTileAxis == Dims4D::Kernel::Y.ind())
                                        ? outputTensorNumTiles[Dims4D::Act::H.ind()]
                                        : outputTensorNumTiles[Dims4D::Act::W.ind()];
    const auto distributionMode = VPU::DistributionMode::OVERLAPPED;

    const auto candidateDistribution = VPU::DistributionInfo(
            distributionMode, outputTensorNumTiles, candidateOverlappedParams.getKernel(),
            candidateOverlappedParams.getStride(), candidateOverlappedParams.getPads(), numTilesPerDim, {},
            /*hasUniformDistributedSegments*/ true, {}, {}, {}, {}, {});

    const auto kernel = SmallVector<int64_t>{1, 1};
    const auto pads = VPU::Padding(0, 0, 0, 0);
    const auto strides = SmallVector<int64_t>{1, 1};
    const auto fallbackOverlappedParams = OverlapDistributionParams(kernel, pads, strides, equalComputeAndMemoryView);

    const auto outputShape =
            (outputType == nullptr) ? getBoundedShape(clusteredOp->getResult(0)) : getBoundedShape(outputType);
    const auto optionalCandidateMemoryShapes = getPerClusterMemoryShapes(outputShape, candidateDistribution);
    if (!optionalCandidateMemoryShapes.has_value()) {
        // If NCEProducer has tiling required and the tiled shape does not satisfy producer op
        return fallbackOverlappedParams;
    }

    const auto candidateMemoryOffsets = getPerClusterMemoryShapeOffsets(outputShape, candidateDistribution);
    const auto candidateComputeOffsets = getPerClusterComputeShapeOffsets(outputShape, candidateDistribution);
    const auto candidateComputeShapes = getPerClusterComputeShapes(outputShape, candidateDistribution);

    // Memory start offset must be before or equal to compute start offset
    for (auto startOffsetsPerClusterZip : zip(candidateMemoryOffsets, candidateComputeOffsets)) {
        for (auto dimZip : zip(std::get<0>(startOffsetsPerClusterZip), std::get<1>(startOffsetsPerClusterZip))) {
            if (std::get<0>(dimZip) > std::get<1>(dimZip)) {
                // candidate shape does not satisfy producer op
                return fallbackOverlappedParams;
            }
        }
    }

    const auto& candidateMemoryShapes = optionalCandidateMemoryShapes.value();
    const auto candidateMemoryEndOffset = getPerClusterEndOffset(candidateMemoryOffsets, candidateMemoryShapes);
    const auto candidateComputeEndOffset = getPerClusterEndOffset(candidateComputeOffsets, candidateComputeShapes);

    // Memory end offset must be after or equal to compute end offset
    for (auto endOffsetsPerClusterZip : zip(candidateMemoryEndOffset, candidateComputeEndOffset)) {
        for (auto dimZip : zip(std::get<0>(endOffsetsPerClusterZip), std::get<1>(endOffsetsPerClusterZip))) {
            if (std::get<0>(dimZip) < std::get<1>(dimZip)) {
                // candidate shape does not satisfy local op
                return fallbackOverlappedParams;
            }
        }
    }

    return candidateOverlappedParams;
}

OverlapDistributionParams vpux::VPU::getOutputOverlappedParamsNoHalo(VPU::ClusteredOpInterface clusteredOp,
                                                                     ArrayRef<int64_t> outputTensorNumTiles) {
    auto archKind = config::getArch(clusteredOp.getOperation());
    VPUX_THROW_WHEN(!mlir::isa<NCEPermuteOp>(clusteredOp.getOperation()),
                    "Arch {0} does not support output OVERLAPPED distribution for op = {1}", archKind, clusteredOp);

    const auto equalComputeAndMemoryView = true;
    const auto kernelTileAxis = extractKernelTileAxis(outputTensorNumTiles);

    SmallVector<VPU::ClusteredOpInterface> sohOverlappedConsumer{};
    for (const auto consumer : clusteredOp->getUsers()) {
        if (auto clusteredConsumer = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(consumer)) {
            auto multiclusterStrategy = clusteredConsumer.getMultiClusterStrategy();
            if (!multiclusterStrategy.has_value()) {
                continue;
            }

            if (multiclusterStrategy.value() == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
                sohOverlappedConsumer.push_back(clusteredConsumer);
            }
        }
    }

    return getOverlappedDistributionParameters(sohOverlappedConsumer, kernelTileAxis, equalComputeAndMemoryView);
}

OverlapDistributionParams vpux::VPU::getOutputOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                               ArrayRef<int64_t> outputTensorNumTiles,
                                                               const bool uniformDistributedSegments,
                                                               vpux::NDTypeInterface outputType,
                                                               const vpux::TileInfo& tileInfo,
                                                               SiblingOpsAnalysis& siblingsAnalysis) {
    // For arch w/o halo support
    // E#106872 to remove arch check
    if (!outputOverlappedParamsIsHaloSupported(clusteredOp.getOperation())) {
        return getOutputOverlappedParamsNoHalo(clusteredOp, outputTensorNumTiles);
    }

    auto consumers = siblingsAnalysis.getConsumers(clusteredOp);
    const auto clusteringAxis = VPU::getDistributedTilingAxis(outputTensorNumTiles);
    return getOverlappedDistributionParameters(
            (outputType != nullptr) ? outputType
                                    : mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType()),
            SmallVector<VPU::ClusteredOpInterface>(consumers.begin(), consumers.end()),
            outputTensorNumTiles[clusteringAxis], outputTensorNumTiles, uniformDistributedSegments, tileInfo);
}

OverlapDistributionParams vpux::VPU::getOutputOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                               ArrayRef<int64_t> outputTensorNumTiles,
                                                               const bool uniformDistributedSegments,
                                                               vpux::NDTypeInterface outputType,
                                                               const vpux::TileInfo& tileInfo) {
    // For arch w/o halo support
    // E#106872 to remove arch check
    if (!outputOverlappedParamsIsHaloSupported(clusteredOp.getOperation())) {
        return getOutputOverlappedParamsNoHalo(clusteredOp, outputTensorNumTiles);
    }

    auto siblingsAnalysis = SiblingOpsAnalysis(clusteredOp);
    return getOutputOverlappedParams(clusteredOp, outputTensorNumTiles, uniformDistributedSegments, outputType,
                                     tileInfo, siblingsAnalysis);
}

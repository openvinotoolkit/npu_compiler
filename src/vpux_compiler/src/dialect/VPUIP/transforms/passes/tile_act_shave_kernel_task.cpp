//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/bounded_buffer.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_TILEACTSHAVEKERNELTASK
#define GEN_PASS_DEF_TILEACTSHAVEKERNELTASK
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
using namespace VPUIP;

namespace {

/*
Experimental number to avoid performance drop for tiling ConvertOp. ConvertOp will have performance drop with input
element num less than the threshold. Need to replace it with cost model when the op is supported by VPUNN.
*/
constexpr size_t TILING_THRESHOLD_FOR_CONVERT = 8192;

Dim convertKernelAxisToDim(mlir::Value tensorArg, int64_t kernelAxis) {
    const auto inOrder = DimsOrder::fromValue(tensorArg);

    const auto shape = getShape(tensorArg);
    auto nDims = checked_cast<uint32_t>(shape.size());

    auto pos = nDims - 1 - kernelAxis;

    return inOrder.dimAt(pos);
}

bool isSoftmax(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    return kernelEntryName == "softmax" || kernelEntryName == "log_softmax";
}

bool isSoftmaxAxis(VPUIP::SwKernelOp swKernelOp, Dim axis) {
    if (!isSoftmax(swKernelOp)) {
        return false;
    }

    auto taskArgs = kernelArgsRange(swKernelOp);
    const auto kernelAxis = taskArgs[0].dyn_cast<mlir::IntegerAttr>().getInt();

    auto softmaxAxis = convertKernelAxisToDim(swKernelOp.getResult(0), kernelAxis);

    if (softmaxAxis == axis) {
        return true;
    }

    return false;
}

bool isTopKAxis(VPUIP::SwKernelOp swKernelOp, Dim axis) {
    auto taskArgs = kernelArgsRange(swKernelOp);
    const auto kernelAxis = taskArgs.front().cast<mlir::IntegerAttr>().getInt();
    auto topKAxis = convertKernelAxisToDim(swKernelOp.getResult(0), kernelAxis);

    return topKAxis == axis;
}

bool isNormalizeL2Axis(VPUIP::SwKernelOp swKernelOp, Dim axis) {
    auto taskArgs = kernelArgsRange(swKernelOp);
    auto numOfAxis = taskArgs[2].cast<mlir::IntegerAttr>().getInt();
    const auto kernelAxises = parseIntArrayAttr<int64_t>(taskArgs[3].cast<mlir::ArrayAttr>());
    return std::find(kernelAxises.begin(), kernelAxises.begin() + numOfAxis, axis.ind()) != kernelAxises.end();
}

Dim getHighestDimOfSoftmax(VPUIP::SwKernelOp swKernelOp) {
    const auto output = swKernelOp->getResult(0);
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();
    const auto outOrder = outputType.getDimsOrder();
    const auto outShape = outputType.getShape();

    auto taskArgs = kernelArgsRange(swKernelOp);
    const auto kernelAxis = taskArgs.front().cast<mlir::IntegerAttr>().getInt();
    auto softmaxAxis = convertKernelAxisToDim(swKernelOp.getResult(0), kernelAxis);

    for (auto i : irange(outOrder.numDims())) {
        auto dim = outOrder.dimAt(i);
        if (outShape[dim] > 1 && dim != softmaxAxis) {
            return dim;
        }
    }
    return outOrder.dimAt(0);
}

std::optional<Dim> getHighestTileableDimOfMvn6(VPUIP::SwKernelOp swKernelOp) {
    const auto output = swKernelOp->getResult(0);
    const auto type = output.getType().cast<vpux::NDTypeInterface>();
    const auto order = type.getDimsOrder();
    const auto shape = type.getShape();

    auto args = kernelArgsRange(swKernelOp);
    const auto axesAttr = args.begin()[5].dyn_cast<mlir::ArrayAttr>();
    VPUX_THROW_UNLESS(axesAttr != nullptr, "Failed to extract axes at '{0}'", swKernelOp->getLoc());
    const auto axes = parseIntArrayAttr<int64_t>(axesAttr);

    vpux::DimArr axesDims;
    for (size_t i = 0; i < axes.size(); i++) {
        axesDims.push_back(convertKernelAxisToDim(swKernelOp.getResult(0), axes[i]));
    }

    for (auto i : irange(order.numDims())) {
        auto dim = order.dimAt(i);
        auto isNormAxis = std::find(axesDims.begin(), axesDims.end(), dim) != axesDims.end();
        if (shape[dim] > 1 && !isNormAxis) {
            return dim;
        }
    }

    return std::nullopt;
}

std::optional<Dim> getHighestTileableDimOfMvn1sum(ShapeRef shape, const DimsOrder& dimOrder) {
    for (auto idx : irange(dimOrder.numDims())) {
        auto curDim = dimOrder.dimAt(idx);
        if (shape[curDim] != 1 && curDim != Dims4D::Act::W) {
            return curDim;
        }
    }

    return std::nullopt;
}

bool hasNon4DOutputShape(VPUIP::SwKernelOp swKernelOp) {
    // Checking for non 4d output, in such cases tiling is not possible except for GatherOp
    if (std::any_of(swKernelOp.getOutputs().begin(), swKernelOp.getOutputs().end(), [](const auto& output) {
            return mlir::cast<vpux::NDTypeInterface>(output.getType()).getRank() != 4;
        })) {
        return true;
    }
    return false;
}

bool hasOnlyOneOffset(VPUIP::SwKernelOp swKernelOp, Dim tileDim) {
    // for the case: input shape is [1,4,83,5], NCHW layer, multicluster on H, sw kernel tile on C
    // two multi cluster tile is [1,4,42,5], [1,4,41,5], the offset for second shave is different
    // one is 2*42*5, another one is 2*41*5, but currently we can only use one offset.
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (clusterTilingOp == nullptr) {
        return true;
    }
    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported type {0}", distributedType);
    auto order = distributedType.getDimsOrder();
    auto dimIdx = VPUIP::getTilingDimIndex(distributedType);
    if (dimIdx.has_value() && order.dimPos(Dim(dimIdx.value())) > order.dimPos(tileDim)) {
        auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        for (auto shape : perClusterShapes) {
            if (shape[Dim(dimIdx.value())] != perClusterShapes.front()[Dim(dimIdx.value())]) {
                return false;
            }
        }
    }
    return true;
}

bool isSegmentedOnDimC(VPUIP::SwKernelOp swKernelOp) {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (clusterTilingOp == nullptr) {
        return false;
    }

    auto outDistributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (outDistributedType == nullptr) {
        return false;
    }

    return VPU::isSegmentedOverC(outDistributedType.getDistribution());
}

Dim getSwKernelTileDim(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "mvn1") {
        // MVN only supports tiling on C
        return Dims4D::Act::C;
    } else if (kernelEntryName == "mvn1_sum") {
        auto outType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getResult(0).getType());
        return getHighestTileableDimOfMvn1sum(outType.getShape(), outType.getDimsOrder()).value_or(Dim(0));
    } else if (kernelEntryName == "mvn6") {
        // MVN6 only supports tiling on non-normalization axes
        auto dim = getHighestTileableDimOfMvn6(swKernelOp);
        VPUX_THROW_UNLESS(dim.has_value(), "Expecting '{0}' at '{1}' to have a tileable axis", swKernelOp->getName(),
                          swKernelOp->getLoc());
        return dim.value();
    } else if (kernelEntryName == "interpolate") {
        return Dims4D::Act::H;
    } else if (kernelEntryName == "softmax" || kernelEntryName == "log_softmax") {
        // Hightest Dim may lead to different offset that cause insert copy.
        auto tileDim = getHighestDimOfSoftmax(swKernelOp);
        if (hasOnlyOneOffset(swKernelOp, tileDim)) {
            return tileDim;
        }
    } else if (kernelEntryName == "gru_sequence") {
        return Dims4D::Act::N;
    } else if (kernelEntryName == "gru_sequence_last_part") {
        return Dims4D::Act::N;
    } else if (kernelEntryName == "lstm_gates") {
        return Dims4D::Act::H;
    } else if (kernelEntryName == "lstm_cell") {
        const auto tileDim = (swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>()).getShape().size() - 1;
        return Dim(tileDim);
    } else if (kernelEntryName == "lstm_sequence") {
        const auto tileDim = (swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>()).getShape().size() - 1;
        return Dim(tileDim);
    }

    auto isHighestDimTilingPerformant = [&]() {
        // original case
        if (isSegmentedOnDimC(swKernelOp)) {
            return true;
        }

        // activation SW ops assumed to follow DPU ops, avoid spilling due tiling on
        // axis which requires stride, prefer highest dim
        if (isActivationSwKernelOp(swKernelOp)) {
            return true;
        }

        // other SW ops can have worse performance due to tiling dim size
        // TODO: heuristic based on DMA cost + SW cost
        // ticket E#117136
        return false;
    };

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getResult(0).getType());
    const auto tileDim = getHighestNonTrivialDim(outputType.getShape(), outputType.getDimsOrder()).value_or(Dim(0));

    // for supported ops try to avoid DMAs by expressing tiling with offsets
    if (isHighestDimTilingPerformant() && hasOnlyOneOffset(swKernelOp, tileDim)) {
        return tileDim;
    }

    // align tiling dim with the distributed buffer
    if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp())) {
        const auto distOutType = clusterTilingOp.getResult(0).getType();
        auto dimIdx = VPUIP::getTilingDimIndex(distOutType);
        if (dimIdx.has_value()) {
            return Dim(dimIdx.value());
        }
    }

    return tileDim;
}

bool isGatherOpTileAtHighestDim(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName != "gather") {
        return false;
    }

    const auto inputType = swKernelOp->getOperand(0).getType().cast<NDTypeInterface>();
    const auto indicesType = swKernelOp->getOperand(1).getType().cast<NDTypeInterface>();

    auto args = kernelArgsRange(swKernelOp);
    const auto kernelAxisAttr = args.begin()[0].dyn_cast<mlir::IntegerAttr>();
    VPUX_THROW_UNLESS(kernelAxisAttr != nullptr, "Failed to extract axis at '{0}'", swKernelOp->getLoc());
    int64_t kernelAxis = kernelAxisAttr.getValue().getSExtValue();
    // Convert the axis from kernel to compiler representation
    int64_t axisVal = inputType.getRank() - 1 - kernelAxis;

    const auto batchDimsAttr = args.begin()[1].dyn_cast<mlir::IntegerAttr>();
    VPUX_THROW_UNLESS(batchDimsAttr != nullptr, "Failed to extract batch_dim at '{0}'", swKernelOp->getLoc());
    int64_t batchDimsVal = batchDimsAttr.getValue().getSExtValue();

    const int64_t inHighestDimVal =
            getHighestNonTrivialDim(inputType.getShape(), inputType.getDimsOrder()).value_or(Dim(0)).ind();
    const int64_t indicesHighestDimVal =
            getHighestNonTrivialDim(indicesType.getShape(), indicesType.getDimsOrder()).value_or(Dim(0)).ind();
    int64_t outTileDimVal = getSwKernelTileDim(swKernelOp).ind();

    // outTileDim is before axisDim
    // Input should be tiled at highestDim, Indices do not require tiling
    auto inTileDimVal = outTileDimVal;
    if (outTileDimVal < axisVal) {
        return inTileDimVal == inHighestDimVal;
    }

    // outTileDim within axisDim
    // Input do not require tiling, Indices should tile at highestDim
    auto indicesTileDimVal = outTileDimVal - axisVal + batchDimsVal;
    if (outTileDimVal >= axisVal && outTileDimVal <= axisVal + indicesType.getRank() - batchDimsVal) {
        return indicesTileDimVal == indicesHighestDimVal;
    }

    // outTileDim is after axisDim
    // Input should be tiled at highestDim, Indices do not require tiling
    inTileDimVal = outTileDimVal - indicesType.getRank() + batchDimsVal;
    return inTileDimVal == inHighestDimVal;
}

bool canGatherElementsOpTileAtHighestDim(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName != "gather_elements") {
        return false;
    }

    const auto inputType = swKernelOp->getOperand(0).getType().cast<NDTypeInterface>();

    auto args = kernelArgsRange(swKernelOp);
    const auto kernelAxisAttr = args.begin()[0].dyn_cast<mlir::IntegerAttr>();
    VPUX_THROW_UNLESS(kernelAxisAttr != nullptr, "Failed to extract axis at '{0}'", swKernelOp->getLoc());
    int64_t kernelAxis = kernelAxisAttr.getValue().getSExtValue();
    // Convert the axis from kernel to compiler representation
    int64_t axisVal = inputType.getRank() - 1 - kernelAxis;

    const int64_t inHighestDimVal =
            getHighestNonTrivialDim(inputType.getShape(), inputType.getDimsOrder()).value_or(Dim(0)).ind();
    return axisVal != inHighestDimVal;
}

bool isOpTileOverWidthDim(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    VPUX_THROW_UNLESS(kernelEntryName == "rms_norm" || kernelEntryName == "rope",
                      "This function was designed for RMSNorm or RoPE operator");

    const auto outTileDimVal = getSwKernelTileDim(swKernelOp);
    if (outTileDimVal == Dims4D::Act::W) {
        return true;
    }
    return false;
}

bool isDynamicTilingSupported(StringRef kernelEntryName) {
    // List of kernel names that support dynamic tiling
    static const std::unordered_set<std::string> supportedKernels = {
            "lstm_sequence",
            // Add more kernel names as they become supported
    };

    return supportedKernels.find(kernelEntryName.str()) != supportedKernels.end();
}

bool doesSwKernelSupportTiling(VPUIP::SwKernelOp swKernelOp, vpux::Logger log) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);

    if (VPUIP::hasBoundedBuffers(swKernelOp)) {
        return isDynamicTilingSupported(kernelEntryName);
    }

    const auto arch = VPU::getArch(swKernelOp);
    // this is a workaround to force tiling of an operation with multiple outputs
    if ((kernelEntryName == "detection_output_sort") && (arch == VPU::ArchKind::NPU37XX)) {
        auto module = swKernelOp.getOperation()->getParentOfType<mlir::ModuleOp>();
        auto tileOp = vpux::IE::getTileExecutor(module);
        VPUX_THROW_UNLESS(tileOp != nullptr, "Expected tileOp executor in order to query SHAVE_ACT executor.");
        VPUX_THROW_UNLESS(tileOp.hasSubExecutor(VPU::ExecutorKind::SHAVE_ACT),
                          "No SHAVE_ACT executor found, check your arch");
        auto actShavePerTile = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT);

        return getShape(swKernelOp->getResult(0))[Dims4D::Act::H] >= actShavePerTile.getCount();
    }

    auto isAllOutputShapeEqual = llvm::all_of(swKernelOp.getOutputs(), [&](auto output) {
        return getShape(output) == getShape(*swKernelOp.getOutputs().begin());
    });

    // GRUSequenceOp/GRUSequenceLastPartOp has two different output shapes.
    if ((kernelEntryName != "gru_sequence") && (kernelEntryName != "gru_sequence_last_part") &&
        (kernelEntryName != "lstm_sequence") && (swKernelOp.getOutputs().size() > 2 || !isAllOutputShapeEqual)) {
        log.trace("SW kernel op has outputs with different shapes at '{0}'", swKernelOp->getLoc());
        return false;
    }

    if (!isSwKernelTilingSupported(swKernelOp)) {
        return false;
    }

    if (hasNon4DOutputShape(swKernelOp) && kernelEntryName != "gather" && kernelEntryName != "gru_sequence" &&
        kernelEntryName != "gru_sequence_last_part") {
        // GatherOp/GRUSequenceOp/GRUSequenceLastPartOp supports non4D input output shapes with tiling.
        log.trace("SW kernel '{0}' op has non-4d output at '{1}'", kernelEntryName, swKernelOp->getLoc());
        return false;
    }

    if (kernelEntryName == "mvn1") {
        auto taskArgs = kernelArgsRange(swKernelOp);
        const auto acrossChannels = taskArgs[0].dyn_cast<mlir::BoolAttr>();
        return !acrossChannels.getValue();
    } else if (kernelEntryName == "mvn1_sum") {
        auto outType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getResult(0).getType());
        auto outHighestDim =
                getHighestTileableDimOfMvn1sum(outType.getShape(), outType.getDimsOrder()).value_or(Dim(0));
        if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp())) {
            if (auto dimIdx = VPUIP::getTilingDimIndex(clusterTilingOp.getOperand(0).getType())) {
                return Dim(dimIdx.value()) == outHighestDim;
            }
        }
        return true;
    } else if (kernelEntryName == "mvn6") {
        auto dim = getHighestTileableDimOfMvn6(swKernelOp);
        return dim.has_value();
    } else if (kernelEntryName == "softmax" || kernelEntryName == "log_softmax") {
        auto highestDim = getHighestDimOfSoftmax(swKernelOp);
        if (isSoftmaxAxis(swKernelOp, highestDim)) {
            return false;
        }
    } else if (kernelEntryName == "convert") {
        auto shape = getShape(swKernelOp.getInputs()[0]);
        if (checked_cast<size_t>(shape.totalSize()) < TILING_THRESHOLD_FOR_CONVERT) {
            return false;
        }

        // E#83794 Case with aligned tiling not supported
        // Offsets for the inputs and outputs needs to be adjusted based on aligned shapes
        if (auto clusterTilingOp = swKernelOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
            auto ndType = clusterTilingOp->getOperand(0).getType().cast<VPUIP::DistributedBufferType>();
            if (ndType != nullptr && ndType.getDistribution().getAlignment() != nullptr) {
                return false;
            }
        }
    } else if (kernelEntryName == "topk") {
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getResult(0).getType());
        auto highestDim = getHighestNonTrivialDim(outputType.getShape(), outputType.getDimsOrder()).value_or(Dim(0));
        if (isTopKAxis(swKernelOp, highestDim)) {
            return false;
        }
    } else if (kernelEntryName == "normalize_l2") {
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getResult(0).getType());
        auto highestDim = getHighestNonTrivialDim(outputType.getShape(), outputType.getDimsOrder()).value_or(Dim(0));
        if (isNormalizeL2Axis(swKernelOp, highestDim)) {
            return false;
        }
    } else if (kernelEntryName == "gather") {
        // Gather kernel not support stride input, if enable multi-shave will produce stride input for gather,
        // additional stride DMAs will be introduced to ensure that the input of the gather is continuous.
        if (!isGatherOpTileAtHighestDim(swKernelOp)) {
            return false;
        }
    } else if (kernelEntryName == "gather_elements") {
        // GatherElements kernel not support stride input, if enable multi-shave will produce stride input for gather,
        // additional stride DMAs will be introduced to ensure that the input of the gather is continuous.
        if (!canGatherElementsOpTileAtHighestDim(swKernelOp)) {
            return false;
        }

    } else if (kernelEntryName == "activation_sigmoid") {
        // E#92211: Measurements for the performance profiling, see this ticket for details.
        const auto inputSize = getTotalSize(swKernelOp.getInputs()[0]);
        if (inputSize < VPUIP::SIGMOID_SW_KERNEL_TILING_THRESHOLD) {
            log.trace("Sigmoid has {0} bytes of total size which is not efficient for multi shave", inputSize);
            return false;
        }
    } else if (kernelEntryName == "depth_to_space") {
        // Do not tile DepthToSpace SW kernel in case when it's legal and beneficial to use DMA
        return !isLegalAndBeneficialConvertToDMA(swKernelOp, log);
    } else if (kernelEntryName == "gru_sequence") {
        const auto outputType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
        const auto outputShape = outputType.getShape().raw();
        const auto batchSize = outputShape[0];
        if (batchSize == 1) {
            return false;
        }
    } else if (kernelEntryName == "gru_sequence_last_part") {
        const auto outputType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
        const auto outputShape = outputType.getShape().raw();
        const auto batchSize = outputShape[0];
        if (batchSize == 1) {
            return false;
        }
    } else if (kernelEntryName == "lstm_gates") {
        // #E124098: Statistic for the LSTMGates multi-Shaves performance.
        const auto inputSize = getTotalSize(swKernelOp.getInputs()[0]) + getTotalSize(swKernelOp.getInputs()[1]);
        const auto minimalSize = Byte(1280);
        if (inputSize < minimalSize) {
            log.trace("lstm_gates total size is {0} bytes which is not efficient for multi shave", inputSize);
            return false;
        }
    } else if (kernelEntryName == "eltwise_mul" || kernelEntryName == "eltwise_select") {
        const auto outputSize = getTotalSize(swKernelOp.getOutputs()[0]);
        const auto minimalSize = Byte(4096);
        if (outputSize < minimalSize) {
            log.trace("Eltwise operation has {0} bytes of total size which is not efficient for multi shave",
                      outputSize);
            return false;
        }
    } else if (kernelEntryName == "rms_norm" || kernelEntryName == "rope") {
        // RMSNorm is not supporting over width tiling
        if (isOpTileOverWidthDim(swKernelOp)) {
            return false;
        }
    } else if (kernelEntryName == "dynamic_dequantize") {
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getResult(0).getType());
        auto highestDim = getHighestNonTrivialDim(outputType.getShape(), outputType.getDimsOrder()).value_or(Dim(0));
        // Tile on W dim may lead to use slow C algo
        if (highestDim == Dims4D::Act::W) {
            return false;
        }
    }

    return true;
}

mlir::FailureOr<OutputTiling> getSwKernelOutputTiling(VPUIP::SwKernelOp swKernelOp, ShapeRef outputShape,
                                                      int64_t maxNumTiles, bool insertSubview, vpux::Logger log) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);

    if (kernelEntryName == "lstm_sequence") {
        OutputTiling dividedTiles;
        TileInfo tileFullOutput(outputShape);
        log.trace("lstm_sequence no tiles is {0}", maxNumTiles);
        if (maxNumTiles > 2) {  // lstm_sequence not support
            log.trace("lstm_sequence not support more that 2 multi-shave implementation maxTile is {0}", maxNumTiles);
            return mlir::failure();
        }
        auto repetLoop = maxNumTiles;
        while (repetLoop--) {
            dividedTiles.push_back(tileFullOutput);
        }
        return dividedTiles;
    }

    // Gather op's output always is non-4D and Gather's backInfer has it's own logic later, skip the check here.
    if (kernelEntryName != "gather") {
        VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported operation '{0}' at '{1}', it has non 4D result",
                          swKernelOp->getName(), swKernelOp->getLoc());
    }

    Shape nTilesOnDim(outputShape.size(), 1);
    const auto tileDim = getSwKernelTileDim(swKernelOp);
    log.trace("Tile Dim is {0}", tileDim);
    nTilesOnDim[tileDim] = std::min(maxNumTiles, outputShape[tileDim]);
    std::optional<ArrayRef<int64_t>> optionalAlignment = std::nullopt;
    // Declare the neutral value outside of the condition.
    // Otherwise alignment vector gets destroyed at the end.
    // optionalAlignment is left with a dangling reference.
    SmallVector<int64_t> alignment(outputShape.size(), 1);
    if (kernelEntryName == "depth_to_space") {
        // Tile DepthToSpace layer with alignment to ensure the tiled output width or height is aligned to block size
        // For example, a DepthToSpace op in below
        // input shape:     [1, 128, 15, 270]
        // output shape:    [1, 8, 60, 1080]
        // block size:      4
        // By default, tiling without alignment would generate 2 tiles have the same output shape [1, 8, 30, 1080]
        // Height value (30) is not aligned to block size (4) after tiling, which is invalid for DepthToSpace layer
        // The valid tiled output shape should be [1, 8, 32, 270] and [1, 8, 28, 270]
        auto taskArgs = kernelArgsRange(swKernelOp);
        VPUX_THROW_WHEN(taskArgs.empty(), "Not kernel args in SwKernelRun {0}", swKernelOp->getLoc());
        const auto blockSize = taskArgs.front().cast<mlir::IntegerAttr>().getValue().getSExtValue();
        VPUX_THROW_WHEN(blockSize == 0, "BlockSize is zero and used as a divisor");

        alignment[tileDim.ind()] = blockSize;
        optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
    } else if (insertSubview) {
        // Shave can gain better performance when data address is 32 bytes aligned, the begin offset on the first shave
        // is already guaranteed with this condition. And for the other shaves, we need to adjust the tiled shape to
        // guarantee it.
        auto dimOrder = DimsOrder::fromValue(swKernelOp->getResult(0));
        auto memShape = dimOrder.toMemoryOrder(outputShape);
        auto memDim = dimOrder.toMemDim(tileDim);
        int64_t strideOnTilingDim = 1;
        for (auto i : irange(memShape.size())) {
            if (i > static_cast<size_t>(memDim.ind())) {
                strideOnTilingDim *= memShape[MemDim(i)];
            }
        }
        const auto arch = VPU::getArch(swKernelOp);
        const auto addrAlign = VPUIP::getSwKernelTilingAddressAlignment(swKernelOp, arch);
        const auto elemSize = swKernelOp.getOutputs().front().getType().cast<vpux::NDTypeInterface>().getElemTypeSize();
        auto alignmentVal =
                std::lcm(strideOnTilingDim, Byte(addrAlign).to<Bit>().count() / elemSize.count()) / strideOnTilingDim;
        if (alignmentVal < outputShape[tileDim]) {
            alignment[tileDim.ind()] = alignmentVal;
            optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
        }
    }
    return fillDividedTiles(nTilesOnDim, outputShape, optionalAlignment);
}

SmallVector<mlir::Value> getOuterMappingOperand(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange innerOperands) {
    auto clusterTilingOp = swKernelOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    auto isClusterTilingApplied = clusterTilingOp != nullptr;
    SmallVector<mlir::Value> outerOperands;
    for (auto operand : innerOperands) {
        if (!isClusterTilingApplied) {
            outerOperands.push_back(operand);
        } else {
            auto blockArg = operand.dyn_cast<mlir::BlockArgument>();
            VPUX_THROW_WHEN(blockArg == nullptr, "Matching argument was not identified");
            auto outerOperand = clusterTilingOp->getOperand(blockArg.getArgNumber());
            outerOperands.push_back(outerOperand);
        }
    }
    return outerOperands;
}

VPUIP::NCEClusterTilingOp createNewTilingCopyOp(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Type outType,
                                                ArrayRef<mlir::Value> operands) {
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
    };
    return rewriter.create<VPUIP::NCEClusterTilingOp>(loc, outType, operands, bodyBuilder);
}

VPUIP::SubViewOp createSubViewOpWithDistributedOutput(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                                      vpux::NDTypeInterface outType, mlir::Value operand,
                                                      ShapeRef offset) {
    auto distributedType = outType.cast<VPUIP::DistributedBufferType>();
    auto distribution = distributedType.getDistribution();
    auto mode = distribution.getMode().getValue();
    auto ctx = rewriter.getContext();
    auto outShape = to_small_vector(outType.getShape());
    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distribution) && mode != VPU::DistributionMode::DUPLICATED) {
        return rewriter.create<VPUIP::SubViewOp>(loc, operand, vpux::getIntArrayAttr(ctx, offset),
                                                 vpux::getIntArrayAttr(ctx, outShape), nullptr,
                                                 distribution.getComputeShapes());
    }
    return rewriter.create<VPUIP::SubViewOp>(loc, operand, vpux::getIntArrayAttr(ctx, offset),
                                             vpux::getIntArrayAttr(ctx, outShape));
}

bool checkSwKernelTilingAlignment(VPUIP::SwKernelOp swKernelOp, const vpux::NDTypeInterface valueType,
                                  VPUIP::DistributedBufferType distBufferType, vpux::Logger log) {
    auto clusterOp = swKernelOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (clusterOp == nullptr) {
        return true;
    }

    // todo: enable unaligned shave on NPU37XX too
    // ticket E#114487
    if (!isArchVPUX3XXX(VPU::getArch(swKernelOp))) {
        return true;
    }

    auto distribution = distBufferType.getDistribution();
    auto alignAttr = distribution.getAlignment();
    if (alignAttr == nullptr) {
        return true;
    }

    const auto alignmentPerTile = parseIntArrayAttr<int64_t>(alignAttr);
    const auto tileDim = getSwKernelTileDim(swKernelOp);
    if (alignmentPerTile[tileDim.ind()] == 1) {
        return true;
    }

    auto moduleOp = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto tileExec = IE::getTileExecutor(moduleOp);
    auto shaveActExec = tileExec.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT);
    auto numSplits = shaveActExec.getCount();

    if (distribution.getNumTiles() != nullptr) {
        const auto numTiles = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
        numSplits *= std::accumulate(numTiles.begin(), numTiles.end(), (int64_t)1, std::multiplies<int64_t>());
    }

    const auto valueShape = valueType.getShape();
    const auto totalAlignment = alignmentPerTile[tileDim.ind()] * numSplits;
    if (valueShape[tileDim] % totalAlignment) {
        log.trace("Skip tiling for swKernelOp {0}, shape is not aligned. Shape '{1}', distribution '{2}', alignment "
                  "'{3}'",
                  swKernelOp->getLoc(), valueShape, distribution, totalAlignment);
        return false;
    }

    return true;
}

//
// SwKernelRewriterBase
//

static OutputTiling computeOutputTiling(const SmallString& kernelEntryName, const TileInfo& firstOutputTile) {
    if (kernelEntryName == "detection_output_sort") {
        return vpux::VPU::DetectionOutputSortOpOutputTiling(firstOutputTile);
    } else if (kernelEntryName == "topk") {
        return {firstOutputTile, firstOutputTile};
    } else if (kernelEntryName == "gru_sequence" || kernelEntryName == "gru_sequence_last_part") {
        return vpux::VPU::GRUSequenceOutputTiling(firstOutputTile);
    } else if ((kernelEntryName == "lstm_gates") || (kernelEntryName == "lstm_cell")) {
        return {firstOutputTile, firstOutputTile};
    } else if (kernelEntryName == "lstm_sequence") {
        return vpux::VPU::lstmSequenceOutputTiling(firstOutputTile);
    }
    return OutputTiling{firstOutputTile};
}

class SwKernelRewriterBase : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    SwKernelRewriterBase(mlir::MLIRContext* ctx, int64_t shaveCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _shaveCount(shaveCount), _log(log) {
        setDebugName("SwKernelRewriterBase");
    }
    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const override;
    virtual bool checkTilePattern(VPUIP::SwKernelOp swKernelOp, bool insertSubview) const = 0;
    virtual bool needInsertSubviewOnly(VPUIP::SwKernelOp swKernelOp) const;
    virtual std::optional<OutputTiling> calculateOutputTiles(VPUIP::SwKernelOp swKernelOp) const = 0;
    virtual std::optional<SmallVector<InputTiling>> calculateInputTiles(VPUIP::SwKernelOp swKernelOp) const = 0;
    virtual size_t getShaveTileSize(VPUIP::SwKernelOp swKernelOp, const OutputTiling& outTiles) const = 0;
    virtual SmallVector<mlir::Value> createNewInputs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                                     bool insertSubview, int64_t outTileIndex,
                                                     mlir::PatternRewriter& rewriter) const = 0;
    virtual SmallVector<mlir::Value> createNewOutBuffs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                                       bool insertSubview, int64_t outTileIndex,
                                                       mlir::PatternRewriter& rewriter) const = 0;
    virtual VPUIP::SwKernelOp createNewSwKernelOp(VPUIP::SwKernelOp swKernelOp, ArrayRef<mlir::Value> newInputs,
                                                  ArrayRef<mlir::Value> newOutBufs, bool insertSubview,
                                                  mlir::PatternRewriter& rewriter) const = 0;
    virtual mlir::FailureOr<VPUIP::ShapeCastOp> getSWKernelWithFusedDims(VPUIP::SwKernelOp swKernelOp,
                                                                         mlir::PatternRewriter& rewriter) const = 0;
    virtual mlir::FailureOr<VPUIP::PermuteCastOp> adjustSWLayout(VPUIP::SwKernelOp swKernelOp,
                                                                 mlir::PatternRewriter& rewriter) const = 0;
    virtual void replaceOpWithConcatView(VPUIP::SwKernelOp origOp, VPUIP::SwKernelOp newSwkernelOp, bool insertSubview,
                                         mlir::PatternRewriter& rewriter) const = 0;
    virtual OutputTiling getOuterMostOutputTiling(VPUIP::SwKernelOp swKernelOp) const = 0;
    virtual InputTiling getOuterMostInputTiling(VPUIP::SwKernelOp swKernelOp, int64_t outTileIndx) const = 0;
    virtual SmallVector<mlir::Attribute> updateSwKernelAttrs(VPUIP::SwKernelOp swKernelOp,
                                                             int64_t outTileIndexInsideCluster) const;
    virtual bool requireBalancingShapeCast(VPUIP::SwKernelOp swKernelOp) const = 0;
    virtual bool requireLayoutChangePermuteCast(VPUIP::SwKernelOp swKernelOp) const = 0;

protected:
    int64_t _shaveCount;
    Logger _log;
};

/*
 Tile SwKernel within a cluster. Note that copy op is inserted to provide continuous buffer for each tile of SwKernel

     |          |                      |
Copy(DDR2CMX) Alloc               /            \
     \        /             SubView          Alloc
      SwKernel                   |              |
    (SwKernelRun)    =>     Copy(DDR2CMX)       |
         |                       \             /
    Copy(CMX2DDR)            SwKernel(Multi SwKerneRun)
                                      |
                                    Concat
*/
mlir::LogicalResult SwKernelRewriterBase::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), getSwKernelEntryName(swKernelOp), swKernelOp->getLoc());
    auto swKernelRun = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    if (std::distance(swKernelRun.begin(), swKernelRun.end()) > 1) {
        // swKernelOp has already been tiled
        return mlir::failure();
    }
    if (!doesSwKernelSupportTiling(swKernelOp, _log)) {
        // swKernelOp doesn't support tiling on mulit shaves
        return mlir::failure();
    }

    // If a SW Op doesn't support input stride access and the tile dimension isn't at the highest dimension
    // CopyOp is inserted to maintain a continuous input buffer
    // Eltwise SW Op are layout agnostic. Inserting a PermuteCastOp to move the tile dimension to the highest level
    // can optimize performance by inserting only SubviewOp
    if (requireLayoutChangePermuteCast(swKernelOp)) {
        // Replace SW with PermuteCast-SW-PermuteCast
        auto permuteResult = adjustSWLayout(swKernelOp, rewriter);
        if (mlir::failed(permuteResult)) {
            _log.trace("Adjust layout to insert subview failed");
        } else {
            auto permuteCastOp = permuteResult.value();
            auto newSwKernelClusteringOp = permuteCastOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
            auto newSwKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(newSwKernelClusteringOp.getInnerTaskOp());
            auto origClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
            swKernelOp = newSwKernelOp;
            rewriter.replaceOp(origClusterTilingOp, permuteCastOp->getResult(0));
            _log.trace("Adjust layout to insert subview succeed");
        }
    }

    if (requireBalancingShapeCast(swKernelOp)) {
        // Replace SW with ShapeCast-SW-ShapeCast
        auto fuseResult = getSWKernelWithFusedDims(swKernelOp, rewriter);
        if (mlir::failed(fuseResult)) {
            _log.trace("balance tiling failed");
        } else {
            auto shapeCastOp = fuseResult.value();
            auto newSwKernelClusteringOp = shapeCastOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
            auto newSwKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(newSwKernelClusteringOp.getInnerTaskOp());
            auto origClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
            swKernelOp = newSwKernelOp;
            rewriter.replaceOp(origClusterTilingOp, shapeCastOp->getResult(0));
            _log.trace("balance tiling succeed");
        }
    }

    // check output tiles on all shaves
    auto outTiles = calculateOutputTiles(swKernelOp);
    if (!outTiles.has_value()) {
        return mlir::failure();
    }

    // check input tiles on all shaves
    auto inTiles = calculateInputTiles(swKernelOp);
    if (!inTiles.has_value()) {
        return mlir::failure();
    }

    auto insertSubview = needInsertSubviewOnly(swKernelOp);
    _log.trace("only insert subview is {0}", insertSubview);

    if (!checkTilePattern(swKernelOp, insertSubview)) {
        return mlir::failure();
    }

    _log.trace("process swKernelOp at {0}", swKernelOp->getLoc());

    SmallVector<mlir::Value> newInputs;
    SmallVector<mlir::Value> newOutBuffs;

    SmallVector<SmallVector<mlir::Attribute>> newAttrs;
    auto tileSize = getShaveTileSize(swKernelOp, outTiles.value());
    for (auto tileIndex : irange(tileSize)) {
        auto inputs = getOuterMappingOperand(swKernelOp, swKernelOp.getInputs());
        auto outBuffs = getOuterMappingOperand(swKernelOp, swKernelOp.getOutputBuffs());

        newInputs.append(createNewInputs(swKernelOp, inputs, insertSubview, tileIndex, rewriter));
        newOutBuffs.append(createNewOutBuffs(swKernelOp, outBuffs, insertSubview, tileIndex, rewriter));
        newAttrs.push_back(updateSwKernelAttrs(swKernelOp, tileIndex));
    }

    auto newSwKernelOp = createNewSwKernelOp(swKernelOp, newInputs, newOutBuffs, insertSubview, rewriter);
    replaceOpWithConcatView(swKernelOp, newSwKernelOp, insertSubview, rewriter);
    auto newSwKernelRuns = newSwKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    auto newSwKernelRunIter = newSwKernelRuns.begin();
    for (auto idx : irange(tileSize)) {
        VPUX_THROW_WHEN(newSwKernelRunIter == newSwKernelRuns.end(), "Cannot get SwKernelRun Op for output tile {0} ",
                        idx);
        auto newSwKernelRun = *newSwKernelRunIter;
        newSwKernelRun.setAttrsAttr(mlir::ArrayAttr::get(newSwKernelOp->getContext(), newAttrs[idx]));
        newSwKernelRunIter++;
    }
    return mlir::success();
}

bool SwKernelRewriterBase::needInsertSubviewOnly(VPUIP::SwKernelOp swKernelOp) const {
    // We can insert subview without strided data access in case all tensors are split on the highest dimension, as
    // all tiled tensors can have contiguous data in memory

    // The operator that has multiple inputs and outputs should be handled correctly
    // For example, a TopK layer in below
    // input shape: 1x5x128x512xf16@NCHW
    // output shape: 1x1x128x512xf16@NCHW and 1x1x128x512xsi32@NCHW
    // Output tensor is split on d2 but d2 is not the highest dimension of input (1x5x128x512xf16@NCHW)
    // SubView is not feasible for this case

    // Gather Op manages two inputs (inData, indices) and one output across scenarios with varying tensor ranks
    // Note: inData and indices may not be tiled simultaneously. Examples:
    // 1. Scenario:
    //    - Input: 1x6x12, Indices: 4, Batch_Dim: 0, Axis: 1
    //    - Output: 1x4x12
    //    - Tiling: Output tileDim is Dim(1), Input does not require tiling, Indices tileDim is Dim(0).
    // 2. Scenario:
    //    - Input: 8x6x12, Indices: 8x1x4, Batch_Dim: 1, Axis: 1
    //    - Output: 8x1x4x12
    //    - Tiling: Output tileDim is Dim(0), Input tileDim is Dim(0), Indices do not require tiling.
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "gather") {
        return isGatherOpTileAtHighestDim(swKernelOp);
    }

    const auto tileDim = getSwKernelTileDim(swKernelOp);

    auto isSplitOnTheHighestDimension = [&](auto type) {
        return tileDim == getHighestNonTrivialDim(type.getShape(), type.getDimsOrder()).value_or(Dim(0));
    };

    auto isMemContiguous = llvm::all_of(getSwKernelTiledTypes(swKernelOp, tileDim), isSplitOnTheHighestDimension);
    if (isMemContiguous) {
        return true;
    }

    // If swkernel doesn't support strided data access, the tiling input has to be created by subview and copy to
    // make sure the new input is continuous
    return isStridedDataAccessSupported(swKernelOp);
}

SmallVector<mlir::Attribute> SwKernelRewriterBase::updateSwKernelAttrs(VPUIP::SwKernelOp swKernelOp,
                                                                       int64_t outTileIndexInsideCluster) const {
    auto swKernelRun = *swKernelOp.getBody().getOps<VPUIP::SwKernelRun>().begin();
    if (!swKernelRun.getAttrs().has_value()) {
        return {};
    }

    const auto outTiles = getOuterMostOutputTiling(swKernelOp);
    const auto inputTiles = getOuterMostInputTiling(swKernelOp, outTileIndexInsideCluster);
    auto origAttr = swKernelRun.getAttrs().value();
    SmallVector<mlir::Attribute> attrs(origAttr.begin(), origAttr.end());
    return VPUIP::getSwkernelNewAttrsAfterTiling(swKernelOp, attrs, inputTiles, outTiles[outTileIndexInsideCluster],
                                                 _log);
}

//
// SwKernelRewriter
//

class SwKernelRewriter final : public SwKernelRewriterBase {
public:
    SwKernelRewriter(mlir::MLIRContext* ctx, int64_t shaveCout, Logger log): SwKernelRewriterBase(ctx, shaveCout, log) {
        setDebugName("SwKernelRewriter");
    }

    bool checkTilePattern(VPUIP::SwKernelOp swKernelOp, bool insertSubview) const override;
    std::optional<OutputTiling> calculateOutputTiles(VPUIP::SwKernelOp swKernelOp) const override;
    std::optional<SmallVector<InputTiling>> calculateInputTiles(VPUIP::SwKernelOp swKernelOp) const override;
    size_t getShaveTileSize(VPUIP::SwKernelOp swKernelOp, const OutputTiling& outTiles) const override;
    SmallVector<mlir::Value> createNewInputs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                             bool insertSubview, int64_t outTileIndex,
                                             mlir::PatternRewriter& rewriter) const override;
    SmallVector<mlir::Value> createNewOutBuffs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                               bool insertSubview, int64_t outTileIndex,
                                               mlir::PatternRewriter& rewriter) const override;

    VPUIP::SwKernelOp createNewSwKernelOp(VPUIP::SwKernelOp swKernelOp, ArrayRef<mlir::Value> newInputs,
                                          ArrayRef<mlir::Value> newOutBufs, bool insertSubview,
                                          mlir::PatternRewriter& rewriter) const override;
    mlir::FailureOr<VPUIP::ShapeCastOp> getSWKernelWithFusedDims(VPUIP::SwKernelOp swKernelOp,
                                                                 mlir::PatternRewriter& rewriter) const override;
    mlir::FailureOr<VPUIP::PermuteCastOp> adjustSWLayout(VPUIP::SwKernelOp swKernelOp,
                                                         mlir::PatternRewriter& rewriter) const override;
    void replaceOpWithConcatView(VPUIP::SwKernelOp origOp, VPUIP::SwKernelOp newSwkernelOp, bool insertSubview,
                                 mlir::PatternRewriter& rewriter) const override;

    OutputTiling getOuterMostOutputTiling(VPUIP::SwKernelOp swKernelOp) const override;
    InputTiling getOuterMostInputTiling(VPUIP::SwKernelOp swKernelOp, int64_t outTileIndx) const override;
    bool requireBalancingShapeCast(VPUIP::SwKernelOp swKernelOp) const override;
    bool requireLayoutChangePermuteCast(VPUIP::SwKernelOp swKernelOp) const override;
};

bool SwKernelRewriter::requireBalancingShapeCast(VPUIP::SwKernelOp /*swKernelOp*/) const {
    // Track E#126764: extend shave balancing for single cluster sw kernels
    return false;
}

mlir::FailureOr<VPUIP::ShapeCastOp> SwKernelRewriter::getSWKernelWithFusedDims(
        VPUIP::SwKernelOp /*swKernelOp*/, mlir::PatternRewriter& /*rewriter*/) const {
    // No need for single cluster op
    return mlir::failure();
}

bool SwKernelRewriter::requireLayoutChangePermuteCast(VPUIP::SwKernelOp /*swKernelOp*/) const {
    // For non-clustered eltwise operations, the tile dimension remains at the highest dimension
    // layout changes are unnecessary
    return false;
}

mlir::FailureOr<VPUIP::PermuteCastOp> SwKernelRewriter::adjustSWLayout(VPUIP::SwKernelOp /*swKernelOp*/,
                                                                       mlir::PatternRewriter& /*rewriter*/) const {
    // No need for none cluster op
    return mlir::failure();
}

bool SwKernelRewriter::checkTilePattern(VPUIP::SwKernelOp swKernelOp, bool insertSubview) const {
    if (mlir::isa<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp())) {
        return false;
    }
    if (insertSubview) {
        return true;
    }

    // Strided data access is not supported, will try to insert extra copy ops for inputs and output buf. So
    // need to check the cmx requirement for:
    // 1. the new input tile copy(CMX2CMX) ops
    // 2. the new output tile copy(CMX2CMX) ops
    // 3. the new swkernel op
    auto getNewTiledAllocSize = [](mlir::Value origOperand, ShapeRef newTiledShape) {
        auto origType = origOperand.getType().dyn_cast<vpux::NDTypeInterface>();
        auto newTiledType = origType.changeShape(newTiledShape);
        return newTiledType.getTotalAllocSize();
    };

    auto totalCMXSize = VPU::getTotalCMXSize(swKernelOp);
    auto inputs = getOuterMappingOperand(swKernelOp, swKernelOp.getInputs());
    auto outTiles = getOuterMostOutputTiling(swKernelOp);
    const auto outType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getResult(0).getType());
    Byte requiredCMXForTiledSwKernelOp(0);
    Byte allocCMXSizeForOutTiles(outType.getTotalAllocSize());
    for (auto outIndex : irange(outTiles.size())) {
        const auto inTiles = getOuterMostInputTiling(swKernelOp, outIndex);
        for (const auto& item : inputs | indexed) {
            auto input = item.value();
            auto index = item.index();
            auto newInputRequiredSize = getNewTiledAllocSize(input, inTiles.tiles[index].shape);
            // Check CMX requirement for each input tile Copy + total alloc for input Subview
            const auto inType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getInputs()[index].getType());
            Byte allocCMXSizeForInTile(inType.getTotalAllocSize());
            Byte requiredCMXForSubviewAndInputCopy = allocCMXSizeForInTile + newInputRequiredSize;
            if (requiredCMXForSubviewAndInputCopy > totalCMXSize) {
                return false;
            }
            requiredCMXForTiledSwKernelOp += newInputRequiredSize;
        }
        auto newOutputRequiredSize = getNewTiledAllocSize(swKernelOp.getResult(0), outTiles[outIndex].shape);
        // Check CMX requirement for each output tile Copy + total alloc for output Subview
        Byte requiredCMXForSubviewAndCopyOut = allocCMXSizeForOutTiles + newOutputRequiredSize;
        if (requiredCMXForSubviewAndCopyOut > totalCMXSize) {
            return false;
        }
        requiredCMXForTiledSwKernelOp += newOutputRequiredSize;
    }

    return requiredCMXForTiledSwKernelOp <= totalCMXSize;
}

std::optional<OutputTiling> SwKernelRewriter::calculateOutputTiles(VPUIP::SwKernelOp swKernelOp) const {
    auto insertSubview = needInsertSubviewOnly(swKernelOp);
    auto tiles =
            getSwKernelOutputTiling(swKernelOp, getShape(swKernelOp.getResult(0)), _shaveCount, insertSubview, _log);
    if (mlir::failed(tiles)) {
        return std::nullopt;
    }

    auto outTiles = tiles.value();
    return outTiles.size() == 1 ? std::optional<OutputTiling>{} : outTiles;
}

std::optional<SmallVector<InputTiling>> SwKernelRewriter::calculateInputTiles(VPUIP::SwKernelOp swKernelOp) const {
    auto outTiles = calculateOutputTiles(swKernelOp);
    if (!outTiles.has_value()) {
        return std::nullopt;
    }
    SmallVector<InputTiling> inTiles;
    auto outTilesValues = outTiles.value();
    for (int i = 0; i < static_cast<int>(outTilesValues.size()); i++) {
        inTiles.push_back(VPUIP::backInferSwKernelInputTile(swKernelOp, outTilesValues, i, _log));
    }
    return inTiles;
}

size_t SwKernelRewriter::getShaveTileSize(VPUIP::SwKernelOp, const OutputTiling& outTiles) const {
    return outTiles.size();
}

SmallVector<mlir::Value> SwKernelRewriter::createNewInputs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                                           bool insertSubview, int64_t outTileIndex,
                                                           mlir::PatternRewriter& rewriter) const {
    const auto inShaveTiles = calculateInputTiles(swKernelOp).value();
    const auto& inTiles = inShaveTiles[outTileIndex];
    SmallVector<mlir::Value> newInputs;
    for (const auto& p : operands | indexed) {
        const auto& index = p.index();
        const auto& operand = p.value();
        const auto& offset = inTiles.tiles[index].offsets;
        const auto& tiledShape = inTiles.tiles[index].shape;

        // handle swkernel's input copy
        if (insertSubview) {
            auto inputSubview = rewriter.create<VPUIP::SubViewOp>(operand.getLoc(), operand, offset, tiledShape);
            newInputs.push_back(inputSubview);
        } else {
            /*
                If there is a CopyOp, create new CopyOps to replace it.
                eg:
                    input
                      |
                     copy
                      |
                 single-shaveOp

                     ||
                     \/

                    input
                    /  \
               subview subview
                   |    |
                  copy copy
                    \ /
                multi-shaveOp
            */
            auto inputCopyOp = operand.getDefiningOp<VPUIP::CopyOp>();
            auto inputSubview = rewriter.create<VPUIP::SubViewOp>(
                    operand.getLoc(), inputCopyOp ? inputCopyOp.getInput() : operand, offset, tiledShape);
            auto allocType = operand.getType().dyn_cast<vpux::NDTypeInterface>();
            auto newAllocType = allocType.changeShape(tiledShape);
            auto newInputAllocOp =
                    rewriter.create<mlir::memref::AllocOp>(operand.getLoc(), newAllocType.cast<mlir::MemRefType>());
            auto newCopyOp =
                    rewriter.create<VPUIP::CopyOp>(operand.getLoc(), inputSubview.getResult(), newInputAllocOp);
            newInputs.push_back(newCopyOp);
        }
    }
    return newInputs;
}

TileInfo inferHoOutput(const TileInfo& tilesY) {
    // The rank of outputHo equals 3.
    TileInfo tilesHo(3);
    tilesHo.shape[Dim(0)] = tilesY.shape[Dim(0)];
    tilesHo.shape[Dim(1)] = tilesY.shape[Dim(1)];
    tilesHo.shape[Dim(2)] = tilesY.shape[Dim(3)];
    tilesHo.offsets[Dim(0)] = tilesY.offsets[Dim(0)];
    tilesHo.offsets[Dim(1)] = tilesY.offsets[Dim(1)];
    tilesHo.offsets[Dim(2)] = tilesY.offsets[Dim(3)];
    return tilesHo;
}

SmallVector<mlir::Value> SwKernelRewriter::createNewOutBuffs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange outBuffs,
                                                             bool insertSubview, int64_t shaveId,
                                                             mlir::PatternRewriter& rewriter) const {
    const auto perShaveFirstOutputTiles = calculateOutputTiles(swKernelOp).value();

    const auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    auto outputTilesOnShave = computeOutputTiling(kernelEntryName, perShaveFirstOutputTiles[shaveId]);
    VPUX_THROW_UNLESS(outBuffs.size() == outputTilesOnShave.size(),
                      "Number of output buffers must be equal to number of outputs on a shave");

    SmallVector<mlir::Value> newOutputs;
    for (auto p : outBuffs | indexed) {
        const auto& idx = p.index();
        const auto& outBuff = outBuffs[idx];
        const auto& outTile = outputTilesOnShave[idx];
        if (insertSubview) {
            // GRUSequenceOp/GRUSequenceLastPartOp has two different output shapes.
            if (kernelEntryName == "gru_sequence" || kernelEntryName == "gru_sequence_last_part") {
                if (idx == 0) {
                    auto outputYSubview = rewriter.create<VPUIP::SubViewOp>(outBuff.getLoc(), outBuff, outTile.offsets,
                                                                            outTile.shape);
                    newOutputs.push_back(outputYSubview);
                } else {
                    const auto outputYTiles = perShaveFirstOutputTiles[shaveId];
                    auto tiledHoOutputTile = inferHoOutput(outputYTiles);
                    auto tiledHoShape = tiledHoOutputTile.shape;
                    auto tiledHoOffset = tiledHoOutputTile.offsets;
                    auto outputHoSubview =
                            rewriter.create<VPUIP::SubViewOp>(outBuff.getLoc(), outBuff, tiledHoOffset, tiledHoShape);
                    newOutputs.push_back(outputHoSubview);
                }
            } else {
                auto outputSubview =
                        rewriter.create<VPUIP::SubViewOp>(outBuff.getLoc(), outBuff, outTile.offsets, outTile.shape);
                newOutputs.push_back(outputSubview);
            }
        } else {
            auto allocType = outBuff.getType().cast<vpux::NDTypeInterface>();
            auto newAllocType = allocType.changeShape(outTile.shape);
            auto newOutputAllocOp =
                    rewriter.create<mlir::memref::AllocOp>(outBuff.getLoc(), newAllocType.cast<mlir::MemRefType>());
            newOutputs.push_back(newOutputAllocOp);
        }
    }
    return newOutputs;
}

VPUIP::SwKernelOp SwKernelRewriter::createNewSwKernelOp(VPUIP::SwKernelOp swKernelOp, ArrayRef<mlir::Value> newInputs,
                                                        ArrayRef<mlir::Value> newOutBufs, bool,
                                                        mlir::PatternRewriter& rewriter) const {
    auto newSwKernelTask = rewriter.create<VPUIP::SwKernelOp>(
            swKernelOp->getLoc(), newInputs, newOutBufs, swKernelOp.getKernelFunction(), swKernelOp.getTileIndexAttr());
    auto swKernelRun = *swKernelOp.getBody().getOps<VPUIP::SwKernelRun>().begin();
    VPUIP::initSwKernel(newSwKernelTask, swKernelRun, _log);

    _log.trace("create new swKernel op {0}", newSwKernelTask);
    return newSwKernelTask;
}

void SwKernelRewriter::replaceOpWithConcatView(VPUIP::SwKernelOp origOp, VPUIP::SwKernelOp newSwKernelOp,
                                               bool insertSubview, mlir::PatternRewriter& rewriter) const {
    const auto origNumberResults = origOp->getNumResults();
    const auto newNumberResults = newSwKernelOp->getNumResults();
    VPUX_THROW_UNLESS(newNumberResults % origNumberResults == 0, "Invalid result number at {0}", origOp->getLoc());
    const auto numberActShaveTiles = newNumberResults / origNumberResults;

    // Get input ops
    SmallVector<mlir::Operation*> inputOps;
    for (const auto& input : origOp.getInputs()) {
        if (const auto& inputOp = input.getDefiningOp()) {
            inputOps.push_back(inputOp);
        }
    }

    const auto getOutputTiles = [&]() {
        if (insertSubview) {
            return OutputTiling{};
        } else {
            const auto outTilesFront = getOuterMostOutputTiling(origOp);
            VPUX_THROW_UNLESS(outTilesFront.size() == numberActShaveTiles, "Invalid tiles number at {0}",
                              newSwKernelOp->getLoc());
            return outTilesFront;
        }
    };

    const auto outTiles = getOutputTiles();

    for (auto resultIndx : irange(origNumberResults)) {
        auto output = origOp->getResult(resultIndx);

        auto handleOutputReplacement = [&](auto origOutBufOp) {
            if (insertSubview) {
                SmallVector<mlir::Value> subResults;
                subResults.reserve(numberActShaveTiles);
                for (auto index : irange(numberActShaveTiles)) {
                    subResults.push_back(newSwKernelOp->getResult(origNumberResults * index + resultIndx));
                }
                auto concatOp = rewriter.create<VPUIP::ConcatViewOp>(origOp->getLoc(), subResults, origOutBufOp);
                output.replaceAllUsesWith(concatOp.getOutput());
            } else {
                auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
                rewriter.setInsertionPointAfterValue(output);
                auto outBufOp =
                        rewriter.create<mlir::memref::AllocOp>(output.getLoc(), outputType.cast<mlir::MemRefType>());

                SmallVector<mlir::Value> results;
                results.reserve(outTiles.size());
                for (const auto& item : outTiles | indexed) {
                    const auto& outTile = item.value();
                    const auto& index = item.index();
                    auto outShape = to_small_vector(outTile.shape);
                    auto outOffset = to_small_vector(outTile.offsets);
                    auto outSubview =
                            rewriter.create<VPUIP::SubViewOp>(newSwKernelOp->getLoc(), outBufOp, outOffset, outShape);
                    auto copyOp = rewriter.create<VPUIP::CopyOp>(
                            newSwKernelOp->getLoc(), newSwKernelOp.getResult(origNumberResults * index + resultIndx),
                            outSubview);
                    results.push_back(copyOp);
                }

                auto concatOp = rewriter.create<VPUIP::ConcatViewOp>(origOp->getLoc(), results, outBufOp);
                output.replaceAllUsesWith(concatOp.getOutput());
                if (origOutBufOp->use_empty()) {
                    rewriter.eraseOp(origOutBufOp);
                }
            }
        };

        auto origOutBufOp = origOp.getOutputBuffs()[resultIndx].getDefiningOp();
        if (VPUIP::hasBoundedBuffers(origOutBufOp)) {
            auto origOutBufOpCast = mlir::dyn_cast<VPUIP::GroupBoundedBufferOp>(origOutBufOp);
            handleOutputReplacement(origOutBufOpCast);
        } else {
            auto origOutBufOpCast = mlir::dyn_cast<mlir::memref::AllocOp>(origOutBufOp);
            handleOutputReplacement(origOutBufOpCast);
        }
    }
    rewriter.eraseOp(origOp);

    std::set<mlir::Operation*> uniqueInputSet(inputOps.begin(), inputOps.end());
    for (auto originInputOp : uniqueInputSet) {
        if (originInputOp != nullptr && originInputOp->use_empty()) {
            rewriter.eraseOp(originInputOp);
        }
    }
}

OutputTiling SwKernelRewriter::getOuterMostOutputTiling(VPUIP::SwKernelOp swKernelOp) const {
    return calculateOutputTiles(swKernelOp).value();
}

InputTiling SwKernelRewriter::getOuterMostInputTiling(VPUIP::SwKernelOp swKernelOp, int64_t outTileIndx) const {
    const auto inTiles = calculateInputTiles(swKernelOp).value();
    return inTiles[outTileIndx];
}

//
// ClusterSwKernelRewriter
//

class ClusterSwKernelRewriter final : public SwKernelRewriterBase {
public:
    ClusterSwKernelRewriter(mlir::MLIRContext* ctx, int64_t shaveCout, Logger log)
            : SwKernelRewriterBase(ctx, shaveCout, log) {
        setDebugName("ClusterSwKernelRewriter");
    }

    bool checkTilePattern(VPUIP::SwKernelOp swKernelOp, bool insertSubview) const override;
    bool needInsertSubviewOnly(VPUIP::SwKernelOp swKernelOp) const override;
    std::optional<OutputTiling> calculateOutputTiles(VPUIP::SwKernelOp swKernelOp) const override;
    std::optional<SmallVector<InputTiling>> calculateInputTiles(VPUIP::SwKernelOp swKernelOp) const override;
    size_t getShaveTileSize(VPUIP::SwKernelOp swKernelOp, const OutputTiling& outTiles) const override;
    SmallVector<mlir::Value> createNewInputs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                             bool insertSubview, int64_t outTileIndex,
                                             mlir::PatternRewriter& rewriter) const override;
    SmallVector<mlir::Value> createNewOutBuffs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                               bool insertSubview, int64_t outTileIndex,
                                               mlir::PatternRewriter& rewriter) const override;
    VPUIP::SwKernelOp createNewSwKernelOp(VPUIP::SwKernelOp swKernelOp, ArrayRef<mlir::Value> newInputs,
                                          ArrayRef<mlir::Value> newOutBufs, bool insertSubview,
                                          mlir::PatternRewriter& rewriter) const override;
    mlir::FailureOr<VPUIP::ShapeCastOp> getSWKernelWithFusedDims(VPUIP::SwKernelOp swKernelOp,
                                                                 mlir::PatternRewriter& rewriter) const override;
    mlir::FailureOr<VPUIP::PermuteCastOp> adjustSWLayout(VPUIP::SwKernelOp swKernelOp,
                                                         mlir::PatternRewriter& rewriter) const override;
    void replaceOpWithConcatView(VPUIP::SwKernelOp origOp, VPUIP::SwKernelOp newSwKernelOp, bool insertSubview,
                                 mlir::PatternRewriter& rewriter) const override;
    OutputTiling getOuterMostOutputTiling(VPUIP::SwKernelOp swKernelOp) const override;
    InputTiling getOuterMostInputTiling(VPUIP::SwKernelOp swKernelOp, int64_t outTileIndx) const override;

    SmallVector<OutputTiling> getMultiOutputTiling(VPUIP::SwKernelOp swKernelOp,
                                                   const OutputTiling& perClusterFirstOutputTiles);

    bool tileOnDifferentDims(VPUIP::SwKernelOp swKernelOp) const;
    bool requireBalancingShapeCast(VPUIP::SwKernelOp swKernelOp) const override;
    bool requireLayoutChangePermuteCast(VPUIP::SwKernelOp swKernelOp) const override;

private:
    bool onlyHasCopyOpUser(VPUIP::SwKernelOp swKernelOp) const;
    vpux::NDTypeInterface getNewTiledDistributedType(
            VPUIP::SwKernelOp swKernelOp, mlir::Value outerOperand, int64_t outTileIndex, ShapeRef tiledShape,
            std::function<TileInfo(int64_t clusterId, int64_t shaveId, int64_t numClusters, VPU::DistributionMode mode,
                                   bool insertSubview)>) const;
    std::optional<vpux::NDTypeInterface> getImplicitDistributedType(VPUIP::SwKernelOp swkernelOp,
                                                                    VPUIP::DistributedBufferType srcDistributedType,
                                                                    ShapeRef newShape,
                                                                    ArrayRef<SmallVector<int64_t>> tiledShape,
                                                                    ArrayRef<SmallVector<int64_t>> tiledOffset) const;
    template <class TileClass>
    TileClass getTileFromList(const SmallVector<TileClass>& tiles, int64_t clusterId, int64_t shaveId, int64_t numTiles,
                              VPU::DistributionMode mode, bool insertSubview) const;
    mlir::ArrayAttr getStrideOnEachCluster(VPUIP::SwKernelOp swKernelOp, bool insertSubview) const;
};

bool ClusterSwKernelRewriter::requireBalancingShapeCast(VPUIP::SwKernelOp swKernelOp) const {
    // Uneven tiling causes worse compute efficiency
    // Use ShapeCast to fuse dimensions and balance the tiling on shaves

    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    bool supported = llvm::find(SW_KERNELS_SUPPORTING_SHAVE_BALANCING, kernelEntryName) !=
                     SW_KERNELS_SUPPORTING_SHAVE_BALANCING.end();
    if (!supported) {
        return false;
    }

    const auto tileDim = getSwKernelTileDim(swKernelOp);
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (clusterTilingOp == nullptr) {
        return false;
    }
    // Track #E125638
    // Other modes should be supported
    const auto hasNonSegOperand = llvm::any_of(clusterTilingOp->getOperands(), [](mlir::Value operand) {
        if (auto operandDistType = operand.getType().dyn_cast<VPUIP::DistributedBufferType>()) {
            auto mode = operandDistType.getDistribution().getMode().getValue();
            return mode != VPU::DistributionMode::SEGMENTED;
        }
        return true;
    });
    if (hasNonSegOperand) {
        return false;
    }

    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();

    bool requiresBalancing = llvm::any_of(perClusterShapes, [&](auto clusterShape) {
        return clusterShape[tileDim] % _shaveCount != 0;
    });
    if (requiresBalancing) {
        _log.trace("SwKernelOp {0} requires balancing tiling", swKernelOp->getName());
        return true;
    }

    // Try to balance shape if the start address of the second shave is not aligned as required
    auto inTiles = calculateInputTiles(swKernelOp);
    auto needInserSubview = needInsertSubviewOnly(swKernelOp);
    if (inTiles.has_value() && needInserSubview) {
        const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
        const auto elemSize = distributedType.getElemTypeSize();
        auto mode = distributedType.getDistribution().getMode().getValue();
        const auto arch = VPU::getArch(swKernelOp);
        for (auto clusterId : irange(numClusters)) {
            // no need to check last shave
            for (auto shaveId : irange(_shaveCount - 1)) {
                auto tiles =
                        getTileFromList(inTiles.value(), clusterId, shaveId, numClusters, mode, needInserSubview).tiles;
                // here only check the first input for eltwise like operation
                auto totalSize = tiles.front().shape.totalSize() * elemSize;
                if (Byte(totalSize).count() % VPUIP::getSwKernelTilingAddressAlignment(swKernelOp, arch) != 0) {
                    _log.trace("SwKernelOp {0} requires balancing tiling for address align", swKernelOp->getName());
                    return true;
                }
            }
        }
    }

    return false;
}

bool ClusterSwKernelRewriter::requireLayoutChangePermuteCast(VPUIP::SwKernelOp swKernelOp) const {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (llvm::find(SW_KERNELS_LAYOUT_AGNOSTIC, kernelEntryName) == SW_KERNELS_LAYOUT_AGNOSTIC.end()) {
        return false;
    }

    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (clusterTilingOp == nullptr) {
        return false;
    }

    if (clusterTilingOp.getResults().size() != 1) {
        return false;
    }

    const auto outputType = mlir::cast<NDTypeInterface>(clusterTilingOp.getResult(0).getType());
    const auto tileDim = getSwKernelTileDim(swKernelOp);
    const auto highestDim = getHighestNonTrivialDim(outputType.getShape(), outputType.getDimsOrder()).value_or(Dim(0));

    return tileDim != highestDim;
}

// Pick up the dimensions that can be fused to the tileDim
// if mcDimFusible is true, the multi cluster dimension is fusible, otherwise it's not.
DimArr getFusibleDims(VPUIP::SwKernelOp swKernelOp, Dim tileDim, bool mcDimFusible = false) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    mlir::DenseSet<size_t> forbiddenDims;
    if (kernelEntryName == "softmax") {
        // The softmax axis can't be fused otherwise the inference would be wrong
        auto taskArgs = kernelArgsRange(swKernelOp);
        const auto kernelAxis = taskArgs.front().cast<mlir::IntegerAttr>().getInt();
        forbiddenDims.insert(convertKernelAxisToDim(swKernelOp.getResult(0), kernelAxis).ind());
    } else if (kernelEntryName == "eltwise_mul") {
        // If one of the two inputs are broadcast
        // this dimension can't be fused otherwise the broadcast won't work
        VPUX_THROW_UNLESS(swKernelOp->getOperands().size() >= 2, "invalid inputs number for eltwise_mul");
        const auto inType0 = swKernelOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto inType1 = swKernelOp->getOperand(1).getType().cast<vpux::NDTypeInterface>();
        VPUX_THROW_UNLESS(inType0.getRank() == inType1.getRank(), "The two inputs' ranks are not aligned");
        auto inShape0 = inType0.getShape();
        auto inShape1 = inType1.getShape();
        for (const auto ind : irange(inType0.getRank())) {
            if (inShape0[Dim(ind)] != inShape1[Dim(ind)]) {
                forbiddenDims.insert(ind);
            }
        }
    }
    auto clusterOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (!mcDimFusible && clusterOp != nullptr) {
        // If the multiCluster tiling is on a different dimension
        // this dimension can't be fused
        auto distributedType = clusterOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
        auto distributionAttr = distributedType.getDistribution();
        if (auto numTiles = distributionAttr.getNumTiles()) {
            const auto multiClusterAxis =
                    Dim(vpux::VPU::getDistributedTilingAxis(parseIntArrayAttr<int64_t>(numTiles)));
            if (multiClusterAxis != tileDim) {
                // Track E#127193: support to fuse multiClusterAxis
                forbiddenDims.insert(multiClusterAxis.ind());
            }
        }
    }
    const auto outType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outOrderList = outType.getDimsOrder().toPermutation();
    // The fusible dimensions can only be on the same side of the tileDim
    // can't cross the forbidden dimensions
    // e.g.
    //     order    N       X       Y       Z           N   X   1   (Y*Z)
    //                  (forbid)          (tile)    ->  only Y is fusible
    //              N       X       Y       Z           1   (N*X)   Y   Z
    //                   (tile) (forbid)            ->  only N is fusible
    //              N       X       Y       Z           1   (N*X*Y)   1   Z
    //                   (tile)         (forbid)    ->  N and Y are fusible
    //              N       X       Y       Z           N   (X*Y)   1   Z
    //           (forbid)  (tile)       (forbid)    ->  only Y fusible
    auto fusibleDims = SmallVector({tileDim});
    auto tileDimInd = std::distance(outOrderList.begin(), llvm::find(outOrderList, tileDim));
    SmallVector<int64_t> forbidDimInds;
    llvm::transform(forbiddenDims, std::back_inserter(forbidDimInds), [&](size_t ind) {
        return std::distance(outOrderList.begin(), llvm::find(outOrderList, Dim(ind)));
    });
    auto hasForbidDimInBetween = [&](Dim d) {
        // Check if any dimension between tileDim and the current dimension is forbidden to fuse
        auto dimInd = std::distance(outOrderList.begin(), llvm::find(outOrderList, d));
        return llvm::any_of(forbidDimInds, [&](int64_t forbidInd) {
            return (forbidInd > tileDimInd && forbidInd < dimInd) || (forbidInd < tileDimInd && forbidInd > dimInd);
        });
    };
    auto outShape = outType.getShape();
    for (const auto dim : outOrderList) {
        if (outShape[dim] == 1) {
            continue;
        }
        if (llvm::find(forbiddenDims, dim.ind()) != forbiddenDims.end() || tileDim == dim) {
            continue;
        }
        if (hasForbidDimInBetween(dim)) {
            continue;
        }
        fusibleDims.push_back(dim);
    }
    return fusibleDims;
}

mlir::FailureOr<VPUIP::PermuteCastOp> ClusterSwKernelRewriter::adjustSWLayout(VPUIP::SwKernelOp swKernelOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (clusterTilingOp == nullptr) {
        return mlir::failure();
    }

    auto distributedOutType = mlir::dyn_cast<VPUIP::DistributedBufferType>(clusterTilingOp.getResult(0).getType());
    const auto tileDim = getSwKernelTileDim(swKernelOp);
    const auto highestDim =
            getHighestNonTrivialDim(distributedOutType.getShape(), distributedOutType.getDimsOrder()).value_or(Dim(0));

    // The tileDim and highestDim can only be swapped when they are fusible
    // Otherwise the conversion breaks computation
    // e.g., multiply with one input broadcast on W, original input is NHWC,
    //       it can't be converted to NCWH because the broadcast data would be different
    const auto fusibleDims = getFusibleDims(swKernelOp, tileDim, true);
    if (llvm::find(fusibleDims, tileDim) == fusibleDims.end() ||
        llvm::find(fusibleDims, highestDim) == fusibleDims.end()) {
        return mlir::failure();
    }

    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "eltwise_mul") {
        // If one of the two inputs need broadcast, skip ths case due to it will case accuracy issue.
        // For example, 1x3x1x2xf16 broadcast to 1x3x160x2xf16, tiling over H, if just insert PermuteCast, the value
        // after broadcasted is not same.
        const auto inType0 = swKernelOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto inType1 = swKernelOp->getOperand(1).getType().cast<vpux::NDTypeInterface>();
        auto inShape0 = inType0.getShape();
        auto inShape1 = inType1.getShape();
        if (inShape0 != inShape1) {
            return mlir::failure();
        }
    }

    const auto origOrder = distributedOutType.getDimsOrder();
    const auto tileInd = origOrder.dimPos(tileDim);
    const auto highestDimInd = origOrder.dimPos(highestDim);

    // Only the tileDim and highestDim are swapped
    // Example: For a SW Op using 'NHWC' layout with tileDim as 'C' and highestDim as 'H'
    // the target dimension order for the SW is 'NCWH'
    auto newDimArr = origOrder.toPermutation();
    newDimArr[highestDimInd] = tileDim;
    newDimArr[tileInd] = highestDim;
    auto targetDimOrder = DimsOrder::fromPermutation(newDimArr);

    SmallVector<mlir::Value> newInputs;
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(clusterTilingOp);
    for (auto inputInd : irange(clusterTilingOp.getInputs().size())) {
        auto distributedInType =
                clusterTilingOp.getOperand(inputInd).getType().dyn_cast<VPUIP::DistributedBufferType>();

        const auto inPermuteOutType = distributedInType.changeDimsOrder(targetDimOrder);
        const auto inPermAttr = mlir::AffineMapAttr::get(targetDimOrder.toAffineMap(ctx));
        auto inPermuteCastOp = rewriter.create<VPUIP::PermuteCastOp>(
                swKernelOp->getLoc(), inPermuteOutType, clusterTilingOp.getOperand(inputInd), inPermAttr, inPermAttr);
        newInputs.push_back(inPermuteCastOp);
    }

    const auto newDistributedType = distributedOutType.changeDimsOrder(targetDimOrder);
    auto newSWAllocOp =
            rewriter.create<VPURT::AllocDistributed>(swKernelOp->getLoc(), newDistributedType, nullptr, nullptr);
    auto newSwKernelOp = createNewSwKernelOp(swKernelOp, newInputs, {newSWAllocOp}, false, rewriter);
    auto newClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(newSwKernelOp->getParentOp());

    const auto outPermAttr = mlir::AffineMapAttr::get(origOrder.toAffineMap(ctx));
    auto outPermuteCast = rewriter.create<VPUIP::PermuteCastOp>(
            swKernelOp->getLoc(), distributedOutType, newClusterTilingOp.getResult(0), outPermAttr, outPermAttr);

    return outPermuteCast;
}

mlir::FailureOr<VPUIP::ShapeCastOp> ClusterSwKernelRewriter::getSWKernelWithFusedDims(
        VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const {
    const auto tileDim = getSwKernelTileDim(swKernelOp);
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (clusterTilingOp == nullptr) {
        return mlir::failure();
    }
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);

    const auto output = swKernelOp->getResult(0);
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();
    const auto outOrder = outputType.getDimsOrder();
    const auto outShape = outputType.getShape();

    const auto fusibleDims = getFusibleDims(swKernelOp, tileDim);
    if (fusibleDims.size() == 1) {
        return mlir::failure();
    }
    SmallVector<mlir::Value> newInputs;
    auto ctx = rewriter.getContext();
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(clusterTilingOp);
    auto newOutType = clusterTilingOp->getResult(0).getType().cast<NDTypeInterface>();

    auto isShapeCastCorrect = [&](VPUIP::DistributedBufferType shapeCastOutType, SmallVector<Shape>& expectedShapes,
                                  SmallVector<Shape>& expectedOffsets) -> bool {
        // Check if the distribution per cluster is unchanged
        const auto newShapeCastPerClusterShapes = shapeCastOutType.getPerClusterMemoryShapes();
        const auto newShapeCastPerClusterOffsets = shapeCastOutType.getPerClusterMemoryShapeOffsets();
        for (auto ind : irange(newShapeCastPerClusterShapes.size())) {
            if (newShapeCastPerClusterShapes[ind] != expectedShapes[ind] ||
                newShapeCastPerClusterOffsets[ind] != expectedOffsets[ind]) {
                return false;
            }
        }
        // Check if shapecast's alignment breaks the tiling
        return checkSwKernelTilingAlignment(swKernelOp, shapeCastOutType, shapeCastOutType, _log);
    };
    for (auto inputInd : irange(clusterTilingOp.getInputs().size())) {
        auto distributedType = clusterTilingOp.getOperand(inputInd).getType().dyn_cast<VPUIP::DistributedBufferType>();
        const auto multiClusterAxis = Dim(vpux::VPU::getDistributedTilingAxis(
                parseIntArrayAttr<int64_t>(distributedType.getDistribution().getNumTiles())));
        auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
        auto perClusterOffsets = distributedType.getPerClusterMemoryShapeOffsets();
        auto distributedAttr = distributedType.getDistribution();
        const bool hasComputeShapesOffsets =
                distributedAttr.getComputeShapes() != nullptr && distributedAttr.getComputeOffsets() != nullptr;
        const bool hasMemoryShapesOffsets =
                distributedAttr.getMemoryShapes() != nullptr && distributedAttr.getMemoryOffsets() != nullptr;
        const auto hasExplicitInputDistribution = hasComputeShapesOffsets || hasMemoryShapesOffsets;
        const auto input = swKernelOp->getOperand(inputInd);
        const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
        const auto inputShape = inputType.getShape();
        const auto neutralShape = Shape(SmallVector<int64_t>(inputType.getRank(), 1));
        const auto neutralOffsets = Shape(SmallVector<int64_t>(inputType.getRank(), 0));

        auto newPerClusterShapes = SmallVector(perClusterShapes.size(), neutralShape);
        auto newPerClusterOffsets = SmallVector(perClusterOffsets.size(), neutralOffsets);

        const auto fusedNewShape = std::invoke([&] {
            auto newShape = neutralShape;
            for (auto i : irange(outOrder.numDims())) {
                auto dim = outOrder.dimAt(i);
                if (llvm::find(fusibleDims, dim) != fusibleDims.end()) {
                    // If the dimension is fusible, fuse it to the tileDim
                    for (auto clusterInd : irange(newPerClusterShapes.size())) {
                        newPerClusterShapes[clusterInd][tileDim] *= perClusterShapes[clusterInd][dim];
                        if ((clusterInd > 0) && (tileDim == multiClusterAxis)) {
                            newPerClusterOffsets[clusterInd][tileDim] = newPerClusterOffsets[clusterInd - 1][tileDim] +
                                                                        newPerClusterShapes[clusterInd - 1][tileDim];
                        }
                    }
                    newShape[tileDim] *= inputShape[dim];
                } else {
                    // If the dimension in not fusible, keep it unchanged
                    for (auto clusterInd : irange(newPerClusterShapes.size())) {
                        newPerClusterShapes[clusterInd][dim] = perClusterShapes[clusterInd][dim];
                        newPerClusterOffsets[clusterInd][dim] = perClusterOffsets[clusterInd][dim];
                    }
                    newShape[dim] = inputShape[dim];
                }
            }
            return newShape;
        });

        auto newPerClusterShapesAttr = vpux::getIntArrayOfArray(ctx, newPerClusterShapes);
        auto newPerClusterOffsetsAttr = vpux::getIntArrayOfArray(ctx, newPerClusterOffsets);

        VPUIP::ShapeCastOp inShapeCastOp;
        if (!hasExplicitInputDistribution) {
            const auto ctx = distributedType.getContext();
            auto fuseNewShapeArray = ArrayRef(fusedNewShape.begin(), fusedNewShape.end());

            // Check SOH legalization
            auto origDistribution = distributedType.getDistribution();
            const auto mode = origDistribution.getMode().getValue();
            if (VPU::bitEnumContainsAny(mode, VPU::DistributionMode::SEGMENTED)) {
                if (!VPUIP::isDistributedCompatibleAfterShapeChangeForViewOps(
                            distributedType, fusedNewShape, distributedType.getDimsOrder(), VPU::getArch(swKernelOp))) {
                    return mlir::failure();
                }
            }
            auto newDistribution = VPUIP::getDistributedAttrAfterShapeCast<VPUIP::DistributedBufferType>(
                    distributedType, fuseNewShapeArray, VPU::getArch(swKernelOp));
            auto outType = distributedType.changeShapeForExplicitDistribution(fusedNewShape, newDistribution);

            auto newShapeCastOutType =
                    VPUIP::DistributedBufferType::get(ctx, fuseNewShapeArray, outType.getElementType(),
                                                      mlir::AffineMapAttr::get(outType.getDimsOrder().toAffineMap(ctx)),
                                                      outType.getMemSpace(), newDistribution);
            if (!isShapeCastCorrect(newShapeCastOutType, newPerClusterShapes, newPerClusterOffsets)) {
                return mlir::failure();
            }
            inShapeCastOp = rewriter.create<VPUIP::ShapeCastOp>(
                    swKernelOp->getLoc(), clusterTilingOp.getOperand(inputInd), getIntArrayAttr(ctx, fusedNewShape));
        } else {
            inShapeCastOp = rewriter.create<VPUIP::ShapeCastOp>(
                    swKernelOp->getLoc(), clusterTilingOp.getOperand(inputInd), getIntArrayAttr(ctx, fusedNewShape),
                    newPerClusterShapesAttr, newPerClusterOffsetsAttr);
        }
        if (getShape(clusterTilingOp.getOperand(inputInd)) == getShape(clusterTilingOp.getResult(0))) {
            // For SW ops with multiply inputs, some of which are broadcast
            // predict the output type by the input with the same shape
            // e.g., for eltwise_mul kernel with input0 [1, 12, 512, 1] and input1 [1, 12, 512, 512], output [1, 12,
            // 512, 512]
            //      input0 needs broadcast. So the new output type should be the same as new input1
            newOutType = inShapeCastOp.getType().cast<NDTypeInterface>();
        }
        newInputs.push_back(inShapeCastOp);
    }

    const auto newDistributedType = newOutType.dyn_cast<VPU::DistributedTypeInterface>();
    auto newAllocCMXOp =
            rewriter.create<VPURT::AllocDistributed>(swKernelOp->getLoc(), newDistributedType, nullptr, nullptr);

    auto newSwKernelOp = createNewSwKernelOp(swKernelOp, newInputs, {newAllocCMXOp}, false, rewriter);

    auto newClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(newSwKernelOp->getParentOp());

    auto distributedOutType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto prevPerClusterShapesAttr = vpux::getIntArrayOfArray(ctx, distributedOutType.getPerClusterMemoryShapes());
    auto prevPerClusterOffsetsAttr =
            vpux::getIntArrayOfArray(ctx, distributedOutType.getPerClusterMemoryShapeOffsets());

    VPUIP::ShapeCastOp outShapeCastOp;
    auto distributedOutAttr = distributedOutType.getDistribution();
    const bool hasOutputComputeShapesOffsets =
            distributedOutAttr.getComputeShapes() != nullptr && distributedOutAttr.getComputeOffsets() != nullptr;
    const bool hasOutputMemoryShapesOffsets =
            distributedOutAttr.getMemoryShapes() != nullptr && distributedOutAttr.getMemoryOffsets() != nullptr;
    const auto hasExplicitOutputDistribution = hasOutputComputeShapesOffsets || hasOutputMemoryShapesOffsets;
    if (!hasExplicitOutputDistribution) {
        outShapeCastOp = rewriter.create<VPUIP::ShapeCastOp>(swKernelOp->getLoc(), newClusterTilingOp.getResult(0),
                                                             getIntArrayAttr(ctx, outShape));
    } else {
        outShapeCastOp = rewriter.create<VPUIP::ShapeCastOp>(swKernelOp->getLoc(), newClusterTilingOp.getResult(0),
                                                             getIntArrayAttr(ctx, outShape), prevPerClusterShapesAttr,
                                                             prevPerClusterOffsetsAttr);
    }

    return outShapeCastOp;
}

bool ClusterSwKernelRewriter::checkTilePattern(VPUIP::SwKernelOp swKernelOp, bool insertSubview) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (clusterTilingOp == nullptr) {
        return false;
    }

    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedType == nullptr) {
        return false;
    }

    auto parentInputDistType = clusterTilingOp->getOperand(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (parentInputDistType == nullptr) {
        return false;
    }

    auto parentOutputDistType = distributedType;
    const auto inputType = swKernelOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    if (!checkSwKernelTilingAlignment(swKernelOp, inputType, parentInputDistType, _log) ||
        !checkSwKernelTilingAlignment(swKernelOp, outputType, parentOutputDistType, _log)) {
        return false;
    }

    auto tileDim = getSwKernelTileDim(swKernelOp);
    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    auto tileOnAllClusters = llvm::all_of(perClusterShapes, [&](const auto& shape) {
        return shape[tileDim] > 1;
    });
    if (!tileOnAllClusters) {
        return false;
    }

    if (insertSubview) {
        return true;
    }

    // Calculate requried cmx size since the input cmx size may be changed due to overlapped input tiles like
    // Interpolate
    auto allInTiles = calculateInputTiles(swKernelOp).value();
    Byte requiredCMX = distributedType.getTotalAllocSize();
    const auto outTiles = getOuterMostOutputTiling(swKernelOp);
    auto inputs = getOuterMappingOperand(swKernelOp, swKernelOp.getInputs());
    for (auto outIndex : irange(outTiles.size())) {
        const auto inTiles = getOuterMostInputTiling(swKernelOp, outIndex);
        for (const auto& item : inputs | indexed) {
            auto input = item.value();
            auto index = item.index();
            auto tiledShape = inTiles.tiles[index].shape;
            auto getTileInfo = [&](int64_t clusterId, int64_t shaveId, int64_t numClusters, VPU::DistributionMode mode,
                                   bool insertSubview) {
                return getTileFromList(allInTiles, clusterId, shaveId, numClusters, mode,
                                       insertSubview || tileOnDifferentDims(swKernelOp))
                        .tiles[index];
            };
            auto newTiledInputDistributedType =
                    getNewTiledDistributedType(swKernelOp, input, outIndex, tiledShape, getTileInfo);
            requiredCMX += newTiledInputDistributedType.getTotalAllocSize();
        }
    }
    return requiredCMX <= VPU::getTotalCMXSize(swKernelOp);
}

bool ClusterSwKernelRewriter::needInsertSubviewOnly(VPUIP::SwKernelOp swKernelOp) const {
    auto clusterOp = swKernelOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    VPUX_THROW_WHEN(clusterOp == nullptr, "Can't get NCEClusterTilingOp");

    auto isOverlapped = [&](mlir::Value val) {
        auto valueType = val.getType();
        auto distributedType = valueType.dyn_cast<VPUIP::DistributedBufferType>();
        VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported type {0}", valueType);

        auto distribution = distributedType.getDistribution();
        auto distributionMode = distribution.getMode().getValue();

        return distributionMode == VPU::DistributionMode::OVERLAPPED;
    };

    auto hasOverlappedInput = llvm::any_of(clusterOp.getInputs(), isOverlapped);
    auto hasOverlappedOutput = llvm::any_of(clusterOp.getOutputs(), isOverlapped);

    if (hasOverlappedInput || hasOverlappedOutput) {
        return false;
    }

    auto tileDim = getSwKernelTileDim(swKernelOp);
    if (!hasOnlyOneOffset(swKernelOp, tileDim)) {
        return false;
    }

    return SwKernelRewriterBase::needInsertSubviewOnly(swKernelOp);
}

std::optional<OutputTiling> ClusterSwKernelRewriter::calculateOutputTiles(VPUIP::SwKernelOp swKernelOp) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (clusterTilingOp == nullptr) {
        return std::nullopt;
    }
    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto perClusterShapes = distributedType.getPerClusterComputeShapes();

    const auto insertSubviewOnly = needInsertSubviewOnly(swKernelOp);

    // Get output tiles on each cluster
    SmallVector<mlir::FailureOr<OutputTiling>> tiles;
    std::transform(perClusterShapes.begin(), perClusterShapes.end(), std::back_inserter(tiles), [&](const auto& shape) {
        return getSwKernelOutputTiling(swKernelOp, shape, _shaveCount, insertSubviewOnly, _log);
    });
    if (tiles.empty()) {
        return std::nullopt;
    }

    auto hasInvalidTiles = llvm::any_of(tiles, [&](const auto& tile) {
        return mlir::failed(tile);
    });
    if (hasInvalidTiles) {
        return std::nullopt;
    }

    SmallVector<OutputTiling> outTiles;
    for (auto& tile : tiles) {
        outTiles.push_back(tile.value());
    }

    // For each cluster, the output tile size should be equal and greater than one
    int64_t tileSize = outTiles[0].size();
    auto findNoSuitableTileSizeOnClusters = llvm::any_of(outTiles, [&](const auto& tile) {
        return tile.size() != static_cast<size_t>(tileSize) || tile.size() <= 1;
    });
    if (findNoSuitableTileSizeOnClusters) {
        return std::nullopt;
    }
    const auto tileDim = getSwKernelTileDim(swKernelOp);
    const auto needAdjustTileSize = llvm::any_of(outTiles, [&](const OutputTiling& outTile) {
        for (auto i : irange(outTile.size() - 1)) {
            if (outTile[i].shape[tileDim] != outTiles.front()[i].shape[tileDim]) {
                return true;
            }
        }
        return false;
    });

    if (insertSubviewOnly && needAdjustTileSize) {
        // Need to adjust the tiling size due to aligment requriement, otherwise the compiler can not get required
        // distributed buffer by subview since the offsets on each cluster are not same.
        // For example shape [1, 33, 1, 1] tiled on C. So the tiled shape could be
        // cluster 0 [1, 9, 1, 1], [1, 8, 1, 1]
        // cluster 1 [1, 8, 1, 1], [1, 8, 1, 1]
        // we can't represent the second distributed buffer {cluster 0[1, 8, 1, 1], cluster 1[1, 8, 1, 1]} since the
        // offset on each cluster are different(cluster 0 offset = 9, cluster 1 offset = 8). So we need adjust the
        // tile size to make sure the offsets are equal for each cluster. The logic is to find the largest tile
        // value and change all the tiles' value equal to it except the last one. If there is no remaining for the last
        // one, change all the tiles' value equal to the smallest tile except the last one. In this case, the tiles are
        // changed to cluster 0 [1, 9, 1, 1], [1, 8, 1, 1] cluster 1 [1, 9, 1, 1], [1, 7, 1, 1].
        // The advantage of choosing the largest tile first is to reduce the highest workload of all the clusters.
        // e.g., split [1, 41, 1, 1] on 6 clusters.
        //      aligning to the smallest is [1, 6, 1, 1] x 5 and [1, 11, 1, 1] x 1
        //      while aligning to the largest it's [1, 7, 1, 1] x 5 and [1, 6, 1, 1] x 1
        //      The slowest cluster is the bottleneck
        auto compareMin = [&](ShapeRef a, ShapeRef b) {
            return a[tileDim] < b[tileDim];
        };
        auto getMaxOutTile = [&]() {
            auto iter = std::max_element(perClusterShapes.begin(), perClusterShapes.end(), compareMin);
            VPUX_THROW_WHEN(iter == perClusterShapes.end(), "Can't find the element in perClusterShapes");
            auto index = std::distance(perClusterShapes.begin(), iter);
            return outTiles[index];
        };
        auto getMinOutTile = [&]() {
            auto iter = std::min_element(perClusterShapes.begin(), perClusterShapes.end(), compareMin);
            VPUX_THROW_WHEN(iter == perClusterShapes.end(), "Can't find the element in perClusterShapes");
            auto index = std::distance(perClusterShapes.begin(), iter);
            return outTiles[index];
        };
        const auto& maxOutTile = getMaxOutTile();
        auto lastTileEnoughToAlign = [&](const OutputTiling& alignOutTile) {
            for (const auto& clusterId : irange(outTiles.size())) {
                int64_t usedSize = 0;
                for (auto i : irange(tileSize - 1)) {
                    usedSize += alignOutTile[i].shape[tileDim];
                }
                if (perClusterShapes[clusterId][tileDim] <= usedSize) {
                    return false;
                }
            }
            return true;
        };
        const auto& alignOutTile = lastTileEnoughToAlign(maxOutTile) ? maxOutTile : getMinOutTile();

        // Adjust the front tiles with same tile value
        for (auto item : outTiles | indexed) {
            const auto& clusterId = item.index();
            auto& outTilePerCluster = item.value();
            int64_t usedSize = 0;
            for (auto i : irange(tileSize - 1)) {
                outTilePerCluster[i].shape[tileDim] = alignOutTile[i].shape[tileDim];
                outTilePerCluster[i].offsets[tileDim] = alignOutTile[i].offsets[tileDim];
                usedSize += outTilePerCluster[i].shape[tileDim];
            }
            // Recalculate the last tile value
            Shape lastTileShape(outTilePerCluster.front().shape);
            lastTileShape[tileDim] = perClusterShapes[clusterId][tileDim] - usedSize;
            Shape lastTileOffset = alignOutTile.back().offsets;
            outTilePerCluster[tileSize - 1] = TileInfo(lastTileShape, lastTileOffset, outTilePerCluster.front().axis);
        }
    }

    // Convert tiles on each cluster to tiles on full output
    // e.g. for output [1, 48, 16, 16] with CL=2, mode=SEGMENTED, alignment=16 tile on DimC
    //      Shape          Offset
    // CL0: [1, 32, 16, 16], [0, 0, 0, 0]
    // CL1: [1, 16, 16, 16], [0, 32, 0, 0]
    // if the multi-shave tiling is still on DimC, the tiles on CL0 could be
    //       Shape          Offset
    // Tile0 [1, 16, 16, 16], [0, 0, 0, 0]
    // Tile1 [1, 16, 16, 16], [0, 16, 0, 0]
    // And the tiles on CL1 could be
    //       Shape          Offset
    // Tile2 [1, 8, 16, 16], [0, 0, 0, 0]
    // Tile3 [1, 8, 16, 16], [0, 8, 0, 0]
    // Note that the alignment=16 is supposed to be removed since the new tiled shape doesn't meet the alignment.
    // And the tiles' offset over the full output could be calculated by adding the per cluster offset
    //       Shape          Offset
    // Tile0 [1, 16, 16, 16], [0, 0, 0, 0]
    // Tile1 [1, 16, 16, 16], [0, 16, 0, 0]
    // Tile2 [1, 8, 16, 16], [0, 32, 0, 0]
    // Tile3 [1, 8, 16, 16], [0, 40, 0, 0]
    auto perClusterOffsets = distributedType.getPerClusterComputeShapeOffsets();
    auto mode = distributedType.getDistribution().getMode().getValue();

    OutputTiling globalOutTiles;
    for (auto clusterId : irange(outTiles.size())) {
        auto baseOutOffset = to_small_vector(perClusterOffsets[clusterId]);
        for (auto& tile : outTiles[clusterId]) {
            // Adjust the offset against the original output
            auto offset = to_small_vector(tile.offsets);
            SmallVector<int64_t> adjustedOffset;
            std::transform(offset.begin(), offset.end(), baseOutOffset.begin(), std::back_inserter(adjustedOffset),
                           std::plus<int64_t>());
            tile.offsets = Shape(adjustedOffset);
            globalOutTiles.push_back(tile);
        }
        if (mode == VPU::DistributionMode::DUPLICATED) {
            break;
        }
    }

    // Global tiles may have unbalanced data size on each cluster, which will cause out of CMX memory issue on some
    // clusters. E.g. for output [1, 6, 1000, 1000] with CL=2, mode=SEGMENTED, tile on DimC. For tiles [2, 1, 2, 1],
    // the data will copy to DDR first, then copy back to CMX, so the data will be like this:
    //       SHV0         SHV1
    // CL0: [1, 2, 1000], [1, 2, 1000]
    // CL1: [1, 1, 1000], [1, 1, 1000]

    // CL0 allocs more buffer size than CL1. So we need to adjust the tile shape size to be not greater than
    // orignal size[1, 3, 1000, 100]. SO tiles size and offset after adjustment:
    //       SHV0         SHV1
    // CL0: [1, 2, 1000], [1, 1, 1000]
    // CL1: [1, 2, 1000], [1, 1, 1000]
    const auto numTiles = distributedType.getDistribution().getNumClusters().getInt();
    auto needAdjustGlobalTileSize = [&]() {
        if (insertSubviewOnly || mode == VPU::DistributionMode::DUPLICATED) {
            return false;
        }
        const auto maxSize = distributedType.getLargestCompactShape()[tileDim];
        for (auto clusterId : irange(numTiles)) {
            int64_t sizeOnTileDim = 0;
            for (auto shaveId : irange(tileSize)) {
                sizeOnTileDim += getTileFromList(globalOutTiles, clusterId, shaveId, numTiles, mode,
                                                 insertSubviewOnly || tileOnDifferentDims(swKernelOp))
                                         .shape[tileDim];
            }
            if (sizeOnTileDim > maxSize) {
                return true;
            }
        }
        return false;
    };

    if (needAdjustGlobalTileSize()) {
        const auto distributionAttr = distributedType.getDistribution();
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
        const auto numDim = distributedType.getDimsOrder().numDims();
        auto isTilingOnSameDim = Dim(axis) == tileDim;
        globalOutTiles.clear();
        for (auto tileId : irange(tileSize)) {
            for (auto clusterId : irange(outTiles.size())) {
                auto tile = outTiles[clusterId][tileId];
                // Adjust the offset against the original output
                auto adjustedOffset = SmallVector<int64_t>(numDim, 0);
                if (isTilingOnSameDim) {
                    // In this case, shave tile dim is same as cluster tiling dim, new offset is sum of pre tiles'
                    // shape size on tile dim. E.g.
                    //       Shape          Offset
                    //  CL0: [1, 32, 3, 16], [0, 0, 0, 0]
                    //  CL1: [1, 32, 3, 16], [0, 0, 3, 0]
                    //
                    //  original global tile:
                    //         Shape           Offset
                    //  Tile0(CL0, SHV0): [1, 32, 2, 16], [0, 0, 0, 0]
                    //  Tile1(CL1, SHV0): [1, 32, 1, 16], [0, 0, 2, 0]
                    //  Tile2(CL0, SHV1): [1, 32, 2, 16], [0, 0, 3, 0]
                    //  Tile3(CL1, SHV1): [1, 32, 1, 16], [0, 0, 5, 0]
                    //
                    //  adjusted global tile:
                    //          Shape          Offset
                    //  Tile0(CL0, SHV0): [1, 32, 2, 16], [0, 0, 0, 0]
                    //  Tile1(CL1, SHV0): [1, 32, 2, 16], [0, 0, 2, 0]
                    //  Tile2(CL0, SHV1): [1, 32, 1, 16], [0, 0, 4, 0]
                    //  Tile3(CL1, SHV1): [1, 32, 1, 16], [0, 0, 5, 0]

                    for (auto& preTile : globalOutTiles) {
                        adjustedOffset[axis] += preTile.shape[Dim(axis)];
                    }
                } else {
                    // In this case, shave tile dim is different with cluster tiling dim,  new offset is sum of dim
                    // size of pre tiles on same cluster. E.g.
                    //   Shape          Offset
                    //  CL0: [1, 16, 3, 16], [0, 0, 0, 0]
                    //  CL1: [1, 15, 3, 16], [0, 16, 0, 0]
                    //
                    //  original global tile:
                    //         Shape          Offset
                    //  Tile0(CL0, SHV0): [1, 16, 2, 16], [0, 0, 0, 0]
                    //  Tile1(CL1, SHV0): [1, 16, 1, 16], [0, 0, 2, 0]
                    //  Tile2(CL0, SHV1): [1, 15, 2, 16], [0, 16, 0, 0]
                    //  Tile3(CL1, SHV1): [1, 15, 1, 16], [0, 16, 2, 0]
                    //
                    //  adjusted global tile:
                    //         Shape          Offset
                    //  Tile0(CL0, SHV0): [1, 16, 2, 16], [0, 0, 0, 0]
                    //  Tile1(CL1, SHV0): [1, 15, 2, 16], [0, 16, 0, 0]
                    //  Tile2(CL0, SHV1): [1, 16, 1, 16], [0, 0, 2, 0]
                    //  Tile3(CL1, SHV1): [1, 15, 1, 16], [0, 16, 2, 0]

                    adjustedOffset = to_small_vector(perClusterOffsets[clusterId]);
                    for (auto i : irange(tileId)) {
                        adjustedOffset[tileDim.ind()] +=
                                getTileFromList(globalOutTiles, clusterId, i, numTiles, mode,
                                                insertSubviewOnly || tileOnDifferentDims(swKernelOp))
                                        .shape[tileDim];
                    }
                }
                tile.offsets = Shape(adjustedOffset);
                globalOutTiles.push_back(tile);
            }
        }
    }
    return globalOutTiles;
}

std::optional<SmallVector<InputTiling>> ClusterSwKernelRewriter::calculateInputTiles(
        VPUIP::SwKernelOp swKernelOp) const {
    const auto outTiles = calculateOutputTiles(swKernelOp);
    if (!outTiles.has_value()) {
        return std::nullopt;
    }
    auto outTilesValue = outTiles.value();
    SmallVector<InputTiling> inTiles;
    for (int i = 0; i < static_cast<int>(outTilesValue.size()); i++) {
        inTiles.push_back(VPUIP::backInferSwKernelInputTile(swKernelOp, outTilesValue, i, _log));
    }
    return inTiles;
}

size_t ClusterSwKernelRewriter::getShaveTileSize(VPUIP::SwKernelOp swKernelOp, const OutputTiling& outTiles) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto mode = distributedType.getDistribution().getMode().getValue();
    if (mode == VPU::DistributionMode::DUPLICATED) {
        return outTiles.size();
    }
    auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    VPUX_THROW_UNLESS(outTiles.size() % numClusters == 0, "Invalid tile size {0}", outTiles.size());
    return outTiles.size() / numClusters;
}

SmallVector<mlir::Value> ClusterSwKernelRewriter::createNewInputs(VPUIP::SwKernelOp swKernelOp,
                                                                  mlir::ValueRange operands, bool insertSubview,
                                                                  int64_t outTileIndex,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto inTiles = getOuterMostInputTiling(swKernelOp, outTileIndex);
    SmallVector<mlir::Value> newInputs;
    VPUX_THROW_UNLESS(operands.size() == inTiles.tiles.size(), " operand size is not equal to tile size");

    // if the operand comes from TilingCopy(DDR2CMX), get the op's input
    auto getSourceBufferFromDDR = [](mlir::Value operand) -> mlir::Value {
        auto sourceOp = operand.getDefiningOp<VPUIP::NCEClusterTilingOp>();
        if (sourceOp == nullptr) {
            return nullptr;
        }
        auto innerCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(sourceOp.getInnerTaskOp());
        if (innerCopyOp == nullptr) {
            return nullptr;
        }
        VPUX_THROW_UNLESS(VPUIP::isCopyFromDDR(innerCopyOp), "Tiling Copy is supposed to be from DDR at '{0}'",
                          sourceOp->getLoc());
        return sourceOp.getInputs()[0];
    };

    auto allInTiles = calculateInputTiles(swKernelOp).value();
    for (const auto& p : operands | indexed) {
        const auto& index = p.index();
        const auto& operand = p.value();
        const auto& offset = inTiles.tiles[index].offsets;
        const auto& tiledShape = inTiles.tiles[index].shape;

        // handle swkernel's input copy
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfterValue(operand);

        auto getTileInfo = [&](int64_t clusterId, int64_t shaveId, int64_t numClusters, VPU::DistributionMode mode,
                               bool insertSubview) {
            return getTileFromList(allInTiles, clusterId, shaveId, numClusters, mode,
                                   insertSubview || tileOnDifferentDims(swKernelOp))
                    .tiles[index];
        };
        auto newDistributedType =
                getNewTiledDistributedType(swKernelOp, operand, outTileIndex, tiledShape, getTileInfo);

        if (insertSubview) {
            auto inputSubview = createSubViewOpWithDistributedOutput(rewriter, operand.getLoc(), newDistributedType,
                                                                     operand, offset);
            newInputs.push_back(inputSubview);
        } else {
            auto sourceBuffer = getSourceBufferFromDDR(operand);
            if (sourceBuffer == nullptr) {
                // Since the compiler doesn't support copy from DistributedBufferType to DistributedBufferType,
                // input data need copy to DDR then copy back to CMX
                auto origType = swKernelOp.getInputs()[index].getType().dyn_cast<vpux::NDTypeInterface>();
                auto newDDRType = origType.changeMemSpace(VPU::MemoryKind::DDR);
                auto newAllocDDROp =
                        rewriter.create<mlir::memref::AllocOp>(operand.getLoc(), newDDRType.cast<mlir::MemRefType>());
                auto tilingCopyBackToDDROp =
                        createNewTilingCopyOp(rewriter, operand.getLoc(), newDDRType, {operand, newAllocDDROp});
                sourceBuffer = tilingCopyBackToDDROp->getResult(0);
            }

            auto inputSubview = rewriter.create<VPUIP::SubViewOp>(operand.getLoc(), sourceBuffer, offset, tiledShape);
            auto newAllocCMXOp =
                    rewriter.create<VPURT::AllocDistributed>(operand.getLoc(), newDistributedType, nullptr, nullptr);
            auto newTilingCopyToCMXOp = createNewTilingCopyOp(rewriter, operand.getLoc(), newDistributedType,
                                                              {inputSubview, newAllocCMXOp});
            newInputs.push_back(newTilingCopyToCMXOp->getResult(0));
        }
    }
    return newInputs;
}

SmallVector<mlir::Value> ClusterSwKernelRewriter::createNewOutBuffs(VPUIP::SwKernelOp swKernelOp,
                                                                    mlir::ValueRange outBuffs, bool insertSubview,
                                                                    int64_t outTileIndex,
                                                                    mlir::PatternRewriter& rewriter) const {
    const auto perClusterFirstOutputTiles = getOuterMostOutputTiling(swKernelOp);
    const auto perShaveFirstOutputTiles = calculateOutputTiles(swKernelOp).value();

    const auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    const auto outputTilesOnCluster = computeOutputTiling(kernelEntryName, perClusterFirstOutputTiles[outTileIndex]);
    VPUX_THROW_UNLESS(outBuffs.size() == outputTilesOnCluster.size(),
                      "Number of output buffers '{0}' must be equal to the number of outputs of a tiled operation on a "
                      "cluster '{1}'",
                      outBuffs.size(), outputTilesOnCluster.size());

    auto outputTilesOnShaves = SmallVector<OutputTiling>();
    for (const auto& onShaveFirstOutputTile : perShaveFirstOutputTiles) {
        outputTilesOnShaves.push_back(computeOutputTiling(kernelEntryName, onShaveFirstOutputTile));
    }

    SmallVector<mlir::Value> newOutputs;
    for (int outputId = 0; outputId < static_cast<int>(outBuffs.size()); outputId++) {
        const auto& tiledShape = outputTilesOnCluster[outputId].shape;
        const auto& offset = outputTilesOnCluster[outputId].offsets;

        // handle swkernel's output buf
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfterValue(outBuffs[outputId]);

        auto allocType = outBuffs[outputId].getType().cast<VPUIP::DistributedBufferType>();

        auto mode = allocType.getDistribution().getMode().getValue();
        VPUX_THROW_WHEN(mode == VPU::DistributionMode::OVERLAPPED,
                        "Unsupported output OVERLAPPED distribution for act shv tiling");

        auto getTileInfo = [&](int64_t clusterId, int64_t shaveId, int64_t numClusters, VPU::DistributionMode mode,
                               bool insertSubview) {
            const auto oneKernelOutputTiles = getTileFromList(outputTilesOnShaves, clusterId, shaveId, numClusters,
                                                              mode, insertSubview || tileOnDifferentDims(swKernelOp));
            return oneKernelOutputTiles[outputId];
        };

        auto newAllocType =
                getNewTiledDistributedType(swKernelOp, outBuffs[outputId], outTileIndex, tiledShape, getTileInfo);
        if (insertSubview) {
            auto newDistributedType = newAllocType.cast<VPUIP::DistributedBufferType>();
            auto outputSubview = createSubViewOpWithDistributedOutput(rewriter, outBuffs[outputId].getLoc(),
                                                                      newDistributedType, outBuffs[outputId], offset);
            newOutputs.push_back(outputSubview);
        } else {
            auto newOutputAllocType = rewriter.create<VPURT::AllocDistributed>(outBuffs[outputId].getLoc(),
                                                                               newAllocType, nullptr, nullptr);
            newOutputs.push_back(newOutputAllocType);
        }
    }

    return newOutputs;
}

VPUIP::SwKernelOp ClusterSwKernelRewriter::createNewSwKernelOp(VPUIP::SwKernelOp swKernelOp,
                                                               ArrayRef<mlir::Value> newInputs,
                                                               ArrayRef<mlir::Value> newOutBufs, bool insertSubview,
                                                               mlir::PatternRewriter& rewriter) const {
    auto swKernelRun = *swKernelOp.getBody().getOps<VPUIP::SwKernelRun>().begin();
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    rewriter.setInsertionPointAfter(clusterTilingOp);
    mlir::ArrayAttr strideAttr = getStrideOnEachCluster(swKernelOp, insertSubview);
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange operands) {
        SmallVector<mlir::Value> inputs(operands.begin(), operands.begin() + newInputs.size());
        SmallVector<mlir::Value> outputs(operands.begin() + newInputs.size(), operands.end());
        auto newSwKernelTask = builder.create<VPUIP::SwKernelOp>(loc, inputs, outputs, swKernelOp.getKernelFunction(),
                                                                 swKernelOp.getTileIndexAttr(), strideAttr);
        VPUIP::initSwKernel(newSwKernelTask, swKernelRun, _log);
    };

    SmallVector<mlir::Value> newOperands;
    newOperands.append(newInputs.begin(), newInputs.end());
    newOperands.append(newOutBufs.begin(), newOutBufs.end());

    SmallVector<mlir::Type> resultTypes;
    for (auto& outBuf : newOutBufs) {
        resultTypes.push_back(outBuf.getType());
    }
    auto newSwKernelTask =
            rewriter.create<VPUIP::NCEClusterTilingOp>(swKernelOp->getLoc(), resultTypes, newOperands, bodyBuilder);
    _log.trace("create new cluster shave {0}", newSwKernelTask);

    return mlir::dyn_cast<VPUIP::SwKernelOp>(newSwKernelTask.getInnerTaskOp());
}

void ClusterSwKernelRewriter::replaceOpWithConcatView(VPUIP::SwKernelOp origOp, VPUIP::SwKernelOp newSwKernelOp,
                                                      bool insertSubview, mlir::PatternRewriter& rewriter) const {
    auto origClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(origOp->getParentOp());
    auto newClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(newSwKernelOp->getParentOp());
    // Get input ops
    SmallVector<mlir::Operation*> inputOps;
    for (const auto& input : origClusterTilingOp.getInputs()) {
        if (const auto& inputOp = input.getDefiningOp()) {
            inputOps.push_back(inputOp);
        }
    }

    const auto origClusterTilingResults = origClusterTilingOp.getResults();
    const auto resultsNum = origClusterTilingResults.size();
    if (insertSubview) {
        llvm::SmallVector<mlir::Value> newConcats;
        for (auto p : origClusterTilingResults | indexed) {
            const auto index = p.index();
            const auto newResults = newClusterTilingOp->getResults();
            auto concatInputs = llvm::SmallVector<mlir::Value>{newResults[index], newResults[resultsNum + index]};
            auto outBufOp = origClusterTilingOp.getOutputBuffs()[index].getDefiningOp();
            auto concatOp = rewriter.create<VPUIP::ConcatViewOp>(newClusterTilingOp->getLoc(), concatInputs,
                                                                 outBufOp->getResult(0));
            newConcats.push_back(concatOp.getResult());
        }
        rewriter.replaceOp(origClusterTilingOp, mlir::ValueRange{newConcats});
        return;
    }

    auto outTiles = getOuterMostOutputTiling(origOp);
    const auto hasCopyUser = onlyHasCopyOpUser(origOp);
    mlir::DenseMap<int64_t, mlir::memref::AllocOp> newAllocDDROpsMap;
    if (hasCopyUser) {
        for (auto user : origClusterTilingOp->getUsers()) {
            if (auto userCopyOp = mlir::cast<VPUIP::NCEClusterTilingOp>(*user)) {
                rewriter.setInsertionPointAfter(userCopyOp);
                auto newAllocDDROp =
                        mlir::cast<mlir::memref::AllocOp>(userCopyOp.getOutputBuffs().front().getDefiningOp());
                auto operandIt = std::find(origClusterTilingResults.begin(), origClusterTilingResults.end(),
                                           userCopyOp.getOperand(0));
                if (operandIt != origClusterTilingResults.end()) {
                    newAllocDDROpsMap[operandIt - origClusterTilingResults.begin()] = newAllocDDROp;
                }
            }
        }
    } else {
        rewriter.setInsertionPointAfter(newClusterTilingOp);
        for (auto result : origOp.getResults() | indexed) {
            auto newDDRType =
                    result.value().getType().cast<vpux::NDTypeInterface>().changeMemSpace(VPU::MemoryKind::DDR);
            auto newAllocDDROp = rewriter.create<mlir::memref::AllocOp>(newClusterTilingOp->getLoc(),
                                                                        newDDRType.cast<mlir::MemRefType>());
            newAllocDDROpsMap[result.index()] = newAllocDDROp;
        }
    }

    mlir::DenseMap<int64_t, mlir::Value> resultMap;
    for (const auto& item : outTiles | indexed) {
        const auto& outTile = item.value();
        const auto& index = item.index();
        auto outShape = to_small_vector(outTile.shape);
        auto outOffset = to_small_vector(outTile.offsets);

        for (auto p : origClusterTilingResults | indexed) {
            const auto result = p.value();
            const auto resultIdx = p.index();
            if (!result.getUsers().empty()) {
                auto it = newAllocDDROpsMap.find(resultIdx);
                auto outSubview = rewriter.create<VPUIP::SubViewOp>(newClusterTilingOp->getLoc(), it->second, outOffset,
                                                                    outShape);
                auto copyOp = createNewTilingCopyOp(
                        rewriter, newClusterTilingOp->getLoc(), outSubview.getType(),
                        {newClusterTilingOp.getResult(checked_cast<unsigned int>(index * resultsNum + resultIdx)),
                         outSubview});
                resultMap[index * resultsNum + resultIdx] = copyOp->getResult(0);
            }
        }
    }

    llvm::SmallVector<mlir::Value> newTilingCopys;
    for (auto p : origClusterTilingResults | indexed) {
        const auto index = p.index();
        const auto value = p.value();

        auto concatInputs = llvm::SmallVector<mlir::Value>{resultMap[index], resultMap[resultsNum + index]};
        if (hasCopyUser) {
            for (auto user : llvm::make_early_inc_range(value.getUsers())) {
                if (auto userCopyOp = mlir::cast<VPUIP::NCEClusterTilingOp>(*user)) {
                    auto operandIt = std::find(origClusterTilingResults.begin(), origClusterTilingResults.end(),
                                               userCopyOp.getOperand(0));
                    auto it = newAllocDDROpsMap.find(operandIt - origClusterTilingResults.begin());
                    rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(userCopyOp, concatInputs, it->second);
                }
            }
            if (origClusterTilingOp != nullptr && origClusterTilingOp->use_empty()) {
                rewriter.eraseOp(origClusterTilingOp);
            }
        } else {
            // result not be used
            if (value.use_empty()) {
                newTilingCopys.push_back(nullptr);
                continue;
            }

            auto it = newAllocDDROpsMap.find(index);
            auto concatOp =
                    rewriter.create<VPUIP::ConcatViewOp>(newClusterTilingOp->getLoc(), concatInputs, it->second);
            auto outType = origClusterTilingOp->getResult(checked_cast<unsigned int>(index))
                                   .getType()
                                   .cast<vpux::NDTypeInterface>();
            auto newAllocCMXOp =
                    rewriter.create<VPURT::AllocDistributed>(origClusterTilingOp->getLoc(), outType, nullptr, nullptr);

            auto newTilingCopyToCMXOp =
                    createNewTilingCopyOp(rewriter, newClusterTilingOp->getLoc(), outType, {concatOp, newAllocCMXOp});
            newTilingCopys.push_back(newTilingCopyToCMXOp.getResult(0));
        }
    }

    if (!newTilingCopys.empty()) {
        rewriter.replaceOp(origClusterTilingOp, mlir::ValueRange{newTilingCopys});
    }

    std::set<mlir::Operation*> uniqueInputSet(inputOps.begin(), inputOps.end());
    for (auto originInputOp : uniqueInputSet) {
        if (originInputOp != nullptr && originInputOp->use_empty()) {
            rewriter.eraseOp(originInputOp);
        }
    }
}

OutputTiling ClusterSwKernelRewriter::getOuterMostOutputTiling(VPUIP::SwKernelOp swKernelOp) const {
    auto outTiles = calculateOutputTiles(swKernelOp).value();

    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    VPUX_THROW_WHEN(clusterTilingOp == nullptr, "Unexpected parent op type at '{0}'", swKernelOp->getLoc());
    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();

    auto mode = distributedType.getDistribution().getMode().getValue();
    if (mode == VPU::DistributionMode::DUPLICATED) {
        return outTiles;
    }

    const auto numDim = distributedType.getDimsOrder().numDims();
    const auto numTiles = distributedType.getDistribution().getNumClusters().getInt();
    const auto insertSubview = needInsertSubviewOnly(swKernelOp);
    auto shaveTileDim = getSwKernelTileDim(swKernelOp);
    auto shaveTileSize = getShaveTileSize(swKernelOp, outTiles);
    auto clusterTileDim = shaveTileDim;
    auto dimIdx = VPUIP::getTilingDimIndex(distributedType);
    if (dimIdx.has_value()) {
        clusterTileDim = Dim(dimIdx.value());
    }

    auto getOuterMostShapeValueOnTiledDim = [&](int64_t idx) {
        int64_t tiledDimShapeValue = 0;
        for (auto clusterId : irange(numTiles)) {
            auto outTile = getTileFromList(outTiles, clusterId, idx, numTiles, mode,
                                           insertSubview || tileOnDifferentDims(swKernelOp));
            tiledDimShapeValue += outTile.shape[clusterTileDim];
        }
        return tiledDimShapeValue;
    };

    OutputTiling outputTiles;
    int64_t offset = 0;

    // Multi-SHAVEs tiling splits tensor on shaveTileDim, offset & axis dim are always on shaveTileDim for current
    // SHAVE's tile info
    const auto offsetDim = shaveTileDim;
    const auto axisDim = shaveTileDim;
    for (auto outTileIndex : irange(shaveTileSize)) {
        // Get outer tile shape
        Shape shape = getTileFromList(outTiles, 0, outTileIndex, numTiles, mode,
                                      insertSubview || tileOnDifferentDims(swKernelOp))
                              .shape;
        // Multi-Cluster tiling splits tensor on clusterTileDim, accumulate dim size on clusterTileDim for current
        // SHAVE's tile info
        shape[clusterTileDim] = getOuterMostShapeValueOnTiledDim(outTileIndex);

        // Get outer tile offset
        Shape offsets(numDim, 0);
        offsets[offsetDim] = offset;
        offset += shape[offsetDim];

        Shape axis(numDim, 1);
        axis[axisDim] = shaveTileSize;
        outputTiles.push_back(TileInfo(shape, offsets, axis));
    }

    return outputTiles;
}

InputTiling ClusterSwKernelRewriter::getOuterMostInputTiling(VPUIP::SwKernelOp swKernelOp, int64_t outTileIdx) const {
    auto outTiles = getOuterMostOutputTiling(swKernelOp);
    return VPUIP::backInferSwKernelInputTile(swKernelOp, outTiles, outTileIdx, _log);
}

bool ClusterSwKernelRewriter::onlyHasCopyOpUser(VPUIP::SwKernelOp swKernelOp) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (!clusterTilingOp->hasOneUse()) {
        return false;
    }
    auto userCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*clusterTilingOp->getUsers().begin());
    return userCopyOp != nullptr && mlir::isa<VPUIP::CopyOp>(userCopyOp.getInnerTaskOp());
}

bool ClusterSwKernelRewriter::tileOnDifferentDims(VPUIP::SwKernelOp swKernelOp) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto mode = distributedType.getDistribution().getMode().getValue();
    if (mode == VPU::DistributionMode::DUPLICATED) {
        return false;
    }

    auto shaveTileDim = getSwKernelTileDim(swKernelOp);
    auto dimIdx = VPUIP::getTilingDimIndex(distributedType);
    return shaveTileDim != Dim(dimIdx.value());
}

template <class TileClass>
TileClass ClusterSwKernelRewriter::getTileFromList(const SmallVector<TileClass>& tiles, int64_t clusterId,
                                                   int64_t shaveId, int64_t numTiles, VPU::DistributionMode mode,
                                                   bool insertSubview) const {
    auto getTileIndex = [&]() {
        if (mode == VPU::DistributionMode::DUPLICATED) {
            return shaveId;
        }
        const int64_t shaveTileSize = tiles.size() / numTiles;
        /*
         For the original entire tile list [Tile0, Tile1, Tile2, Tile3, Tile4, Tile5],
         if subview is used or MC & MS are tiling on the same dimension, the index distribution looks like:
                  SHV0       SHV1
             CL0  [Tile0     Tile1]
             CL1  [Tile2     Tile3]
             CL2  [Tile4     Tile5]
         if copy is used, the index distribution will be changed as:
                  SHV0       SHV1
             CL0  [Tile0     Tile3]
             CL1  [Tile1     Tile4]
             CL2  [Tile2     Tile5]
        */
        return insertSubview ? shaveTileSize * clusterId + shaveId : shaveId * numTiles + clusterId;
    };
    auto index = getTileIndex();
    VPUX_THROW_UNLESS(checked_cast<size_t>(index) < tiles.size(), "Tile index {0} is out of range", index);
    return tiles[index];
}

vpux::NDTypeInterface ClusterSwKernelRewriter::getNewTiledDistributedType(
        VPUIP::SwKernelOp swKernelOp, mlir::Value outerOperand, int64_t outTileIndex, ShapeRef tiledShape,
        std::function<TileInfo(int64_t clusterId, int64_t shaveId, int64_t numClusters, VPU::DistributionMode mode,
                               bool insertSubview)>
                getTileInfo) const {
    auto distributedType = outerOperand.getType().cast<VPUIP::DistributedBufferType>();
    auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();
    const auto insertSubview = needInsertSubviewOnly(swKernelOp);
    const auto numClusters = distributionAttr.getNumClusters().getInt();
    const auto dimSize = distributedType.getShape().size();

    // For the shave with id `outTileIndex`, need to calculate the related outer distributed buffer's compute/memory
    // shapes and offsets
    SmallVector<SmallVector<int64_t>> newTiledShape;
    SmallVector<SmallVector<int64_t>> newTiledOffset;
    for (auto clusterId : irange(numClusters)) {
        auto tile = getTileInfo(clusterId, outTileIndex, numClusters, mode, insertSubview);
        newTiledShape.push_back(to_small_vector(tile.shape));

        SmallVector<int64_t> adjustedOffset;
        if (mode == VPU::DistributionMode::DUPLICATED) {
            adjustedOffset = SmallVector<int64_t>(dimSize, 0);
        } else if (insertSubview) {
            // When subview is used to generate the tiled distributed type. the actual buffer on each cluster is not
            // overlapped with the others. So the compute offset can be infered from the previous tile shapes.
            const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
            const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
            adjustedOffset = SmallVector<int64_t>(dimSize, 0);
            for (auto preClusterId : irange(clusterId)) {
                auto preTileOnSameShave = getTileInfo(preClusterId, outTileIndex, numClusters, mode, insertSubview);
                adjustedOffset[axis] += preTileOnSameShave.shape[Dim(axis)];
            }
        } else {
            // In this case, CopyOp is used to generate the tiled distributed type. So the original buffer will copy
            // back to DDR first So the compute offset can be calculated by its tiling offset - the first tile's
            // tiling offset on cluster0 and shave `outTileIndex`
            auto currentOffset = to_small_vector(tile.offsets);
            auto firstTileOffset =
                    to_small_vector(getTileInfo(0, outTileIndex, numClusters, mode, insertSubview).offsets);
            std::transform(currentOffset.begin(), currentOffset.end(), firstTileOffset.begin(),
                           std::back_inserter(adjustedOffset), std::minus<int64_t>());
        }
        newTiledOffset.push_back(to_small_vector(adjustedOffset));
    }

    // return the distributed type without explicit shapes if it has correct per cluster shapes/offsets
    auto newTypeWithImplicitShapes =
            getImplicitDistributedType(swKernelOp, distributedType, tiledShape, newTiledShape, newTiledOffset);
    if (newTypeWithImplicitShapes.has_value()) {
        return newTypeWithImplicitShapes.value();
    }
    // In this case, the implicit type can not be used. So the new type is created with shape/offsets specified. For
    // example, C=128, numCluster=6, Alignment=16, then perClusterShape is [32, 32, 16, 16, 16, 16]. For sliceShape
    // C 80, with offset 48, perClusterShape should be [24, 24, 8, 8, 8] which can not be created by the orignal
    // dist attr ` {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1],
    // uniform_distributed_segments}'
    auto ctx = swKernelOp->getContext();
    auto shapesAttr = vpux::getIntArrayOfArray(ctx, newTiledShape);
    auto offsetsAttr = vpux::getIntArrayOfArray(ctx, newTiledOffset);
    auto newDistribution = VPU::DistributionInfoAttr::get(
            ctx, distributionAttr.getMode(), distributionAttr.getNumTiles(), distributionAttr.getKernel(),
            distributionAttr.getPads(), distributionAttr.getStrides(), distributionAttr.getNumClusters(),
            /*alignment*/ nullptr, distributionAttr.getUniformDistributedSegments(), shapesAttr, offsetsAttr,
            shapesAttr, offsetsAttr, nullptr);

    return VPUIP::DistributedBufferType::get(ctx, tiledShape.raw(), distributedType.getElementType(),
                                             distributedType.getLayout(), distributedType.getMemSpace(),
                                             newDistribution, distributedType.getSparsityCompression());
}

std::optional<vpux::NDTypeInterface> ClusterSwKernelRewriter::getImplicitDistributedType(
        VPUIP::SwKernelOp swKernelOp, VPUIP::DistributedBufferType srcDistributedType, ShapeRef newShape,
        ArrayRef<SmallVector<int64_t>> tiledShape, ArrayRef<SmallVector<int64_t>> tiledOffset) const {
    auto distributionAttr = srcDistributedType.getDistribution();
    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributionAttr)) {
        return std::nullopt;
    }
    // update subview alignment if needed
    auto ctx = swKernelOp->getContext();
    distributionAttr = VPU::updateSliceLikeOpsAlignment(ctx, srcDistributedType.getShape(), newShape, distributionAttr);

    const auto memoryShapes = VPU::getPerClusterMemoryShapes(newShape, distributionAttr);
    if (!memoryShapes.has_value()) {
        return std::nullopt;
    }
    const auto memoryOffsets = VPU::getPerClusterMemoryShapeOffsets(newShape, distributionAttr);
    auto hasSameShapeValue = [&](ArrayRef<Shape> implicitShapes, ArrayRef<SmallVector<int64_t>> expectedShapes) {
        if (implicitShapes.size() != expectedShapes.size()) {
            return false;
        }
        for (auto item : zip(implicitShapes, expectedShapes)) {
            auto& implicitShape = std::get<0>(item);
            auto expectedShape = Shape(std::get<1>(item));
            if (implicitShape != expectedShape) {
                return false;
            }
        }
        return true;
    };
    // If any memory shapes/offsets have same different value with the tiled shape/offsets, implicit type can not be
    // used
    if (!hasSameShapeValue(memoryShapes.value(), tiledShape) || !hasSameShapeValue(memoryOffsets, tiledOffset)) {
        return std::nullopt;
    }

    return VPUIP::DistributedBufferType::get(ctx, newShape.raw(), srcDistributedType.getElementType(),
                                             srcDistributedType.getLayout(), srcDistributedType.getMemSpace(),
                                             distributionAttr, srcDistributedType.getSparsityCompression());
}

mlir::ArrayAttr ClusterSwKernelRewriter::getStrideOnEachCluster(VPUIP::SwKernelOp swKernelOp,
                                                                bool insertSubview) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    VPUX_THROW_WHEN(clusterTilingOp == nullptr, "Unexpected parent op type at '{0}'", swKernelOp->getLoc());
    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto dimOrder = distributedType.getDimsOrder();
    mlir::ArrayAttr strideAttr = nullptr;
    SmallVector<SmallVector<int64_t>> strideOnPerClusters;
    if (insertSubview) {
        // If swkernel supports stride access, the operands and results are created by subview of the original
        // distributed buffer. Need calculate the stride by the original shape on each cluster
        for (auto& shape : distributedType.getPerClusterComputeShapes()) {
            SmallVector<int64_t> strideOnPerCluster(shape.size());
            int64_t preStride = 1;
            for (int64_t idx = dimOrder.numDims() - 1; idx >= 0; idx--) {
                auto dim = dimOrder.dimAt(idx);
                strideOnPerCluster[dim.ind()] = preStride;
                preStride *= shape[dim];
            }
            strideOnPerClusters.push_back(strideOnPerCluster);
        }
        strideAttr = vpux::getIntArrayOfArray(swKernelOp->getContext(), strideOnPerClusters);
    }
    return strideAttr;
}

//
// TileActShaveKernelTaskPass
//

class TileActShaveKernelTaskPass final : public VPUIP::impl::TileActShaveKernelTaskBase<TileActShaveKernelTaskPass> {
public:
    explicit TileActShaveKernelTaskPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void TileActShaveKernelTaskPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto tileOp = IE::getTileExecutor(module);
    auto shaveActCount = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT).getCount();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwKernelRewriter>(&ctx, shaveActCount, _log);
    patterns.add<ClusterSwKernelRewriter>(&ctx, shaveActCount, _log);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createTileActShaveKernelTaskPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createTileActShaveKernelTaskPass(Logger log) {
    return std::make_unique<TileActShaveKernelTaskPass>(log);
}

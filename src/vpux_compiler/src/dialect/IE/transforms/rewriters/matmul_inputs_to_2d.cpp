//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_dequantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LogicalResult.h>
#include <climits>
#include <cstdint>

namespace vpux::IE {
#define GEN_PASS_DECL_MATMULINPUTSTO2D
#define GEN_PASS_DEF_MATMULINPUTSTO2D
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MatMulOpConverter
//

class MatMulOpConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    MatMulOpConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log, bool enableGroupedMatMul)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx, benefit), _log(log), _enableGroupedMatMul(enableGroupedMatMul) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _enableGroupedMatMul;
};

static SmallVector<mlir::Value> sliceTensor(const mlir::Value tensorToSplit, const mlir::Location location,
                                            mlir::PatternRewriter& rewriter, StringRef tensorName) {
    const auto tensorShape = getShape(tensorToSplit);
    int64_t batch = 1;
    int64_t width = 1;
    int64_t height = 1;
    auto channelDim = Dim(0);
    if (tensorShape.size() == 3) {
        batch = tensorShape[Dim(0)];
        height = tensorShape[Dim(1)];
        width = tensorShape[Dim(2)];
        channelDim = Dim(0);
    } else if (tensorShape.size() == 4) {
        batch = tensorShape[Dim(1)];
        height = tensorShape[Dim(2)];
        width = tensorShape[Dim(3)];
        channelDim = Dim(1);
    } else if (tensorShape.size() == 2) {
        return {tensorToSplit};
    }
    SmallVector<mlir::Value> weightSlices;
    Shape rhsShape2D{height, width};
    const auto rhsShape2DAttr = getIntArrayAttr(rewriter.getContext(), rhsShape2D);
    if (batch > 1) {
        for (int64_t sliceIdx = 0; sliceIdx < batch; sliceIdx++) {
            Shape sliceOffsets = Shape(tensorShape.size(), 0);
            sliceOffsets[channelDim] = checked_cast<int64_t>(sliceIdx);
            auto staticOffsetsAttr = getIntArrayAttr(rewriter.getContext(), sliceOffsets);

            Shape sliceSizes = tensorShape.raw();
            sliceSizes[channelDim] = 1;
            auto staticSizesAttr = getIntArrayAttr(rewriter.getContext(), sliceSizes);
            auto newSubViewOp = rewriter.create<IE::SliceOp>(appendLoc(location, "{0}_slice_{1}", tensorName, sliceIdx),
                                                             tensorToSplit, staticOffsetsAttr, staticSizesAttr);

            auto rhs2d = rewriter.create<IE::ReshapeOp>(appendLoc(location, "{0}_reshape_{1}", tensorName, sliceIdx),
                                                        newSubViewOp, rhsShape2DAttr);
            weightSlices.push_back(rhs2d);
        }
    } else {
        auto rhs2d = rewriter.create<IE::ReshapeOp>(appendLoc(location, "{0}_reshape", tensorName), tensorToSplit,
                                                    rhsShape2DAttr);
        weightSlices.push_back(rhs2d);
    }

    return weightSlices;
}

// Structure to hold DynamicDequantize (+ optional Gather) chain information
struct DequantizeGatherChainInfo {
    IE::DynamicDequantizeOp dequantizeOp = nullptr;
    IE::ConvertOp convertOp = nullptr;                         // optional
    IE::GatherOp inputGatherOp = nullptr;                      // optional
    IE::AffineReshapeOp inputGatherAffineReshapeOp = nullptr;  // optional, reshape between Gather and DequantizeOp

    bool hasConvert() const {
        return convertOp != nullptr;
    }
    bool hasInputGather() const {
        return inputGatherOp != nullptr;
    }
};

// Check if an op is a reshape that adds a leading dim=1.
// Returns the reshape input value if valid, or std::nullopt otherwise.
static std::optional<mlir::Value> lookThroughLeadingDimReshape(mlir::Value value) {
    auto op = value.getDefiningOp();
    if (op == nullptr) {
        return std::nullopt;
    }
    if (!mlir::isa<IE::AffineReshapeOp, IE::ReshapeOp>(op)) {
        return std::nullopt;
    }
    if (!op->getResult(0).hasOneUse()) {
        return std::nullopt;
    }
    auto reshapeInShape = getShape(op->getOperand(0));
    auto reshapeOutShape = getShape(op->getResult(0));
    if (reshapeOutShape.size() != reshapeInShape.size() + 1 || reshapeOutShape.front() != 1) {
        return std::nullopt;
    }
    for (size_t i = 0; i < reshapeInShape.size(); ++i) {
        if (reshapeOutShape[Dim(i + 1)] != reshapeInShape[Dim(i)]) {
            return std::nullopt;
        }
    }
    return op->getOperand(0);
}

// Trace DynamicDequantize (+ optional Gather) chain feeding a MatMul weight input
[[nodiscard]] std::optional<DequantizeGatherChainInfo> traceDequantizeGatherChain(mlir::Value input) {
    DequantizeGatherChainInfo info;
    auto currentValue = input;

    // Look through optional AffineReshape or Reshape that adds a leading dim=1
    if (auto reshapeInput = lookThroughLeadingDimReshape(currentValue)) {
        currentValue = *reshapeInput;
    }

    // Check for optional Convert
    if (auto convertOp = currentValue.getDefiningOp<IE::ConvertOp>()) {
        if (!convertOp.getOutput().hasOneUse()) {
            return std::nullopt;
        }
        info.convertOp = convertOp;
        currentValue = convertOp.getInput();
    }

    // Check for DynamicDequantize (required)
    auto dequantizeOp = currentValue.getDefiningOp<IE::DynamicDequantizeOp>();
    if (dequantizeOp == nullptr) {
        return std::nullopt;
    }
    if (!dequantizeOp.getOutput().hasOneUse()) {
        return std::nullopt;
    }
    info.dequantizeOp = dequantizeOp;

    // Only raw data types are supported; reject if input comes from QuantizeCast
    if (mlir::isa_and_nonnull<IE::QuantizeCastOp>(dequantizeOp.getInput().getDefiningOp())) {
        return std::nullopt;
    }

    // Verify the input is 3D with a batch dimension to slice
    auto inputShape = getShape(dequantizeOp.getInput());
    if (inputShape.size() != 3 || inputShape[Dim(0)] <= 1) {
        return std::nullopt;
    }
    const auto batch = inputShape[Dim(0)];

    // Optionally trace through AffineReshape + Gather for the dequantize input.
    const auto isRequiredGatherOp = [](IE::GatherOp gatherOp) {
        return gatherOp.getAxisValue() == 0 && gatherOp.getBatchDims() == 0 && gatherOp.getIndicesRank() == 1;
    };

    // AffineReshape first dim_mapping group must be [0, 1]: the flat gather output
    // AffineReshape from like 64x32xf16 to 4x16x32xf16, 4 is the batch dimension.
    const auto isRequiredAffineReshape = [&](IE::AffineReshapeOp affineReshapeOp) {
        const auto outShape = getShape(affineReshapeOp.getOutput());
        const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMappingAttr());
        return !dimMapping.empty() && dimMapping[0].size() == 2 && dimMapping[0][0] == 0 && dimMapping[0][1] == 1 &&
               outShape[Dim(0)] == batch;
    };

    auto affineReshapeOp = dequantizeOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    if (affineReshapeOp != nullptr && affineReshapeOp->hasOneUse() && isRequiredAffineReshape(affineReshapeOp)) {
        auto gatherOp = affineReshapeOp.getInput().getDefiningOp<IE::GatherOp>();
        if (gatherOp != nullptr && gatherOp->hasOneUse() && isRequiredGatherOp(gatherOp)) {
            info.inputGatherOp = gatherOp;
            info.inputGatherAffineReshapeOp = affineReshapeOp;
        }
    }

    return info;
}

// Slice DynamicDequantize (+ optional Gather) chain per batch element.
// When chainInfo.hasInputGather(), slices flat_indices in chunks and creates per-batch Gathers;
// otherwise slices the reshaped tensor directly.
SmallVector<mlir::Value> sliceDequantizeGatherChain(DequantizeGatherChainInfo& chainInfo, const mlir::Location location,
                                                    mlir::PatternRewriter& rewriter, StringRef tensorName) {
    const auto ctx = rewriter.getContext();

    auto inputToSlice = chainInfo.dequantizeOp.getInput();

    // Get shapes
    auto inputShape = getShape(inputToSlice);
    auto dequantizeScale = chainInfo.dequantizeOp.getScale();

    // Original shape should be [B, H, W] where B is batch to slice
    int64_t batch = inputShape[Dim(0)];
    int64_t height = inputShape[Dim(1)];
    int64_t width = inputShape[Dim(2)];

    if (batch <= 1) {
        return {};
    }

    SmallVector<mlir::Value> resultSlices;

    // Slice along batch dimension
    for (int64_t sliceIdx = 0; sliceIdx < batch; sliceIdx++) {
        // Create input slice: either by slicing Gather flat-indices in chunks, or
        // by slicing the reshaped tensor directly.
        mlir::Value currentValue;
        if (chainInfo.hasInputGather()) {
            // Slice flat indices for this batch element and create a smaller Gather.
            // Gather(data, flat_indices[batch*rows]) + AffineReshape -> [batch, rows, width]
            // becomes: Gather(data, Slice(flat_indices, [i*rows], [rows])) -> [rows, width]
            //          + Reshape -> [1, rows, width]
            auto gatherOp = chainInfo.inputGatherOp;
            Shape indexOffset = {sliceIdx * height};
            Shape indexSize = {height};
            auto indexSlice = rewriter.create<IE::SliceOp>(
                    appendLoc(location, "{0}_input_idx_slice_{1}", tensorName, sliceIdx), gatherOp.getIndices(),
                    getIntArrayAttr(ctx, indexOffset), getIntArrayAttr(ctx, indexSize));
            auto perBatchGather =
                    rewriter.create<IE::GatherOp>(appendLoc(location, "{0}_input_gather_{1}", tensorName, sliceIdx),
                                                  gatherOp.getInput(), indexSlice.getOutput(), gatherOp.getAxisValue(),
                                                  gatherOp.getBatchDims(), gatherOp.getIndicesRankAttr());
            Shape perBatchShape = getShape(chainInfo.inputGatherAffineReshapeOp.getOutput()).raw();
            perBatchShape[Dim(0)] = 1;
            currentValue = rewriter.create<IE::AffineReshapeOp>(
                                           appendLoc(location, "{0}_input_reshape_{1}", tensorName, sliceIdx),
                                           perBatchGather.getResult(),
                                           chainInfo.inputGatherAffineReshapeOp.getDimMappingAttr(),
                                           getIntArrayAttr(ctx, perBatchShape))
                                   .getOutput();
        } else {
            Shape sliceOffsets = {sliceIdx, 0, 0};
            Shape sliceSizes = {1, height, width};
            currentValue =
                    rewriter.create<IE::SliceOp>(appendLoc(location, "{0}_input_slice_{1}", tensorName, sliceIdx),
                                                 inputToSlice, getIntArrayAttr(ctx, sliceOffsets),
                                                 getIntArrayAttr(ctx, sliceSizes))
                            .getOutput();
        }

        // Slice scale if it has batch dimension
        mlir::Value scaleSlice = dequantizeScale;
        auto scaleShape = getShape(dequantizeScale);
        if (scaleShape.size() >= 1 && scaleShape.front() == batch) {
            Shape scaleSliceOffsets(scaleShape.size(), 0);
            scaleSliceOffsets.front() = sliceIdx;
            Shape scaleSliceSizes = scaleShape.raw();
            scaleSliceSizes.front() = 1;
            scaleSlice = rewriter.create<IE::SliceOp>(appendLoc(location, "{0}_scale_slice_{1}", tensorName, sliceIdx),
                                                      dequantizeScale, getIntArrayAttr(ctx, scaleSliceOffsets),
                                                      getIntArrayAttr(ctx, scaleSliceSizes));
        }

        // Slice zero-point if present and has batch dimension
        mlir::Value zpSlice = chainInfo.dequantizeOp.getZp();
        if (zpSlice) {
            auto zpShape = getShape(zpSlice);
            if (zpShape.size() >= 1 && zpShape.front() == batch) {
                Shape zpSliceOffsets(zpShape.size(), 0);
                zpSliceOffsets.front() = sliceIdx;
                Shape zpSliceSizes = zpShape.raw();
                zpSliceSizes.front() = 1;
                zpSlice = rewriter.create<IE::SliceOp>(appendLoc(location, "{0}_zp_slice_{1}", tensorName, sliceIdx),
                                                       zpSlice, getIntArrayAttr(ctx, zpSliceOffsets),
                                                       getIntArrayAttr(ctx, zpSliceSizes));
            }
        }

        // Create sliced DynamicDequantize
        auto dequantSlice = rewriter.create<IE::DynamicDequantizeOp>(
                appendLoc(location, "{0}_dequant_slice_{1}", tensorName, sliceIdx), currentValue, scaleSlice, zpSlice,
                chainInfo.dequantizeOp.getDstElemType());
        if (chainInfo.dequantizeOp->hasAttr(IE::SYNTHETIC_DYN_DEQUANT_ATTR)) {
            dequantSlice->setAttr(IE::SYNTHETIC_DYN_DEQUANT_ATTR,
                                  chainInfo.dequantizeOp->getAttr(IE::SYNTHETIC_DYN_DEQUANT_ATTR));
        }

        currentValue = dequantSlice.getOutput();

        // Apply Convert if present in original chain
        if (chainInfo.hasConvert()) {
            auto convertSlice =
                    rewriter.create<IE::ConvertOp>(appendLoc(location, "{0}_convert_slice_{1}", tensorName, sliceIdx),
                                                   currentValue, chainInfo.convertOp.getDstElemType());
            currentValue = convertSlice.getOutput();
        }

        // Reshape to 2D [H, W]
        Shape shape2D = {height, width};
        auto reshapeSlice =
                rewriter.create<IE::ReshapeOp>(appendLoc(location, "{0}_reshape_slice_{1}", tensorName, sliceIdx),
                                               currentValue, getIntArrayAttr(ctx, shape2D));

        resultSlices.push_back(reshapeSlice.getOutput());
    }

    return resultSlices;
}

// Create slice for specific GroupConvolution
mlir::Value createGroupConvSlice(mlir::PatternRewriter& rewriter, mlir::Value input, int64_t groupIdx, int64_t groups,
                                 ShapeRef inputShape, mlir::Location loc, StringRef name) {
    int64_t startBatch = groupIdx * VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    Shape sliceOffsets = Shape(inputShape.size(), 0);
    sliceOffsets[Dim(0)] = startBatch;
    Shape sliceSizes = inputShape.raw();
    sliceSizes[Dim(0)] = groups;

    return rewriter.create<IE::SliceOp>(appendLoc(loc, "{0}_slice_{1}", name, groupIdx), input,
                                        getIntArrayAttr(rewriter.getContext(), sliceOffsets),
                                        getIntArrayAttr(rewriter.getContext(), sliceSizes));
}

// Convert MatMul to GroupConvolutions
std::optional<mlir::Value> convertMatMulToGroupConvolutions(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) {
    const auto ctx = rewriter.getContext();
    const auto loc = matmulOp->getLoc();
    const auto input1 = matmulOp.getInput1();
    const auto input2 = matmulOp.getInput2();
    const auto input1Shape = getShape(input1);
    const auto input2Shape = getShape(input2);
    const int rank3D = 3;
    if (input1Shape.size() < rank3D) {
        return std::nullopt;
    }

    auto convertToShape3D = [](ShapeRef inputShape) -> Shape {
        if (inputShape.size() <= rank3D) {
            return inputShape.toValues();
        }
        const auto rank = inputShape.size();
        int64_t collapsedBatch = 1;
        for (size_t i = 0; i < rank - 2; ++i) {
            collapsedBatch *= inputShape[Dim(i)];
        }

        return Shape{collapsedBatch, inputShape[Dim(rank - 2)], inputShape[Dim(rank - 1)]};
    };

    Shape input1Shape3d = convertToShape3D(input1Shape);
    Shape input2Shape3d = convertToShape3D(input2Shape);
    if (input1Shape3d.size() != rank3D) {
        return std::nullopt;
    }

    const int64_t totalBatches = input1Shape3d[Dims3D::Act::B];
    int64_t groups = VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    int64_t numGroupConvs = totalBatches / VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    int64_t remainBatches = totalBatches % VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    if (remainBatches > 0) {
        numGroupConvs += 1;
    }
    const int64_t inputH = matmulOp.getTransposeA() ? input1Shape3d[Dims3D::Act::IC] : input1Shape3d[Dims3D::Act::H];
    const int64_t inputW = matmulOp.getTransposeA() ? input1Shape3d[Dims3D::Act::H] : input1Shape3d[Dims3D::Act::IC];
    const int64_t weightsH = matmulOp.getTransposeB() ? input2Shape3d[Dims3D::Act::H] : input2Shape3d[Dims3D::Act::IC];
    const int64_t weightsW = matmulOp.getTransposeB() ? input2Shape3d[Dims3D::Act::IC] : input2Shape3d[Dims3D::Act::H];

    // Reshape inputs
    Shape inputGroupConvShape = {totalBatches, inputH, inputW};
    auto inputReshapeOp = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "input_groupConv_reshape"), input1,
                                                         getIntArrayAttr(ctx, inputGroupConvShape));
    Shape weightsGroupConvShape = {totalBatches, weightsH, weightsW};
    auto weightsReshapeOp = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "weights_groupConv_reshape"), input2,
                                                           getIntArrayAttr(ctx, weightsGroupConvShape));

    // Process each group with GroupConvolution
    SmallVector<mlir::Value> groupResults;
    for (int64_t idx = 0; idx < numGroupConvs; ++idx) {
        if (idx == numGroupConvs - 1 && remainBatches > 0) {
            groups = remainBatches;
        }

        auto inputSliceOp = createGroupConvSlice(rewriter, inputReshapeOp.getOutput(), idx, groups, inputGroupConvShape,
                                                 loc, "input");

        Shape newInputSliceShape = {1, groups, inputH, inputW};
        auto newInputSliceOp = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "input_slice_reshape_{0}", idx),
                                                              inputSliceOp, getIntArrayAttr(ctx, newInputSliceShape));

        auto weightsSliceOp = createGroupConvSlice(rewriter, weightsReshapeOp.getOutput(), idx, groups,
                                                   weightsGroupConvShape, loc, "weights");

        Shape newWeightsSliceShape = {groups, 1, weightsH, weightsW};
        auto newWeightsSliceOp =
                rewriter.create<IE::ReshapeOp>(appendLoc(loc, "weights_slice_reshape_{0}", idx), weightsSliceOp,
                                               getIntArrayAttr(ctx, newWeightsSliceShape));

        auto groupConvOp = rewriter.create<IE::GroupConvolutionOp>(
                appendLoc(loc, "group_conv_{0}", idx), newInputSliceOp, newWeightsSliceOp,
                /*bias=*/nullptr, getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1}),
                getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0}), getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0}),
                getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1}), rewriter.getI64IntegerAttr(groups),
                /*post_opAttr=*/nullptr, /*clampAttr=*/nullptr,
                /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);

        groupResults.push_back(groupConvOp.getOutput());
    }

    // Concat all GroupConvolution results along channel dimension
    auto concatOp = rewriter.create<IE::ConcatOp>(appendLoc(loc, "group_concat"), groupResults, Dims4D::Act::C);

    // Final reshape
    auto finalShape = getShape(matmulOp.getOutput());
    return rewriter.create<IE::ReshapeOp>(appendLoc(loc, "final_reshape"), concatOp.getOutput(),
                                          getIntArrayAttr(ctx, finalShape));
}

mlir::LogicalResult MatMulOpConverter::matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const {
    // E-122051:
    // MatMulInputsTo2dPass should be moved to a new pass `ConvertMatMulToFullyConnected`.
    // This check should be moved in a `addDynamicallyLegalOp<IE::MatMulOp>`.
    // Transpose should be done after `ReshapeNDInputConverter` (not in canonicalizer), experiments show that it
    // is faster when batch dimensions are merged.
    if (VPU::MatMulOp::isSupported(matmulOp)) {
        if (matmulOp.getTransposeB()) {
            auto input2 = matmulOp.getInput2();
            auto input2Rank = getShape(input2).size();
            VPUX_THROW_UNLESS(input2Rank > 2,
                              "VPU::MatMulOp only supports input 2 rank bigger than 2. "
                              "If that changes, this code needs update. Input 2 rank = '{0}'",
                              input2Rank);
            SmallVector<uint32_t> perm(input2Rank, 0);
            std::iota(perm.begin(), perm.end(), 0);
            std::iter_swap(perm.end() - 1, perm.end() - 2);
            const auto orderAttr =
                    mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, matmulOp->getContext()));
            input2 = rewriter.create<IE::TransposeOp>(takeOpLoc(matmulOp, "input_b_transpose"), input2, nullptr,
                                                      orderAttr)
                             .getOutput();
            rewriter.replaceOp(matmulOp, cloneMatMulOp(rewriter, matmulOp, matmulOp.getInput1(), input2, false, false));
            return mlir::success();
        }
        return mlir::failure();
    }

    auto input1Shape = getShape(matmulOp.getInput1());
    auto input2Shape = getShape(matmulOp.getInput2());

    // 1. Cover 3D input or weights.
    // 2. Cover 4D input and weights without batch.
    if (!(input1Shape.size() == 3 && input2Shape.size() == 3) &&
        !(input1Shape.size() == 4 &&
          ((input2Shape.size() == 4 || input2Shape.size() == 3) && input1Shape[Dim(0)] == 1))) {
        return mlir::failure();
    }

    // Ideally this should be skipped using calculation from ReshapeNDInputConverter
    if (_enableGroupedMatMul && IE::isGroupedMatMulBeneficial(matmulOp, input1Shape, input2Shape)) {
        if (isGroupedMatMulBeneficialToGroupConv(matmulOp)) {
            // Handle the MatMul with huge batch
            auto convertedOp = convertMatMulToGroupConvolutions(matmulOp, rewriter);
            if (convertedOp.has_value()) {
                rewriter.replaceOp(matmulOp, convertedOp.value());
                return mlir::success();
            }
        }

        return mlir::failure();
    }

    SmallVector<mlir::Value> activationSlices =
            sliceTensor(matmulOp.getInput1(), matmulOp->getLoc(), rewriter, "activation");

    // Check if input2 has DynamicDequantize Gather chain pattern
    SmallVector<mlir::Value> weightSlices;
    auto dequantChain = traceDequantizeGatherChain(matmulOp.getInput2());
    if (dequantChain.has_value()) {
        // Slice the entire DynamicDequantize chain preserving scale/zp correctly
        weightSlices = sliceDequantizeGatherChain(dequantChain.value(), matmulOp->getLoc(), rewriter, "weights");
        // If sliceDequantizeGatherChain returns empty (batch <= 1 case), it should be handled by other converters
        if (weightSlices.empty()) {
            return mlir::failure();
        }
    } else {
        // Normal tensor slicing
        weightSlices = sliceTensor(matmulOp.getInput2(), matmulOp->getLoc(), rewriter, "weights");
    }

    // Handle broadcasting by replicating the slices of the broadcasted input to match
    // the number of slices of the non-broadcasted input.
    if (activationSlices.size() != weightSlices.size()) {
        if (activationSlices.size() == 1) {
            activationSlices = SmallVector<mlir::Value>(weightSlices.size(), activationSlices[0]);
        } else if (weightSlices.size() == 1) {
            weightSlices = SmallVector<mlir::Value>(activationSlices.size(), weightSlices[0]);
        } else {
            VPUX_THROW("Mismatch activationSlices number '{0}' with weightSlices number '{1}'", activationSlices.size(),
                       weightSlices.size());
        }
    }

    SmallVector<mlir::Value> matmulSlices;
    for (size_t sliceIdx = 0; sliceIdx < activationSlices.size(); sliceIdx++) {
        auto lhs2d = activationSlices[sliceIdx];
        auto rhs2d = weightSlices[weightSlices.size() == 1 ? 0 : sliceIdx];
        auto op = cloneMatMulOp(rewriter, matmulOp, lhs2d, rhs2d);
        op->setLoc(takeOpLoc(matmulOp, "slice_{0}", sliceIdx));

        matmulSlices.push_back(op->getResult(0));
    }

    VPUX_THROW_WHEN(matmulSlices.empty(), "Cannot slice MatMul operation with input shape {0}, weights' shape {1}",
                    input1Shape, input2Shape);

    auto newOp = matmulSlices.size() != 1
                         ? rewriter.create<IE::ConcatOp>(takeOpLoc(matmulOp, "slice_gather"), matmulSlices, 0)
                         : matmulSlices.front();

    const auto outShape4D = getShape(matmulOp.getOutput());
    const auto outShape4DAttr = getIntArrayAttr(rewriter.getContext(), outShape4D);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, newOp, outShape4DAttr);

    return mlir::success();
}

//
// ReshapeNDInputConverter
//

class ReshapeNDInputConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    ReshapeNDInputConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log, bool enableGroupedMatMul)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx, benefit), _log(log), _enableGroupedMatMul(enableGroupedMatMul) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _enableGroupedMatMul;
};

mlir::LogicalResult ReshapeNDInputConverter::matchAndRewrite(IE::MatMulOp matmulOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), matmulOp->getName(), matmulOp->getLoc());

    auto transposeA = matmulOp.getTransposeA();
    auto transposeB = matmulOp.getTransposeB();
    auto input1Shape = getShape(matmulOp.getInput1());
    auto input2Shape = getShape(matmulOp.getInput2());

    auto adjustTo3DShape = [](ShapeRef origShape, const bool isFirstInput) {
        if (origShape.size() == 1) {
            return isFirstInput ? Shape{1, 1, origShape.front()} : Shape{1, origShape.front(), 1};
        }

        auto batchSize = std::accumulate(origShape.begin(), origShape.end() - 2, 1, std::multiplies<int64_t>());
        return Shape{batchSize, origShape[Dim(origShape.size() - 2)], origShape[Dim(origShape.size() - 1)]};
    };

    // Adjust second input
    // Step 1: Adjust input shape to a tensor rank of 3D [Batch, Height, Width]
    //  If the tensor is 1D, the size is assigned to Height, and both Batch and Width are set to 1
    //    - For example: [6] -> [1, 6, 1]
    //  If the tensor is larger or equal than 2D, the last two dimensions are assigned to Height and Width
    //  and Batch is set to the product of the remaining dimensions
    //    - For example: [2, 3] -> [1, 2, 3]; [1, 1, 6] -> [1, 1, 6]; [3, 1, 6, 4, 2] -> [18, 4, 2]
    // Step 2: The batch dimension can be removed if its size equals 1
    //    - For example: [1, 1, 1, 8] -> [1, 1, 8] -> [1, 8]
    //                   [1, 6, 1, 8] -> [6, 1, 8] -> [6, 1, 8]
    auto newIn2Shape = adjustTo3DShape(input2Shape, /*isFirstInput=*/false);
    if (newIn2Shape.front() == 1) {
        newIn2Shape.erase(newIn2Shape.begin());
    }

    // Adjust first input
    // Step 1: Adjust input shape to a tensor rank of 3D [Batch, Height, Width]
    //  If the tensor is 1D, the size is assigned to Width, and both Batch and Height are set to 1
    //    - For example: [6] -> [1, 1, 6]
    //  If the tensor is larger or equal than 2D, the last two dimensions are assigned to Height and Width
    //  and Batch is set to the product of the remaining dimensions
    //    - For example: [2, 3] -> [1, 2, 3]; [1, 2, 6] -> [1, 2, 6]; [3, 1, 6, 4, 2] -> [18, 4, 2]
    // Step 2: If transposeA is set to false or batch equal 1 and the new second input shape lacks a batch dimension
    //         the Batch can be integrated into the Height dimension.
    // For example:
    //   MatMul(2x3x6x4, 4x8) {transposeA = false} collapses to MatMul(36x4, 4x8) {transposeA = false}
    //   MatMul(2x3x4x6, 4x8) {transposeA = true} collapses to MatMul(6x4x6, 4x8) {transposeA = true}
    //   MatMul(1x2x6x4, 2x4x8) {transposeA = false} collapses to MatMul(2x6x4, 2x4x8) {transposeA = false}
    auto newIn1Shape = adjustTo3DShape(input1Shape, /*isFirstInput=*/true);
    if (newIn2Shape.size() == 2 && (!transposeA || newIn1Shape.front() == 1)) {
        auto batchSize = newIn1Shape.front();
        newIn1Shape.erase(newIn1Shape.begin());
        newIn1Shape[Dim(0)] = newIn1Shape[Dim(0)] * batchSize;
    }
    if (newIn1Shape == input1Shape && newIn2Shape == input2Shape) {
        return mlir::failure();
    }

    // If both inputs are 3D after adjustment, check that collapsing original batch dims into a single batch
    // dimension is semantics-preserving. This is true only when either:
    //   (a) all aligned batch dims are element-wise equal, OR
    //   (b) one operand's batch product is 1 (broadcast at 3D preserves semantics).
    // Counter-example: input1 batch [2,1] × input2 batch [1,2] both flatten to batch=2,
    // but the correct output requires 2×2=4 matmuls (broadcast), not 2.
    if (newIn1Shape.size() == 3 && newIn2Shape.size() == 3) {
        const auto batch1 = newIn1Shape.front();
        const auto batch2 = newIn2Shape.front();
        if (batch1 != batch2 && batch1 != 1 && batch2 != 1) {
            return mlir::failure();
        }
        if (batch1 != 1 && batch2 != 1) {
            const auto numBatchDims1 = input1Shape.size() - 2;
            const auto numBatchDims2 = input2Shape.size() - 2;
            const auto maxBatchDims = std::max(numBatchDims1, numBatchDims2);
            auto getAlignedBatchDim = [&](ShapeRef shape, size_t numBatchDims, size_t i) -> int64_t {
                return (i >= maxBatchDims - numBatchDims) ? shape[Dim(i - (maxBatchDims - numBatchDims))] : 1;
            };
            for (size_t i = 0; i < maxBatchDims; ++i) {
                if (getAlignedBatchDim(input1Shape, numBatchDims1, i) !=
                    getAlignedBatchDim(input2Shape, numBatchDims2, i)) {
                    return mlir::failure();
                }
            }
        }
    }

    if (_enableGroupedMatMul && newIn1Shape.size() > 2 && newIn2Shape.size() > 2 && newIn1Shape.front() != 1 &&
        newIn2Shape.front() != 1) {
        if (IE::isGroupedMatMulBeneficial(matmulOp, newIn1Shape, newIn2Shape)) {
            return mlir::failure();
        }
    }

    // Check if the original input shapes are either both 3D or both 4D without batch
    // If they can be converted to MatMul without batch, the shapes will be adjusted
    // If not, the following conversion logic can directly slice them without the need for reshaping
    const auto isNewShapeWithoutBatch = (newIn1Shape.size() == 2) && (newIn2Shape.size() == 2);
    const auto is3DInput = (input1Shape.size() == 3) && (input2Shape.size() == 3);
    const auto is4DInputWithoutBatch =
            input1Shape.size() == 4 && (input2Shape.size() == 4 || input2Shape.size() == 3) && input1Shape[Dim(0)] == 1;
    if (!isNewShapeWithoutBatch && (is3DInput || is4DInputWithoutBatch)) {
        return mlir::failure();
    }

    // Adjust MatMul inputs
    auto adjustInputTensor = [&](mlir::Value input, ShapeRef newShape, mlir::Location newLoc) {
        return rewriter.createOrFold<IE::ReshapeOp>(newLoc, input, getIntArrayAttr(rewriter.getContext(), newShape));
    };

    auto reshapeInput1 = adjustInputTensor(matmulOp.getInput1(), newIn1Shape, takeOpLoc(matmulOp, "in1_reshape"));
    auto reshapeInput2 = adjustInputTensor(matmulOp.getInput2(), newIn2Shape, takeOpLoc(matmulOp, "in2_reshape"));

    auto newMatMul = cloneMatMulOp(rewriter, matmulOp, reshapeInput1, reshapeInput2, transposeA, transposeB);

    const auto origOutShape = getShape(matmulOp.getOutput());
    const auto origOutShapeAttr = getIntArrayAttr(rewriter.getContext(), origOutShape);
    auto newOp = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, newMatMul->getResult(0), origOutShapeAttr);
    extendOpLoc(newOp, "out_reshape");
    return mlir::success();
}

//
// SwapInputsConverter
//

/*
Swap input1 with input2 when input1 is 2D after invalid dimensions removed and dimensions of input2 are bigger than 2D
For example:
    IE.MatMul(input1, input2) {transpose_b} : tensor<1x1x32x64xf32>, tensor<1x6336x1x64xf32> -> tensor<1x6336x32x1xf32>
-->
    IE.MatMul(input2, input1) {transpose_b} : tensor<1x6336x1x64xf32>, tensor<32x64xf32> -> tensor<1x6336x1x32xf32>

Then using ReshapeNDInputConverter pattern to convert the new input1 to 2D as:
    IE.MatMul(input1, input2) {transpose_b} : tensor<6336x64xf32>, tensor<32x64xf32> -> tensor<6336x32xf32>
*/
class SwapInputsConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    SwapInputsConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapInputsConverter::matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), matmulOp->getName(), matmulOp->getLoc());
    auto ctx = rewriter.getContext();

    auto transposeA = matmulOp.getTransposeA();
    auto transposeB = matmulOp.getTransposeB();
    auto input1Shape = getShape(matmulOp.getInput1());
    auto input2Shape = getShape(matmulOp.getInput2());

    auto adjustTo2DShape = [](ShapeRef origShape) -> std::optional<Shape> {
        if (origShape.size() == 1) {
            return std::nullopt;
        }

        auto batchSize = std::accumulate(origShape.begin(), origShape.end() - 2, 1, std::multiplies<int64_t>());
        if (batchSize != 1) {
            return std::nullopt;
        }

        return Shape{origShape[Dim(origShape.size() - 2)], origShape[Dim(origShape.size() - 1)]};
    };

    auto newIn1ShapeValue = adjustTo2DShape(input1Shape);
    if (!newIn1ShapeValue.has_value()) {
        return mlir::failure();
    }
    auto newIn1Shape = newIn1ShapeValue.value();

    // When both input1 and input2 can be reshaped to 2D, matmulOp can directly be rewrited by ReshapeNDInputConverter
    auto inShape2Size = input2Shape.size();
    if (inShape2Size <= 2 || adjustTo2DShape(input2Shape).has_value()) {
        return mlir::failure();
    }

    const auto dimToCheck = transposeB ? input2Shape[Dim(inShape2Size - 2)] : input2Shape[Dim(inShape2Size - 1)];
    if (dimToCheck != 1) {
        return mlir::failure();
    }

    auto layerWithPostOp = mlir::cast<IE::LayerWithPostOpInterface>(matmulOp.getOperation());
    if (layerWithPostOp) {
        const auto postOp = layerWithPostOp.getPostOp();
        if (postOp != nullptr && !postOp.isChannelAgnostic()) {
            return mlir::failure();
        }
    }

    auto newTransposeB = transposeA ? false : true;
    auto newTransposeA = transposeB ? false : true;

    auto in2ReshapeOp =
            rewriter.create<IE::ReshapeOp>(matmulOp->getLoc(), matmulOp.getInput1(), getIntArrayAttr(ctx, newIn1Shape));

    auto newMatMulOp = cloneMatMulOp(rewriter, matmulOp, matmulOp.getInput2(), in2ReshapeOp.getOutput(), newTransposeA,
                                     newTransposeB);

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, newMatMulOp->getResult(0),
                                               getIntArrayAttr(ctx, getShape(matmulOp.getOutput())));
    return mlir::success();
}

}  // namespace

void vpux::IE::registerMatMulInputsTo2dRewriters(RewriterRegistry& registry, Logger log,
                                                 ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index,
                                                 bool enableGroupedMatMul) {
    auto benefitLevelsSubset = extractBenefitLevels(benefitLevels, /*statIndex*/ index, /*numLevels*/ 3);
    registry.registerRewriterSet(
            "matmul-inputs-to-2d-set", [&registry, log, benefitLevelsSubset, enableGroupedMatMul]() {
                registry.registerRewriter<SwapInputsConverter>("swap-inputs-converter", benefitLevelsSubset[0], log);
                registry.registerRewriter<ReshapeNDInputConverter>("reshape-nd-input-converter", benefitLevelsSubset[1],
                                                                   log, enableGroupedMatMul);
                registry.registerRewriter<MatMulOpConverter>("matmul-converter", benefitLevelsSubset[2], log,
                                                             enableGroupedMatMul);
                IE::registerReshapeOpRewriters(registry, benefitLevelsSubset, 2);
            });
}

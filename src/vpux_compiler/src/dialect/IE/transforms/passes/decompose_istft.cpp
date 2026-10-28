//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/transposed_convolution_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <cmath>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSEISTFT
#define GEN_PASS_DEF_DECOMPOSEISTFT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ISTFTOpConverter
//

class ISTFTOpConverter final : public mlir::OpRewritePattern<IE::ISTFTOp> {
public:
    ISTFTOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ISTFTOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ISTFTOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    // Strategy selector - dispatches to appropriate overlap-add implementation
    mlir::Value createHybridOverlapAdd(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::MLIRContext* ctx,
                                       mlir::Value frames, int64_t numFrames, int64_t frameSizeVal,
                                       int64_t frameStepVal, int64_t outputLength, mlir::Type elemType,
                                       const SmallVector<int64_t>& batchDims, bool isCenter) const;

    // Large case: tiled overlap-add with MatMul for memory efficiency
    mlir::Value createTiledOverlapAdd(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::MLIRContext* ctx,
                                      mlir::Value frames, int64_t numFrames, int64_t frameSizeVal, int64_t frameStepVal,
                                      int64_t olaLength, mlir::Type elemType, const SmallVector<int64_t>& batchDims,
                                      bool isCenter) const;

    // Small case: explicit frame-by-frame overlap-add
    mlir::Value createExplicitOverlapAdd(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::MLIRContext* ctx,
                                         mlir::Value frames, int64_t numFrames, int64_t frameSizeVal,
                                         int64_t frameStepVal, int64_t olaLength, mlir::Type elemType,
                                         const SmallVector<int64_t>& batchDims, bool isCenter) const;

    // Helper: process single chunk in tiled strategy
    void processChunk(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::MLIRContext* ctx, mlir::Value& result,
                      mlir::Value frames, int64_t chunkStart, int64_t chunkEnd, int64_t frameSizeVal,
                      int64_t frameStepVal, int64_t olaLength, mlir::Type elemType,
                      const SmallVector<int64_t>& batchDims, bool isCenter) const;
};

mlir::LogicalResult ISTFTOpConverter::matchAndRewrite(IE::ISTFTOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto ctx = origOp.getContext();
    const auto loc = origOp.getLoc();

    _log.trace("Decomposing ISTFT operation at {0}", loc);

    auto signal = origOp.getSignal();
    const auto signalType = mlir::cast<vpux::NDTypeInterface>(signal.getType());
    const auto inShape = signalType.getShape().raw();
    const auto elemType = signalType.getElementType();

    const int64_t numFrames = inShape[inShape.size() - 2];
    _log.trace("Decomposing ISTFT with {0} frames", numFrames);

    auto frameSize = origOp.getFrameSize();
    auto frameStep = origOp.getFrameStep();
    auto frameSizeConstOp = frameSize.getDefiningOp<Const::DeclareOp>();
    auto frameStepConstOp = frameStep.getDefiningOp<Const::DeclareOp>();

    if (!frameSizeConstOp || !frameStepConstOp) {
        return mlir::failure();
    }

    const auto frameSizeContent = frameSizeConstOp.getContent();
    const auto frameStepContent = frameStepConstOp.getContent();
    const auto frameSizeVal = frameSizeContent.getSplatValue<int64_t>();
    const auto frameStepVal = frameStepContent.getSplatValue<int64_t>();

    const bool isCenter = origOp.getCenter().has_value() && origOp.getCenter().value();
    const bool isNormalized = origOp.getNormalized().has_value() && origOp.getNormalized().value();

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto outputShape = outputType.getShape().raw();
    const int64_t expectedOutputLength = outputShape[outputShape.size() - 1];

    int64_t explicitSignalLength = -1;
    bool hasExplicitSignalLength = false;

    if (auto signalLengthInput = origOp.getSignalLength()) {
        auto signalLengthConstOp = signalLengthInput.getDefiningOp<Const::DeclareOp>();
        if (!signalLengthConstOp) {
            _log.trace("Skip ISTFT decomposition: signal_length is non-constant");
            return mlir::failure();
        }
        const auto signalLengthContent = signalLengthConstOp.getContent();
        explicitSignalLength = signalLengthContent.getSplatValue<int64_t>();
        hasExplicitSignalLength = true;
    }

    // Calculate output length based on ISTFT parameters and centering mode
    int64_t standardOlaLength = (numFrames - 1) * frameStepVal + frameSizeVal;

    int64_t olaLength;
    int64_t desiredOutLength;
    bool hasSignalLength;

    if (isCenter) {
        if (hasExplicitSignalLength) {
            desiredOutLength = explicitSignalLength;
            olaLength = standardOlaLength;
            hasSignalLength = true;
        } else {
            desiredOutLength = (numFrames - 1) * frameStepVal;
            olaLength = desiredOutLength;
            hasSignalLength = false;
        }
    } else {
        olaLength = standardOlaLength;

        if (hasExplicitSignalLength) {
            desiredOutLength = explicitSignalLength;
            hasSignalLength = true;
        } else {
            desiredOutLength = olaLength;
            hasSignalLength = false;
        }
    }

    if (desiredOutLength != expectedOutputLength) {
        desiredOutLength = expectedOutputLength;
        hasSignalLength = true;
    }

    // Extract batch dimensions (all dimensions except the last 3: batch, frames, freq_bins)
    SmallVector<int64_t> batchDims;
    for (size_t i = 0; i < inShape.size() - 3; ++i) {
        batchDims.push_back(inShape[i]);
    }

    // Create transpose order to swap frames and frequency bins for vectorized IRDFT
    // Input: [..., freq_bins, frames, complex] -> [..., frames, freq_bins, complex]
    SmallVector<uint32_t> transposeOrderVec;
    const size_t inputRank = inShape.size();
    for (size_t i = 0; i < inputRank; ++i) {
        if (i == inputRank - 3) {
            // Swap freq_bins (rank-3) with frames (rank-2)
            transposeOrderVec.push_back(inputRank - 2);
        } else if (i == inputRank - 2) {
            // Swap frames (rank-2) with freq_bins (rank-3)
            transposeOrderVec.push_back(inputRank - 3);
        } else {
            // Keep other dimensions unchanged
            transposeOrderVec.push_back(i);
        }
    }

    const auto transposeOrder = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transposeOrderVec, ctx));
    auto transposedSignal =
            IE::TransposeOp::create(rewriter, appendLoc(loc, "transpose"), signal, nullptr, transposeOrder);

    auto transposedType = mlir::cast<mlir::ShapedType>(transposedSignal.getOutput().getType());
    auto transposedShape = transposedType.getShape();

    // Configure IRDFT to operate on frequency dimension
    SmallVector<int64_t> irdftAxes = {static_cast<int64_t>(transposedShape.size() - 2)};
    SmallVector<int64_t> irdftSignalSize = {frameSizeVal};

    // Calculate output shape after IRDFT: frequency dimension becomes time dimension
    SmallVector<int64_t> timeShape;
    for (size_t i = 0; i < transposedShape.size(); ++i) {
        if (i == transposedShape.size() - 2) {
            // Frequency dimension becomes time dimension with frameSizeVal samples
            timeShape.push_back(frameSizeVal);
        } else if (i == transposedShape.size() - 1) {
            // Skip complex dimension - IRDFT converts complex to real
        } else {
            // Keep other dimensions unchanged
            timeShape.push_back(transposedShape[i]);
        }
    }

    // Perform vectorized IRDFT on all frames simultaneously
    auto timeType = mlir::RankedTensorType::get(timeShape, elemType);
    auto timeFrames =
            IE::IRDFTOp::create(rewriter, appendLoc(loc, "irdft"), timeType, transposedSignal.getOutput(), nullptr,
                                nullptr, getIntArrayAttr(ctx, irdftAxes), getIntArrayAttr(ctx, irdftSignalSize));

    auto autoBroadcast = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
    mlir::Value windowedFrames = timeFrames.getResult();

    if (isNormalized) {
        const float sqrtFrameSize = std::sqrt(static_cast<float>(frameSizeVal));
        const auto scalarType = mlir::RankedTensorType::get({}, elemType);
        auto sqrtConst = Const::createConst(rewriter, loc, scalarType, ArrayRef<float>{sqrtFrameSize});
        windowedFrames = IE::MultiplyOp::create(rewriter, appendLoc(loc, "denormalize"), windowedFrames, sqrtConst,
                                                autoBroadcast, nullptr, nullptr, nullptr, nullptr)
                                 .getOutput();
    }

    auto window = origOp.getWindow();
    if (window) {
        windowedFrames = IE::MultiplyOp::create(rewriter, appendLoc(loc, "window_multiply"), windowedFrames, window,
                                                autoBroadcast, nullptr, nullptr, nullptr, nullptr)
                                 .getOutput();
    }

    // Perform overlap-add reconstruction using hybrid strategy
    mlir::Value olaResult = createHybridOverlapAdd(rewriter, loc, ctx, windowedFrames, numFrames, frameSizeVal,
                                                   frameStepVal, olaLength, elemType, std::move(batchDims), isCenter);

    mlir::Value finalResult = olaResult;

    // Apply window normalization if window function was used
    if (window) {
        // Create full window data with proper padding
        SmallVector<float> windowDataFull(frameSizeVal, 1.0f);
        auto winConst = window.getDefiningOp<Const::DeclareOp>();
        if (winConst) {
            auto windowD = winConst.getContent();
            auto windowValues = to_small_vector(windowD.getValues<float>());
            int64_t windowLen = windowValues.size();

            if (windowLen < frameSizeVal) {
                // Center-pad shorter windows
                int64_t padLeft = (frameSizeVal - windowLen) / 2;
                std::fill(windowDataFull.begin(), windowDataFull.begin() + padLeft, 0.0f);
                std::copy(windowValues.begin(), windowValues.end(), windowDataFull.begin() + padLeft);
                std::fill(windowDataFull.begin() + padLeft + windowLen, windowDataFull.end(), 0.0f);
            } else {
                windowDataFull.assign(windowValues.begin(), windowValues.end());
            }
        }

        // Calculate window sum for normalization (sum of squared window values at each output position)
        SmallVector<float> windowSumData(olaLength, 0.0f);

        for (int64_t frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
            int64_t frameStart;

            if (isCenter) {
                frameStart = frameIdx * frameStepVal - frameSizeVal / 2;
            } else {
                frameStart = frameIdx * frameStepVal;
            }

            for (int64_t i = 0; i < frameSizeVal; ++i) {
                int64_t outputPos = frameStart + i;

                if (outputPos >= 0 && outputPos < olaLength) {
                    float windowVal = windowDataFull[i];
                    windowSumData[outputPos] += windowVal * windowVal;
                }
            }
        }

        // Add small epsilon to prevent division by zero
        if (isCenter) {
            for (auto& val : windowSumData) {
                if (val < 1e-8f) {
                    val = 1.0f;
                } else {
                    val += 1e-8f;
                }
            }
        } else {
            for (auto& val : windowSumData) {
                val += 1e-8f;
            }
        }

        // Create 1D window sum tensor and rely on NUMPY broadcasting over batch dimensions
        auto windowSumType = mlir::RankedTensorType::get({olaLength}, elemType);
        auto windowSumConst = Const::createConst(rewriter, loc, windowSumType,
                                                 ArrayRef<float>{windowSumData.data(), windowSumData.size()});

        // Normalize by window sum
        finalResult = IE::DivideOp::create(rewriter, appendLoc(loc, "window_normalize"), olaResult, windowSumConst,
                                           autoBroadcast)
                              .getResult();
    }

    // Apply final length adjustment if needed (crop or pad to desired output length)
    if (hasSignalLength || olaLength != desiredOutLength) {
        // Extract batch dimensions again for final adjustment (batchDims was moved)
        SmallVector<int64_t> finalBatchDims;
        for (size_t i = 0; i < inShape.size() - 3; ++i) {
            finalBatchDims.push_back(inShape[i]);
        }

        if (olaLength > desiredOutLength) {
            // Crop to desired length
            SmallVector<int64_t> cropShape = finalBatchDims;
            cropShape.push_back(desiredOutLength);
            auto croppedType = mlir::RankedTensorType::get(cropShape, elemType);

            SmallVector<int64_t> cropOffsets(finalBatchDims.size(), 0);

            if (isCenter) {
                // Center crop for centered mode
                int64_t totalCrop = olaLength - desiredOutLength;
                cropOffsets.push_back(totalCrop / 2);
            } else {
                // Crop from beginning for non-centered mode
                cropOffsets.push_back(0);
            }

            SmallVector<int64_t> cropSizes = std::move(finalBatchDims);
            cropSizes.push_back(desiredOutLength);

            finalResult = IE::SliceOp::create(rewriter, appendLoc(loc, "signal_length_crop"), croppedType, finalResult,
                                              getIntArrayAttr(ctx, cropOffsets), getIntArrayAttr(ctx, cropSizes))
                                  .getResult();
        } else if (olaLength < desiredOutLength) {
            // Pad to desired length
            int64_t padAmount = desiredOutLength - olaLength;
            auto padModeAttr = IE::PadModeAttr::get(ctx, IE::PadMode::CONSTANT);

            SmallVector<int64_t> padBefore(finalBatchDims.size(), 0);
            SmallVector<int64_t> padAfter(finalBatchDims.size(), 0);

            if (isCenter) {
                // Center pad for centered mode
                int64_t padBeforeAmount = padAmount / 2;
                int64_t padAfterAmount = padAmount - padBeforeAmount;
                padBefore.push_back(padBeforeAmount);
                padAfter.push_back(padAfterAmount);
            } else {
                // Pad at end for non-centered mode
                padBefore.push_back(0);
                padAfter.push_back(padAmount);
            }

            finalResult =
                    IE::PadOp::create(rewriter, appendLoc(loc, "signal_length_pad"), finalResult, nullptr, nullptr,
                                      nullptr, getIntArrayAttr(ctx, padBefore), getIntArrayAttr(ctx, padAfter),
                                      getFPAttr(ctx, 0.0), padModeAttr, nullptr, nullptr, nullptr, nullptr)
                            .getResult();
        }
    }

    rewriter.replaceOp(origOp, finalResult);
    return mlir::success();
}

mlir::Value ISTFTOpConverter::createHybridOverlapAdd(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                                     mlir::MLIRContext* ctx, mlir::Value frames, int64_t numFrames,
                                                     int64_t frameSizeVal, int64_t frameStepVal, int64_t olaLength,
                                                     mlir::Type elemType, const SmallVector<int64_t>& batchDims,
                                                     bool isCenter) const {
    if (numFrames > 200) {
        _log.trace("Using optimized tiled overlap-add for large case: {0} frames", numFrames);
        return createTiledOverlapAdd(rewriter, loc, ctx, frames, numFrames, frameSizeVal, frameStepVal, olaLength,
                                     elemType, batchDims, isCenter);
    }

    _log.trace("Using explicit frame-by-frame overlap-add for small case: {0} frames", numFrames);
    return createExplicitOverlapAdd(rewriter, loc, ctx, frames, numFrames, frameSizeVal, frameStepVal, olaLength,
                                    elemType, batchDims, isCenter);
}

mlir::Value ISTFTOpConverter::createTiledOverlapAdd(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                                    mlir::MLIRContext* ctx, mlir::Value frames, int64_t numFrames,
                                                    int64_t frameSizeVal, int64_t frameStepVal, int64_t olaLength,
                                                    mlir::Type elemType, const SmallVector<int64_t>& batchDims,
                                                    bool isCenter) const {
    SmallVector<int64_t> outputShape = batchDims;
    outputShape.push_back(olaLength);
    auto outputType = mlir::RankedTensorType::get(outputShape, elemType);
    auto result = Const::createZerosConst(rewriter, appendLoc(loc, "tiled_init_zeros"), outputType);

    // Process frames in chunks of 100 to manage memory usage
    const int64_t chunkSize = 100;
    for (int64_t chunkStart = 0; chunkStart < numFrames; chunkStart += chunkSize) {
        int64_t chunkEnd = std::min(chunkStart + chunkSize, numFrames);
        processChunk(rewriter, loc, ctx, result, frames, chunkStart, chunkEnd, frameSizeVal, frameStepVal, olaLength,
                     elemType, batchDims, isCenter);
    }

    return result;
}

void ISTFTOpConverter::processChunk(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::MLIRContext* ctx,
                                    mlir::Value& result, mlir::Value frames, int64_t chunkStart, int64_t chunkEnd,
                                    int64_t frameSizeVal, int64_t frameStepVal, int64_t olaLength, mlir::Type elemType,
                                    const SmallVector<int64_t>& batchDims, bool isCenter) const {
    auto autoBroadcast = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
    int64_t chunkNumFrames = chunkEnd - chunkStart;
    std::string chunkId = "chunk_" + std::to_string(chunkStart) + "_to_" + std::to_string(chunkEnd);

    // Extract current chunk of frames
    SmallVector<int64_t> chunkSliceOffsets(batchDims.size(), 0);
    chunkSliceOffsets.push_back(chunkStart);
    chunkSliceOffsets.push_back(0);

    SmallVector<int64_t> chunkSliceSizes = batchDims;
    chunkSliceSizes.push_back(chunkNumFrames);
    chunkSliceSizes.push_back(frameSizeVal);

    auto chunkSliceType = mlir::RankedTensorType::get(chunkSliceSizes, elemType);
    auto chunkFrames =
            IE::SliceOp::create(rewriter, appendLoc(loc, "slice_" + chunkId), chunkSliceType, frames,
                                getIntArrayAttr(ctx, chunkSliceOffsets), getIntArrayAttr(ctx, chunkSliceSizes))
                    .getResult();

    int64_t chunkOlaLength = (chunkNumFrames - 1) * frameStepVal + frameSizeVal;

    // Create transformation matrix for vectorized overlap-add
    SmallVector<float> chunkMatrix(chunkNumFrames * frameSizeVal * chunkOlaLength, 0.0f);
    for (int64_t i = 0; i < chunkNumFrames; ++i) {
        int64_t startPos = i * frameStepVal;
        for (int64_t j = 0; j < frameSizeVal; ++j) {
            int64_t outputPos = startPos + j;
            if (outputPos >= 0 && outputPos < chunkOlaLength) {
                int64_t inputIdx = i * frameSizeVal + j;
                chunkMatrix[inputIdx * chunkOlaLength + outputPos] = 1.0f;
            }
        }
    }

    auto chunkMatrixType = mlir::RankedTensorType::get({chunkNumFrames * frameSizeVal, chunkOlaLength}, elemType);
    auto chunkMatrixConst = Const::createConst(rewriter, appendLoc(loc, "matrix_" + chunkId), chunkMatrixType,
                                               ArrayRef<float>{chunkMatrix.data(), chunkMatrix.size()});

    // Flatten chunk for matrix multiplication
    SmallVector<int64_t> flatChunkShape = batchDims;
    flatChunkShape.push_back(chunkNumFrames * frameSizeVal);
    auto flatChunkType = mlir::RankedTensorType::get(flatChunkShape, elemType);
    auto flatChunk = IE::ReshapeOp::create(rewriter, appendLoc(loc, "flatten_chunk"), flatChunkType, chunkFrames,
                                           getIntArrayAttr(ctx, flatChunkShape))
                             .getResult();

    // Perform vectorized overlap-add using MatMul
    SmallVector<int64_t> chunkOutputShape = batchDims;
    chunkOutputShape.push_back(chunkOlaLength);
    auto chunkOutputType = mlir::RankedTensorType::get(chunkOutputShape, elemType);
    auto chunkResult = IE::MatMulOp::create(rewriter, appendLoc(loc, "matmul_" + chunkId), chunkOutputType, flatChunk,
                                            chunkMatrixConst, false, false)
                               .getResult();

    // Calculate global position of chunk in final output
    int64_t globalStartPos = isCenter ? chunkStart * frameStepVal - frameSizeVal / 2 : chunkStart * frameStepVal;

    if (globalStartPos >= 0 && globalStartPos + chunkOlaLength <= olaLength) {
        // Simple case: chunk fits entirely within output bounds
        SmallVector<int64_t> targetOffsets(batchDims.size(), 0);
        targetOffsets.push_back(globalStartPos);
        SmallVector<int64_t> targetSizes = batchDims;
        targetSizes.push_back(chunkOlaLength);

        auto targetSliceType = mlir::RankedTensorType::get(targetSizes, elemType);
        auto targetSlice = IE::SliceOp::create(rewriter, appendLoc(loc, "target_" + chunkId), targetSliceType, result,
                                               getIntArrayAttr(ctx, targetOffsets), getIntArrayAttr(ctx, targetSizes));

        auto updatedSlice = IE::AddOp::create(rewriter, appendLoc(loc, "update_" + chunkId), targetSlice.getResult(),
                                              chunkResult, autoBroadcast, nullptr, nullptr, nullptr, nullptr);

        SmallVector<mlir::Value> concatParts;

        if (globalStartPos > 0) {
            SmallVector<int64_t> beforeOffsets(batchDims.size(), 0);
            beforeOffsets.push_back(0);
            SmallVector<int64_t> beforeSizes = batchDims;
            beforeSizes.push_back(globalStartPos);

            auto beforeSliceType = mlir::RankedTensorType::get(beforeSizes, elemType);
            auto beforeSlice =
                    IE::SliceOp::create(rewriter, appendLoc(loc, "before_" + chunkId), beforeSliceType, result,
                                        getIntArrayAttr(ctx, beforeOffsets), getIntArrayAttr(ctx, beforeSizes));
            concatParts.push_back(beforeSlice.getResult());
        }

        concatParts.push_back(updatedSlice.getResult());

        if (globalStartPos + chunkOlaLength < olaLength) {
            SmallVector<int64_t> afterOffsets(batchDims.size(), 0);
            afterOffsets.push_back(globalStartPos + chunkOlaLength);
            SmallVector<int64_t> afterSizes = batchDims;
            afterSizes.push_back(olaLength - globalStartPos - chunkOlaLength);

            auto afterSliceType = mlir::RankedTensorType::get(afterSizes, elemType);
            auto afterSlice = IE::SliceOp::create(rewriter, appendLoc(loc, "after_" + chunkId), afterSliceType, result,
                                                  getIntArrayAttr(ctx, afterOffsets), getIntArrayAttr(ctx, afterSizes));
            concatParts.push_back(afterSlice.getResult());
        }

        result = IE::ConcatOp::create(rewriter, appendLoc(loc, "rebuild_" + chunkId), mlir::ValueRange(concatParts),
                                      getIntAttr(ctx, batchDims.size()))
                         .getOutput();

    } else {
        // Complex case: chunk extends beyond output bounds, requires cropping
        int64_t validStart = std::max(globalStartPos, int64_t(0));
        int64_t validEnd = std::min(globalStartPos + chunkOlaLength, olaLength);

        if (validStart < validEnd) {
            int64_t validSize = validEnd - validStart;
            int64_t chunkOffset = validStart - globalStartPos;

            SmallVector<int64_t> cropOffsets(batchDims.size(), 0);
            cropOffsets.push_back(chunkOffset);
            SmallVector<int64_t> cropSizes = batchDims;
            cropSizes.push_back(validSize);

            auto croppedChunkType = mlir::RankedTensorType::get(cropSizes, elemType);
            auto croppedChunk =
                    IE::SliceOp::create(rewriter, appendLoc(loc, "crop_" + chunkId), croppedChunkType, chunkResult,
                                        getIntArrayAttr(ctx, cropOffsets), getIntArrayAttr(ctx, cropSizes));

            SmallVector<int64_t> targetOffsets(batchDims.size(), 0);
            targetOffsets.push_back(validStart);
            SmallVector<int64_t> targetSizes = batchDims;
            targetSizes.push_back(validSize);

            auto targetSliceType = mlir::RankedTensorType::get(targetSizes, elemType);
            auto targetSlice =
                    IE::SliceOp::create(rewriter, appendLoc(loc, "target_" + chunkId), targetSliceType, result,
                                        getIntArrayAttr(ctx, targetOffsets), getIntArrayAttr(ctx, targetSizes));

            auto updatedSlice =
                    IE::AddOp::create(rewriter, appendLoc(loc, "update_" + chunkId), targetSlice.getResult(),
                                      croppedChunk.getResult(), autoBroadcast, nullptr, nullptr, nullptr, nullptr);

            SmallVector<mlir::Value> concatParts;

            if (validStart > 0) {
                SmallVector<int64_t> beforeSizes = batchDims;
                beforeSizes.push_back(validStart);
                auto beforeSliceType = mlir::RankedTensorType::get(beforeSizes, elemType);
                auto beforeSlice =
                        IE::SliceOp::create(rewriter, appendLoc(loc, "before_" + chunkId), beforeSliceType, result,
                                            getIntArrayAttr(ctx, SmallVector<int64_t>(batchDims.size() + 1, 0)),
                                            getIntArrayAttr(ctx, beforeSizes));
                concatParts.push_back(beforeSlice.getResult());
            }

            concatParts.push_back(updatedSlice.getResult());

            if (validEnd < olaLength) {
                SmallVector<int64_t> afterOffsets(batchDims.size(), 0);
                afterOffsets.push_back(validEnd);
                SmallVector<int64_t> afterSizes = batchDims;
                afterSizes.push_back(olaLength - validEnd);

                auto afterSliceType = mlir::RankedTensorType::get(afterSizes, elemType);
                auto afterSlice =
                        IE::SliceOp::create(rewriter, appendLoc(loc, "after_" + chunkId), afterSliceType, result,
                                            getIntArrayAttr(ctx, afterOffsets), getIntArrayAttr(ctx, afterSizes));
                concatParts.push_back(afterSlice.getResult());
            }

            result = IE::ConcatOp::create(rewriter, appendLoc(loc, "rebuild_" + chunkId), mlir::ValueRange(concatParts),
                                          getIntAttr(ctx, batchDims.size()))
                             .getOutput();
        }
    }
}

mlir::Value ISTFTOpConverter::createExplicitOverlapAdd(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                                       mlir::MLIRContext* ctx, mlir::Value frames, int64_t numFrames,
                                                       int64_t frameSizeVal, int64_t frameStepVal, int64_t olaLength,
                                                       mlir::Type elemType, const SmallVector<int64_t>& batchDims,
                                                       bool isCenter) const {
    auto autoBroadcast = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);

    size_t numBatchDims = batchDims.size();
    int64_t batchSize = 1;
    for (size_t i = 0; i < numBatchDims; ++i) {
        batchSize *= batchDims[i];
    }

    // Flatten batch dimensions for explicit processing
    SmallVector<int64_t> flatBatchShape = {batchSize, numFrames, frameSizeVal};
    auto flatBatchType = mlir::RankedTensorType::get(flatBatchShape, elemType);
    auto flatBatchFrames = IE::ReshapeOp::create(rewriter, appendLoc(loc, "flatten_batch"), flatBatchType, frames,
                                                 getIntArrayAttr(ctx, flatBatchShape))
                                   .getResult();

    SmallVector<mlir::Value> batchResults;

    for (int64_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        auto batchElement = IE::SliceOp::create(rewriter, appendLoc(loc, "batch_{0}", batchIdx), flatBatchFrames,
                                                getIntArrayAttr(ctx, SmallVector<int64_t>{batchIdx, 0, 0}),
                                                getIntArrayAttr(ctx, SmallVector<int64_t>{1, numFrames, frameSizeVal}));

        SmallVector<int64_t> batchElemShape = {numFrames, frameSizeVal};
        auto batchElemType = mlir::RankedTensorType::get(batchElemShape, elemType);
        auto batchElemReshaped =
                IE::ReshapeOp::create(rewriter, appendLoc(loc, "batch_{0}_reshape", batchIdx), batchElemType,
                                      batchElement.getResult(), getIntArrayAttr(ctx, batchElemShape))
                        .getResult();

        auto olaType = mlir::RankedTensorType::get({olaLength}, elemType);
        mlir::Value output = Const::createZerosConst(rewriter, loc, olaType);

        for (int64_t frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
            auto frameSlice =
                    IE::SliceOp::create(rewriter, appendLoc(loc, "frame_{0}_{1}", batchIdx, frameIdx),
                                        batchElemReshaped, getIntArrayAttr(ctx, SmallVector<int64_t>{frameIdx, 0}),
                                        getIntArrayAttr(ctx, SmallVector<int64_t>{1, frameSizeVal}));

            auto frame1DType = mlir::RankedTensorType::get({frameSizeVal}, elemType);
            auto frame1D = IE::ReshapeOp::create(rewriter, appendLoc(loc, "frame_{0}_{1}_1d", batchIdx, frameIdx),
                                                 frame1DType, frameSlice.getResult(),
                                                 getIntArrayAttr(ctx, SmallVector<int64_t>{frameSizeVal}))
                                   .getResult();

            int64_t frameStart = isCenter ? frameIdx * frameStepVal - frameSizeVal / 2 : frameIdx * frameStepVal;
            int64_t validStart = std::max(frameStart, int64_t(0));
            int64_t validEnd = std::min(frameStart + frameSizeVal, olaLength);

            if (validStart >= validEnd) {
                continue;
            }

            int64_t validSize = validEnd - validStart;
            int64_t frameOffset = validStart - frameStart;

            mlir::Value validFrame = frame1D;
            if (frameOffset > 0 || validSize < frameSizeVal) {
                auto validType = mlir::RankedTensorType::get({validSize}, elemType);
                validFrame =
                        IE::SliceOp::create(rewriter, appendLoc(loc, "valid_{0}_{1}", batchIdx, frameIdx), validType,
                                            frame1D, getIntArrayAttr(ctx, SmallVector<int64_t>{frameOffset}),
                                            getIntArrayAttr(ctx, SmallVector<int64_t>{validSize}))
                                .getResult();
            }

            // Pad frame to output length and accumulate
            auto padModeAttr = IE::PadModeAttr::get(ctx, IE::PadMode::CONSTANT);
            auto paddedFrame =
                    IE::PadOp::create(rewriter, appendLoc(loc, "pad_{0}_{1}", batchIdx, frameIdx), validFrame, nullptr,
                                      nullptr, nullptr, getIntArrayAttr(ctx, SmallVector<int64_t>{validStart}),
                                      getIntArrayAttr(ctx, SmallVector<int64_t>{olaLength - validEnd}),
                                      getFPAttr(ctx, 0.0), padModeAttr, nullptr, nullptr, nullptr, nullptr);

            output = IE::AddOp::create(rewriter, appendLoc(loc, "add_frame_{0}_{1}", batchIdx, frameIdx), output,
                                       paddedFrame.getResult(), autoBroadcast, nullptr, nullptr, nullptr, nullptr)
                             .getOutput();
        }

        batchResults.push_back(output);
    }

    // Reconstruct batch dimensions
    mlir::Value stackedOutput;
    if (batchSize == 1) {
        auto reshapeType = mlir::RankedTensorType::get({1, olaLength}, elemType);
        stackedOutput = IE::ReshapeOp::create(rewriter, appendLoc(loc, "expand_single_batch"), reshapeType,
                                              batchResults[0], getIntArrayAttr(ctx, SmallVector<int64_t>{1, olaLength}))
                                .getResult();
    } else {
        SmallVector<mlir::Value> expandedResults;
        for (int64_t i = 0; i < batchSize; ++i) {
            auto expandedType = mlir::RankedTensorType::get({1, olaLength}, elemType);
            auto expanded =
                    IE::ReshapeOp::create(rewriter, appendLoc(loc, "expand_batch_{0}", i), expandedType,
                                          batchResults[i], getIntArrayAttr(ctx, SmallVector<int64_t>{1, olaLength}))
                            .getResult();
            expandedResults.push_back(expanded);
        }
        stackedOutput = IE::ConcatOp::create(rewriter, appendLoc(loc, "concat_batches"),
                                             mlir::ValueRange(expandedResults), getIntAttr(ctx, 0))
                                .getOutput();
    }

    // Restore original batch shape
    SmallVector<int64_t> preNormalizeShape = batchDims;
    preNormalizeShape.push_back(olaLength);

    auto preNormType = mlir::RankedTensorType::get(preNormalizeShape, elemType);
    return IE::ReshapeOp::create(rewriter, appendLoc(loc, "unflatten_batch"), preNormType, stackedOutput,
                                 getIntArrayAttr(ctx, preNormalizeShape))
            .getResult();
}

//
// DecomposeISTFTPass
//

class DecomposeISTFTPass final : public IE::impl::DecomposeISTFTBase<DecomposeISTFTPass> {
public:
    explicit DecomposeISTFTPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DecomposeISTFTPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::ISTFTOp>();
    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::IRDFTOp>();
    target.addLegalOp<IE::MultiplyOp>();
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<IE::DivideOp>();
    target.addLegalOp<IE::PadOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::MatMulOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ISTFTOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeISTFTPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeISTFTPass(Logger log) {
    return std::make_unique<DecomposeISTFTPass>(log);
}

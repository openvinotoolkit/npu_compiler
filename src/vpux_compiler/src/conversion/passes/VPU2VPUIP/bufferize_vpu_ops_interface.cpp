//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/m2i_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"

using namespace vpux;

namespace {

//
// createCopyResult
//

mlir::OpResult createCopyResult(mlir::Type type, mlir::Value inputBuffer, mlir::Value outputBuffer,
                                mlir::RewriterBase& rewriter, mlir::Location location) {
    if (type == nullptr) {
        return mlir::OpResult();
    }

    auto dataType = type;
    if (auto sparseBuffer = dataType.dyn_cast<VPUIP::SparseBufferType>()) {
        dataType = sparseBuffer.getData();
    }

    if (dataType.isa<mlir::MemRefType>()) {
        auto copyOp = rewriter.create<VPUIP::CopyOp>(location, inputBuffer, outputBuffer);

        return copyOp.getOperation()->getResult(0);
    } else if (dataType.isa<VPUIP::DistributedBufferType>()) {
        // Create NCEClusterTiling with CopyOp inside
        SmallVector<mlir::Value> inputsOutputOperands = {inputBuffer, outputBuffer};

        const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
            builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
        };

        auto clusterTilingOp =
                rewriter.create<VPUIP::NCEClusterTilingOp>(location, type, inputsOutputOperands, bodyBuilder);
        return clusterTilingOp.getOperation()->getResult(0);
    }
    VPUX_THROW("Unexpected data type to copy: {0}", dataType);
}

//
// createSubviewOp
//

mlir::Value createSubviewOp(NDTypeInterface outType, mlir::Value inputBuff, mlir::Location loc,
                            mlir::RewriterBase& rewriter, mlir::ArrayAttr svOffsets, mlir::ArrayAttr svSizes,
                            mlir::ArrayAttr svStrides = nullptr) {
    auto subviewVal = rewriter.create<VPUIP::SubViewOp>(loc, inputBuff, svOffsets, svSizes, svStrides);
    auto subviewType = subviewVal.getType().cast<NDTypeInterface>();

    auto distributedIf = outType.dyn_cast<VPU::DistributedTypeInterface>();
    if (distributedIf == nullptr) {
        return subviewVal;
    }

    auto subviewDistributedIf = subviewType.dyn_cast<VPU::DistributedTypeInterface>();
    VPUX_THROW_WHEN(subviewDistributedIf == nullptr,
                    "Subview output's type should also implement DistributedTypeInterface; it does not = {0}",
                    subviewType);

    if (!distributedIf.containsDistributedTypes()) {
        return subviewVal;
    }

    VPUX_THROW_WHEN(!subviewDistributedIf.containsDistributedTypes(),
                    "Subview output's type should also contain DistributedBufferTypes; it does not = {0}", subviewType);

    auto updateDistribution = [&](VPUIP::DistributedBufferType subviewType,
                                  VPUIP::DistributedBufferType inputDistributedType) -> VPUIP::DistributedBufferType {
        return VPUIP::DistributedBufferType::get(rewriter.getContext(), subviewType.getShape().raw(),
                                                 subviewType.getElementType(), subviewType.getLayout(),
                                                 subviewType.getMemSpace(), inputDistributedType.getDistribution());
    };

    if (auto sparseBuffer = outType.dyn_cast<VPUIP::SparseBufferType>()) {
        auto subviewSparseBuff = subviewType.dyn_cast<VPUIP::SparseBufferType>();
        VPUX_THROW_WHEN(subviewSparseBuff == nullptr, "Subview outputs's type should also be sparse; it is not = {0}",
                        subviewType);

        auto newDataType = updateDistribution(subviewSparseBuff.getData().cast<VPUIP::DistributedBufferType>(),
                                              sparseBuffer.getData().cast<VPUIP::DistributedBufferType>());
        auto newSparseMapType =
                (subviewSparseBuff.getSparsityMap() != nullptr)
                        ? updateDistribution(subviewSparseBuff.getSparsityMap().cast<VPUIP::DistributedBufferType>(),
                                             sparseBuffer.getSparsityMap().cast<VPUIP::DistributedBufferType>())
                        : nullptr;
        auto newSETableType =
                (subviewSparseBuff.getStorageElementTable() != nullptr)
                        ? updateDistribution(
                                  subviewSparseBuff.getStorageElementTable().cast<VPUIP::DistributedBufferType>(),
                                  sparseBuffer.getStorageElementTable().cast<VPUIP::DistributedBufferType>())
                        : nullptr;

        auto newSparseBuffType =
                VPUIP::SparseBufferType::get(newDataType, newSparseMapType, newSETableType, sparseBuffer.getIsWeights(),
                                             sparseBuffer.getSparsityCompression(), sparseBuffer.getSeAttr());

        subviewVal.getResult().setType(newSparseBuffType);
        return subviewVal;
    }

    auto distributedBuffer = distributedIf.getDistributedTypes().front().cast<VPUIP::DistributedBufferType>();
    auto distributedSubview = subviewDistributedIf.getDistributedTypes().front().cast<VPUIP::DistributedBufferType>();
    auto newDistributedType = updateDistribution(distributedSubview, distributedBuffer);

    subviewVal.getResult().setType(newDistributedType);

    return subviewVal;
}

}  // namespace

//
// bufferize VPU::CopyOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::CopyOp origOp, VPU::CopyOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUCopyOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto outputBuffers = allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                         /*individualBuffers =*/false);
    auto newOp = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), newArgs.getInput(), outputBuffers[0]);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::ConvertOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::ConvertOp origOp, VPU::ConvertOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUConvertOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (!isConvertSupportedOnDMA<VPU::ConvertOp>(origOp)) {
        log.trace("VPU::ConvertOp Operation not supported on DMA '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }
    const auto outputBuffers = allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                               /*individualBuffers =*/false);
    auto newOp = rewriter.create<VPUIP::ConvertDMAOp>(origOp->getLoc(), newArgs.getInput(), outputBuffers[0]);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::ExpandOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::ExpandOp origOp, VPU::ExpandOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUExpandOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto outputBuffers = allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                         /*individualBuffers =*/false);
    auto newOp = rewriter.create<VPUIP::ExpandOp>(origOp->getLoc(), newArgs.getInput(), outputBuffers[0],
                                                  origOp.getPadsBegin(), origOp.getPadsEnd());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::StridedSliceOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::StridedSliceOp origOp,
                                      VPU::StridedSliceOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUStridedSliceOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newOutType = vpux::getBufferType(origOp.getType());
    auto outShape = getShape(origOp.getOutput());
    auto outShapeAttr = getIntArrayAttr(rewriter, outShape.raw());
    auto subView = createSubviewOp(newOutType, newArgs.getInput(), origOp->getLoc(), rewriter,
                                   origOp.getBeginsAttrAttr(), outShapeAttr, origOp.getStridesAttrAttr());
    auto outputBuffers = allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                         /*individualBuffers =*/false);
    auto newResult = createCopyResult(newOutType, subView, outputBuffers[0], rewriter, origOp->getLoc());

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newResult);

    return mlir::success();
}

//
// bufferize ReshapeOp
//

template <typename ConcreteOp>
mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, ConcreteOp origOp, typename ConcreteOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUReshapeOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto newOutType = vpux::getBufferType(origOp.getType());
    auto newOp = rewriter.create<VPUIP::GenericReshapeOp>(origOp->getLoc(), newOutType, newArgs.getInput());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::SliceOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::SliceOp origOp, VPU::SliceOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUSliceOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newOutType = vpux::getBufferType(origOp.getType());
    auto subView = createSubviewOp(newOutType, newArgs.getSource(), origOp->getLoc(), rewriter,
                                   origOp.getStaticOffsetsAttr(), origOp.getStaticSizesAttr());
    auto outputBuffers = allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                         /*individualBuffers =*/false);
    auto newResult = createCopyResult(newOutType, subView, outputBuffers[0], rewriter, origOp->getLoc());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newResult);
    return mlir::success();
}

//
// bufferize VPU::SplitOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::SplitOp origOp, VPU::SplitOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUSplitOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (!origOp.getAxisValue().has_value()) {
        return matchFailed(rewriter, origOp, "Got non constant axis");
    }

    const auto inputType = newArgs.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();
    const auto axis = Dim(origOp.getAxisValue().value());
    auto outputBuffers = allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                         /*individualBuffers =*/false);
    // Prepare strides array for subview. We have dense array, so all strides have to be equal 1
    SmallVector<int64_t> svOffsets(inputShape.size(), 0);
    SmallVector<mlir::Value> newResults;
    const auto offsetStep = inputShape[axis] / origOp.getNumSplits();

    for (auto i : irange(origOp->getNumResults())) {
        const auto origOutputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(i).getType());
        const auto newOutputType = vpux::getBufferType(origOutputType);

        const auto svSizes = origOutputType.getShape().raw();

        log.trace("Create SubView for output #'{0}'", i);
        auto subView = createSubviewOp(newOutputType, newArgs.getInput(), origOp->getLoc(), rewriter,
                                       getIntArrayAttr(ctx, svOffsets), getIntArrayAttr(ctx, svSizes));
        log.trace("Copy SubView result to output buffer");
        auto newOp = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), subView, outputBuffers[i]);
        newResults.push_back(newOp.getOutput());

        svOffsets[axis.ind()] += offsetStep;
    }

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newResults);
    return mlir::success();
}

//
// bufferize VPU::ConcatOp
//

namespace {

SmallVector<mlir::Value> rewriteWithAxis(const Logger& log, VPU::ConcatOp origOp, VPU::ConcatOp::Adaptor newArgs,
                                         ArrayRef<mlir::Value> allocatedBufs, mlir::RewriterBase& rewriter) {
    SmallVector<mlir::Value> results;
    auto ctx = origOp->getContext();

    const auto axis = origOp.getPerAxisAttr().getAxis().getValue().getSExtValue();
    const auto offset =
            origOp.getPerAxisAttr().getOffset() ? origOp.getPerAxisAttr().getOffset().getValue().getSExtValue() : 0;
    const auto stride =
            origOp.getPerAxisAttr().getStride() ? origOp.getPerAxisAttr().getStride().getValue().getSExtValue() : 1;

    const auto outputRank = origOp.getType().cast<vpux::NDTypeInterface>().getRank();

    SmallVector<int64_t> svOffsets(outputRank, 0);

    SmallVector<int64_t> svElemStrides;
    if (stride != 1) {
        svElemStrides.resize(outputRank, 1);
        svElemStrides[axis] = stride;
    }

    for (auto i : irange(origOp->getNumOperands())) {
        const auto newInput = newArgs.getInputs()[i];
        const auto newInputType = newInput.getType().cast<vpux::NDTypeInterface>();
        const auto svSizes = newInputType.getShape().raw();

        log.trace("Create SubView for input #'{0}'", i);
        mlir::Value subViewVal;

        auto svOffsetsAttr = getIntArrayAttr(ctx, svOffsets);
        auto svSizesAttr = getIntArrayAttr(ctx, svSizes);
        if (svElemStrides.empty()) {
            subViewVal = createSubviewOp(newInputType, allocatedBufs[0], origOp->getLoc(), rewriter, svOffsetsAttr,
                                         svSizesAttr);
            svOffsets[axis] += svSizes[axis];
        } else {
            auto svElemStridesAttr = getIntArrayAttr(ctx, svElemStrides);
            subViewVal = createSubviewOp(newInputType, allocatedBufs[0], origOp->getLoc(), rewriter, svOffsetsAttr,
                                         svSizesAttr, svElemStridesAttr);
            svOffsets[axis] += offset;
        }

        log.trace("Copy new operand to SubView");

        auto newOutType = subViewVal.getType();

        // Copy to the SubView
        mlir::OpResult newResult = createCopyResult(newOutType, newInput, subViewVal, rewriter, origOp->getLoc());
        results.push_back(newResult);
    }

    return results;
}

SmallVector<mlir::Value> rewriteWithOffsets(const Logger& log, VPU::ConcatOp origOp, VPU::ConcatOp::Adaptor newArgs,
                                            ArrayRef<mlir::Value> allocatedBufs, mlir::RewriterBase& rewriter) {
    SmallVector<mlir::Value> results;

    const auto allOffsets = origOp.getStaticOffsetsAttr().getAsRange<mlir::ArrayAttr>();

    const auto inRank = origOp.getInputs().front().getType().cast<vpux::NDTypeInterface>().getRank();
    const auto dummyStridesAttr = getIntArrayAttr(origOp->getContext(), SmallVector<int64_t>(inRank, 1));
    SmallVector<mlir::ArrayAttr> allStrides(newArgs.getInputs().size(), dummyStridesAttr);

    if (origOp.getStrides().has_value()) {
        allStrides = parseCustomAttrArray<mlir::ArrayAttr>(origOp.getStridesAttr());
    }

    for (const auto p : zip(newArgs.getInputs(), allOffsets, allStrides)) {
        const auto newInput = std::get<0>(p);

        const auto curShape = newInput.getType().cast<vpux::NDTypeInterface>().getShape().raw();
        const auto curOffsets = std::get<1>(p);
        const auto curStrides = origOp.getStrides().has_value() ? std::get<2>(p) : nullptr;

        log.trace("Create SubView");

        auto subviewVal =
                createSubviewOp(newInput.getType().cast<NDTypeInterface>(), allocatedBufs[0], origOp->getLoc(),
                                rewriter, curOffsets, getIntArrayAttr(origOp->getContext(), curShape), curStrides);

        log.trace("Copy new operand to SubView");

        auto newOutType = subviewVal.getType();

        // Copy to the SubView
        mlir::OpResult newResult = createCopyResult(newOutType, newInput, subviewVal, rewriter, origOp->getLoc());
        results.push_back(newResult);
    }

    return results;
}

}  // namespace

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::ConcatOp origOp, VPU::ConcatOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUConcatOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newOutType = vpux::getBufferType(origOp.getResult().getType());
    log.trace("Add Alloc Operations for results");
    auto outputBuffers = allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                         /*individualBuffers =*/false);

    const auto results = origOp.getPerAxisAttr() ? rewriteWithAxis(log, origOp, newArgs, outputBuffers, rewriter)
                                                 : rewriteWithOffsets(log, origOp, newArgs, outputBuffers, rewriter);

    auto newOp = rewriter.create<VPUIP::ConcatViewOp>(origOp->getLoc(), newOutType, results, outputBuffers[0]);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::PermuteCastOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::PermuteCastOp origOp,
                                      VPU::PermuteCastOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUPermuteCastOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto newOutType = vpux::getBufferType(origOp.getType());
    auto newOp = rewriter.create<VPUIP::PermuteCastOp>(origOp->getLoc(), newOutType, newArgs.getInput(),
                                                       origOp.getDstOrderAttr(), origOp.getMemPermAttr());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::QuantizeCastOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::QuantizeCastOp origOp,
                                      VPU::QuantizeCastOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUQuantizeCastOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto newOutType = vpux::getBufferType(origOp.getType());
    auto newOp = rewriter.create<VPUIP::QuantizeCastOp>(origOp->getLoc(), newOutType, newArgs.getInput());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::DistributedCastOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::DistributedCastOp origOp,
                                      VPU::DistributedCastOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUDistributedCastOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto newOutType = vpux::getBufferType(origOp.getType());
    auto newOp = rewriter.create<VPUIP::DistributedCastOp>(origOp->getLoc(), newOutType, newArgs.getInput());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::M2ITaskOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::M2ITaskOp origOp, VPU::M2ITaskOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUM2ITaskOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto outputBuffer = allocateBuffers(log, origOp->getLoc(), rewriter, {origOp.getOutput()},
                                              /*individualBuffers =*/false);
    const auto inputShapeRaw = newArgs.getInput().getType().cast<NDTypeInterface>().getShape().raw();
    const auto outputShapeRaw = outputBuffer[0].getType().cast<NDTypeInterface>().getShape().raw();

    struct M2IDims {
        unsigned w;
        unsigned h;
    };

    const auto getDimsFromRawShape = [](const VPU::M2iColorFmt fmt, const auto& shape) -> M2IDims {
        M2IDims dims;
        if ((fmt == VPU::M2iColorFmt::PL_YUV420_8) || (fmt == VPU::M2iColorFmt::SP_NV12_8)) {
            dims.h = (shape[1] * 2) / 3;
            dims.w = shape[2];
        } else if (fmt == VPU::M2iColorFmt::PL_RGB24 || fmt == VPU::M2iColorFmt::PL_FP16_RGB) {
            dims.h = shape[2];
            dims.w = shape[3];
        } else if (fmt == VPU::M2iColorFmt::IL_RGB888) {
            dims.h = shape[1];
            dims.w = shape[2];
        } else {
            VPUX_THROW("M2iTask currently unsupported format '{0}'", fmt);
        }
        return dims;
    };

    const auto inDims = getDimsFromRawShape(origOp.getInFmt(), inputShapeRaw);
    const auto outDims = getDimsFromRawShape(origOp.getOutFmt(), outputShapeRaw);

    const auto scaleFactorWidth =
            VPU::getM2iFixedPointScaleFactor(inDims.w, outDims.w, VPU::M2I_SCALE_FACTOR_FRACTIONAL_BITS);
    const auto scaleFactorHeight =
            VPU::getM2iFixedPointScaleFactor(inDims.h, outDims.h, VPU::M2I_SCALE_FACTOR_FRACTIONAL_BITS);

    auto m2iOp = rewriter.create<VPUIP::M2ITaskOp>(
            origOp->getLoc(), newArgs.getInput(), outputBuffer[0], nullptr, origOp.getDoCsc(), origOp.getDoNorm(),
            origOp.getInFmt(), origOp.getOutFmt(), origOp.getChromaInReverseChannels(),
            origOp.getChromaOutReverseChannels(), origOp.getLumaInReverseChannels(), origOp.getLumaOutReverseChannels(),
            scaleFactorWidth, scaleFactorHeight, origOp.getNormAttr(), nullptr, nullptr, /*profilingMetadata*/ nullptr,
            origOp.getInterp());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, m2iOp.getOutput());
    return mlir::success();
}

//
// bufferize VPU::StubOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::StubOp origOp, VPU::StubOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUStubOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    SmallVector<mlir::Type> outputTypes;
    for (auto out : origOp.getResults()) {
        outputTypes.push_back(vpux::getBufferType(out.getType()));
    }
    auto newOp = rewriter.create<VPUIP::DistributedCastOp>(origOp->getLoc(), outputTypes, newArgs.getOperands());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::GroupSparseTensorOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::GroupSparseTensorOp origOp,
                                      VPU::GroupSparseTensorOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUGroupSparseTensorOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    VPUIP::SparsityCompressionAttr sparsityCompression = nullptr;
    if (origOp.getSparsityCompressionAttr() != nullptr) {
        auto origCompression = origOp.getSparsityCompressionAttr();
        sparsityCompression =
                VPUIP::SparsityCompressionAttr::get(origCompression.getContext(), origCompression.getAxis(),
                                                    origCompression.getNumElems(), origCompression.getAlignment());
    }
    auto newOp = rewriter.create<VPUIP::GroupSparseBufferOp>(
            origOp->getLoc(), newArgs.getData(), newArgs.getSparsityMap(), newArgs.getStorageElementTable(),
            origOp.getIsWeightsAttr(), sparsityCompression, origOp.getSeAttr().value_or(nullptr));
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::UngroupSparseTensorOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::UngroupSparseTensorOp origOp,
                                      VPU::UngroupSparseTensorOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUUngroupSparseTensorOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newOp = rewriter.create<VPUIP::UngroupSparseBufferOp>(origOp->getLoc(), newArgs.getInput());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::StorageElementTableOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::StorageElementTableOp origOp,
                                      VPU::StorageElementTableOp::Adaptor /*newArgs*/, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUStorageElementTableOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newOp = rewriter.create<VPUIP::StorageElementTableOp>(
            origOp->getLoc(), origOp.getDataShapeAttr(), origOp.getDataElemTypeAttr(), origOp.getSeSizeAttr(),
            origOp.getSeDepthAttr(), origOp.getSeAttrAttr(), origOp.getDataStridesAttr(), origOp.getBasePtrsAttr());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::ShapeCastOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::ShapeCastOp origOp, VPU::ShapeCastOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUShapeCastOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newOp = rewriter.create<VPUIP::ShapeCastOp>(origOp->getLoc(), newArgs.getSource(), newArgs.getShape());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());

    return mlir::success();
}

//
// bufferize VPU::LayoutCastOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::LayoutCastOp origOp, VPU::LayoutCastOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPULayoutCastOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto newOutType = vpux::getBufferType(origOp.getType());
    const auto outOrder = DimsOrder::fromValue(origOp.getOutput());
    const auto outMap = outOrder.toAffineMap(origOp.getContext());
    const auto mapAttr = mlir::AffineMapAttr::get(outMap);

    auto newOp = rewriter.create<VPUIP::PermuteCastOp>(origOp->getLoc(), newOutType, newArgs.getInput(),
                                                       origOp.getDstOrderAttr(), mapAttr);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::GatherDMAOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::GatherDMAOp origOp, VPU::GatherDMAOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUGatherDMAOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto ctx = origOp->getContext();
    const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);

    // Hardware Limitation: In Gather addressing mode, indices must reside in CMX
    // Currently, this implementation only handles GatherDMAOp where the input is in DDR and the output is in CMX
    auto indices = newArgs.getIndices();
    auto indicesCMXBuffers = allocateBuffer(log, origOp.getLoc(), rewriter, indices, memSpaceCMX);
    auto indicesCMXCopy = rewriter.create<VPUIP::CopyOp>(origOp.getLoc(), indices, indicesCMXBuffers);

    auto outputCMXBuffers = allocateBuffer(log, origOp.getLoc(), rewriter, origOp.getOutput(), memSpaceCMX);
    auto newOp = rewriter.create<VPUIP::GatherDMAOp>(origOp.getLoc(), newArgs.getInput(), indicesCMXCopy.getOutput(),
                                                     outputCMXBuffers, 0, 0, 0);
    newOp.setChannelType(VPUIP::DmaChannelType::DDR);

    auto outputDDRBuffers = allocateBuffers(log, origOp.getLoc(), rewriter, origOp->getOpResults(),
                                            /*individualBuffers =*/false);
    auto newResult =
            createCopyResult(newOp.getType(), newOp.getOutput(), outputDDRBuffers[0], rewriter, origOp.getLoc());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newResult);

    return mlir::success();
}

//
// bufferize VPU::WorkloadCastOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::WorkloadCastOp origOp,
                                      VPU::WorkloadCastOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUWorkloadCastOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto newOutType = vpux::getBufferType(origOp.getType());
    auto newOp = rewriter.create<VPUIP::WorkloadCastOp>(origOp->getLoc(), newOutType, newArgs.getInput());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::UpsamplingOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::UpsamplingOp origOp, VPU::UpsamplingOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUUpsamplingOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto outputBuffers = allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                         /*individualBuffers =*/false);
    auto newOp = rewriter.create<VPUIP::UpsamplingOp>(origOp->getLoc(), newArgs.getInput(), outputBuffers[0],
                                                      origOp.getUpsamplingFactorAttr(), origOp.getPadAttr());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::ShapeOfOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::ShapeOfOp origOp, VPU::ShapeOfOp::Adaptor newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUShapeOfOp", 0);

    auto ctx = origOp.getContext();

    const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);

    auto newOperand = newArgs.getOperands()[0];
    VPUX_THROW_UNLESS(mlir::isa<VPUIP::BoundedBufferType>(newOperand.getType()),
                      "Expected to have BoundedBufferType as input to ShapeOf, got: {0}", newOperand.getType());

    auto ungroupInput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(newOperand.getLoc(), newOperand);
    auto shapeValue = ungroupInput.getDynamicShape();
    auto shapeAlloc = allocateBuffer(log, shapeValue.getLoc(), rewriter, shapeValue, memSpaceCMX);

    auto copyOp = rewriter.create<VPUIP::CopyOp>(shapeValue.getLoc(), shapeValue, shapeAlloc);
    SmallVector<mlir::Value> swKernelOperands{copyOp.getOutput()};

    auto origResult = origOp->getResult(0);
    auto outputBuffer = allocateBuffer(log, origResult.getLoc(), rewriter, origResult, memSpaceCMX);
    SmallVector<mlir::Value> swKernelResults{outputBuffer};

    auto op = origOp.getOperation();
    auto module = getModuleOp(op);
    VPUIP::createRuntimeKernelDefinition(module, log.nest(), VPU::getArch(op));

    auto layerOp = mlir::cast<VPU::LayerOpInterface>(op);
    auto swLayerOp = mlir::cast<VPUIP::SoftwareLayerOpInterface>(op);

    auto builtInFunction = VPUIP::createBuiltInFunction(module, layerOp, swKernelOperands, swKernelResults,
                                                        swLayerOp.getKernelInfo(), log.nest());

    const int64_t tileIndex = 0;
    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(origOp.getLoc(), swKernelOperands, swKernelResults,
                                                         builtInFunction, getIntAttr(ctx, tileIndex));

    vpux::VPUIP::initSwKernel(swKernelOp, swKernelOperands, swKernelResults, swLayerOp.getKernelInfo().args,
                              log.nest());

    log.trace("Added kernel operation: {0}", swKernelOp);

    auto newResult = swKernelOp.getResult(0);
    auto resultAlloc = allocateBuffer(log, newResult.getLoc(), rewriter, newResult, nullptr);
    auto resultCopy = rewriter.create<VPUIP::CopyOp>(swKernelOp->getLoc(), newResult, resultAlloc);
    SmallVector<mlir::Value> newResults{resultCopy.getOutput()};

    log.trace("Replace origin op {0} with new outputs from SW Kernel {1}", origOp.getLoc(), newResults);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newResults);
    return mlir::success();
}

//
// registerVPUBufferizableOpInterfaces
//

void vpux::registerVPUBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.insert<Const::ConstDialect>();
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::CopyOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::CopyOp>>(*ctx);
        VPU::ExpandOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::ExpandOp>>(*ctx);
        VPU::StridedSliceOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::StridedSliceOp>>(*ctx);
        VPU::AffineReshapeOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::AffineReshapeOp>>(*ctx);
        VPU::ReshapeOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::ReshapeOp>>(*ctx);
        VPU::SqueezeOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::SqueezeOp>>(*ctx);
        VPU::UnsqueezeOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::UnsqueezeOp>>(*ctx);
        VPU::SliceOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::SliceOp>>(*ctx);
        VPU::SplitOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::SplitOp>>(*ctx);
        VPU::PermuteCastOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::PermuteCastOp>>(*ctx);
        VPU::QuantizeCastOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::QuantizeCastOp>>(*ctx);
        VPU::DistributedCastOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::DistributedCastOp>>(*ctx);
        VPU::M2ITaskOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::M2ITaskOp>>(*ctx);
        VPU::StubOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::StubOp>>(*ctx);
        VPU::GroupSparseTensorOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::GroupSparseTensorOp>>(*ctx);
        VPU::UngroupSparseTensorOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::UngroupSparseTensorOp>>(*ctx);
        VPU::StorageElementTableOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::StorageElementTableOp>>(*ctx);
        VPU::ShapeCastOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::ShapeCastOp>>(*ctx);
        VPU::LayoutCastOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::LayoutCastOp>>(*ctx);
        VPU::WorkloadCastOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::WorkloadCastOp>>(*ctx);
        VPU::UpsamplingOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::UpsamplingOp>>(*ctx);
        VPU::ShapeOfOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::ShapeOfOp>>(*ctx);
    });
}

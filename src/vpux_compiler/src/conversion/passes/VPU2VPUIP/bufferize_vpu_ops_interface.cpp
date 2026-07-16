//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_call_ops_interface.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/allocate_buffers.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

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
    if (auto sparseBuffer = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(dataType)) {
        dataType = sparseBuffer.getData();
    }

    if (mlir::isa<mlir::MemRefType, VPUIP::DistributedBufferType>(dataType)) {
        auto copyOp = rewriter.create<VPUIP::CopyOp>(location, inputBuffer, outputBuffer);

        return copyOp.getOperation()->getResult(0);
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
    auto subviewType = mlir::cast<vpux::NDTypeInterface>(subviewVal.getType());

    auto distributedIf = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(outType);
    if (distributedIf == nullptr) {
        return subviewVal;
    }

    auto subviewDistributedIf = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(subviewType);
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

    if (auto sparseBuffer = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(outType)) {
        auto subviewSparseBuff = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(subviewType);
        VPUX_THROW_WHEN(subviewSparseBuff == nullptr, "Subview outputs's type should also be sparse; it is not = {0}",
                        subviewType);

        auto newDataType =
                updateDistribution(mlir::cast<vpux::VPUIP::DistributedBufferType>(subviewSparseBuff.getData()),
                                   mlir::cast<vpux::VPUIP::DistributedBufferType>(sparseBuffer.getData()));
        auto newSparseMapType =
                (subviewSparseBuff.getSparsityMap() != nullptr)
                        ? updateDistribution(
                                  mlir::cast<vpux::VPUIP::DistributedBufferType>(subviewSparseBuff.getSparsityMap()),
                                  mlir::cast<vpux::VPUIP::DistributedBufferType>(sparseBuffer.getSparsityMap()))
                        : nullptr;
        auto newSETableType =
                (subviewSparseBuff.getStorageElementTable() != nullptr)
                        ? updateDistribution(
                                  mlir::cast<vpux::VPUIP::DistributedBufferType>(
                                          subviewSparseBuff.getStorageElementTable()),
                                  mlir::cast<vpux::VPUIP::DistributedBufferType>(sparseBuffer.getStorageElementTable()))
                        : nullptr;

        auto newSparseBuffType =
                VPUIP::SparseBufferType::get(newDataType, newSparseMapType, newSETableType, sparseBuffer.getIsWeights(),
                                             sparseBuffer.getSparsityCompression(), sparseBuffer.getSeAttr());

        subviewVal.getResult().setType(newSparseBuffType);
        return subviewVal;
    }

    auto distributedBuffer =
            mlir::cast<vpux::VPUIP::DistributedBufferType>(distributedIf.getDistributedTypes().front());
    auto distributedSubview =
            mlir::cast<vpux::VPUIP::DistributedBufferType>(subviewDistributedIf.getDistributedTypes().front());
    auto newDistributedType = updateDistribution(distributedSubview, distributedBuffer);

    subviewVal.getResult().setType(newDistributedType);

    return subviewVal;
}

}  // namespace

//
// bufferize VPU::CopyOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::CopyOp origOp, VPU::CopyOp::Adaptor& newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUCopyOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto outputBuffers = VPUIP::allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                                /*individualBuffers =*/false);
    auto newOp = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), newArgs.getInput(), outputBuffers[0]);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::ConvertOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::ConvertOp origOp, VPU::ConvertOp::Adaptor& newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUConvertOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (!isConvertSupportedOnDMA<VPU::ConvertOp>(origOp)) {
        log.trace("VPU::ConvertOp Operation not supported on DMA '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }
    const auto outputBuffers = VPUIP::allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                                      /*individualBuffers =*/false);
    auto newOp = rewriter.create<VPUIP::ConvertDMAOp>(origOp->getLoc(), newArgs.getInput(), outputBuffers[0]);
    copyLoopAttributes(origOp, newOp);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::ExpandOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::ExpandOp origOp, VPU::ExpandOp::Adaptor& newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUExpandOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto outputBuffers = VPUIP::allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                                /*individualBuffers =*/false);
    auto newOp = rewriter.create<VPUIP::ExpandOp>(origOp->getLoc(), newArgs.getInput(), outputBuffers[0],
                                                  origOp.getPadsBegin(), origOp.getPadsEnd());
    copyLoopAttributes(origOp, newOp);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::StridedSliceOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::StridedSliceOp origOp,
                                      VPU::StridedSliceOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUStridedSliceOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newOutType = vpux::getBufferType(origOp.getType());
    auto outShape = getShape(origOp.getOutput());
    auto outShapeAttr = getIntArrayAttr(rewriter, outShape.raw());
    auto subView = createSubviewOp(newOutType, newArgs.getInput(), origOp->getLoc(), rewriter,
                                   origOp.getBeginsAttrAttr(), outShapeAttr, origOp.getStridesAttrAttr());
    auto outputBuffers = VPUIP::allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                                /*individualBuffers =*/false);
    auto newResult = createCopyResult(newOutType, subView, outputBuffers[0], rewriter, origOp->getLoc());

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newResult);

    return mlir::success();
}

//
// bufferize ReshapeOp
//

template <typename ConcreteOp>
mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, ConcreteOp origOp, typename ConcreteOp::Adaptor& newArgs,
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

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::SliceOp origOp, VPU::SliceOp::Adaptor& newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUSliceOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newOutType = vpux::getBufferType(origOp.getType());
    auto subView = createSubviewOp(newOutType, newArgs.getInput(), origOp->getLoc(), rewriter,
                                   origOp.getStaticOffsetsAttr(), origOp.getStaticSizesAttr());
    auto outputBuffers = VPUIP::allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                                /*individualBuffers =*/false);
    auto newResult = createCopyResult(newOutType, subView, outputBuffers[0], rewriter, origOp->getLoc());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newResult);
    return mlir::success();
}

//
// bufferize VPU::SplitOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::SplitOp origOp, VPU::SplitOp::Adaptor& newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUSplitOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (!origOp.getAxisValue().has_value()) {
        return matchFailed(rewriter, origOp, "Got non constant axis");
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(newArgs.getInput().getType());
    const auto inputShape = inputType.getShape();
    const auto axis = Dim(origOp.getAxisValue().value());
    auto outputBuffers = VPUIP::allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                                /*individualBuffers =*/false);
    // Prepare strides array for subview. We have dense array, so all strides have to be equal 1
    SmallVector<int64_t> svOffsets(inputShape.size(), 0);
    SmallVector<mlir::Value> newResults;

    const auto numSplits = origOp.getNumSplits();
    VPUX_THROW_WHEN(numSplits <= 0, "Invalid number of splits: {0}", numSplits);

    const auto offsetStep = inputShape[axis] / numSplits;

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

SmallVector<mlir::Value> rewriteWithAxis(const Logger& log, VPU::ConcatOp origOp, VPU::ConcatOp::Adaptor& newArgs,
                                         ArrayRef<mlir::Value> allocatedBufs, mlir::RewriterBase& rewriter) {
    SmallVector<mlir::Value> results;
    auto ctx = origOp->getContext();

    const auto axis = origOp.getPerAxisAttr().getAxis().getValue().getSExtValue();
    const auto offset =
            origOp.getPerAxisAttr().getOffset() ? origOp.getPerAxisAttr().getOffset().getValue().getSExtValue() : 0;
    const auto stride =
            origOp.getPerAxisAttr().getStride() ? origOp.getPerAxisAttr().getStride().getValue().getSExtValue() : 1;

    const auto outputRank = mlir::cast<vpux::NDTypeInterface>(origOp.getType()).getRank();

    SmallVector<int64_t> svOffsets(outputRank, 0);

    SmallVector<int64_t> svElemStrides;
    if (stride != 1) {
        svElemStrides.resize(outputRank, 1);
        svElemStrides[axis] = stride;
    }

    for (auto i : irange(origOp->getNumOperands())) {
        const auto newInput = newArgs.getInputs()[i];
        const auto newInputType = mlir::cast<vpux::NDTypeInterface>(newInput.getType());
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

SmallVector<mlir::Value> rewriteWithOffsets(const Logger& log, VPU::ConcatOp origOp, VPU::ConcatOp::Adaptor& newArgs,
                                            ArrayRef<mlir::Value> allocatedBufs, mlir::RewriterBase& rewriter) {
    SmallVector<mlir::Value> results;

    const auto allOffsets = origOp.getStaticOffsetsAttr().getAsRange<mlir::ArrayAttr>();

    const auto inRank = mlir::cast<vpux::NDTypeInterface>(origOp.getInputs().front().getType()).getRank();
    const auto dummyStridesAttr = getIntArrayAttr(origOp->getContext(), SmallVector<int64_t>(inRank, 1));
    SmallVector<mlir::ArrayAttr> allStrides(newArgs.getInputs().size(), dummyStridesAttr);

    if (origOp.getStrides().has_value()) {
        allStrides = parseCustomAttrArray<mlir::ArrayAttr>(origOp.getStridesAttr());
    }

    for (const auto p : zip(newArgs.getInputs(), allOffsets, allStrides)) {
        const auto newInput = std::get<0>(p);

        const auto curShape = mlir::cast<vpux::NDTypeInterface>(newInput.getType()).getShape().raw();
        const auto curOffsets = std::get<1>(p);
        const auto curStrides = origOp.getStrides().has_value() ? std::get<2>(p) : nullptr;

        log.trace("Create SubView");

        auto subviewVal = createSubviewOp(mlir::cast<vpux::NDTypeInterface>(newInput.getType()), allocatedBufs[0],
                                          origOp->getLoc(), rewriter, curOffsets,
                                          getIntArrayAttr(origOp->getContext(), curShape), curStrides);

        log.trace("Copy new operand to SubView");

        auto newOutType = subviewVal.getType();

        // Copy to the SubView
        mlir::OpResult newResult = createCopyResult(newOutType, newInput, subviewVal, rewriter, origOp->getLoc());
        results.push_back(newResult);
    }

    return results;
}

}  // namespace

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::ConcatOp origOp, VPU::ConcatOp::Adaptor& newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUConcatOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newOutType = vpux::getBufferType(origOp.getResult().getType());
    log.trace("Add Alloc Operations for results");
    auto outputBuffers = VPUIP::allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
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
                                      VPU::PermuteCastOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
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
                                      VPU::QuantizeCastOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
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
                                      VPU::DistributedCastOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUDistributedCastOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto newOutType = vpux::getBufferType(origOp.getType());
    auto newOp = rewriter.create<VPUIP::DistributedCastOp>(origOp->getLoc(), newOutType, newArgs.getInput());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::StubOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::StubOp origOp, VPU::StubOp::Adaptor& newArgs,
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
                                      VPU::GroupSparseTensorOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
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
            origOp.getIsWeights(), sparsityCompression, origOp.getSeAttr().value_or(nullptr));
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::UngroupSparseTensorOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::UngroupSparseTensorOp origOp,
                                      VPU::UngroupSparseTensorOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
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
                                      VPU::StorageElementTableOp::Adaptor& /*newArgs*/, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUStorageElementTableOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newOp = rewriter.create<VPUIP::StorageElementTableOp>(
            origOp->getLoc(), origOp.getDataShapeAttr(), origOp.getDataElemTypeAttr(), origOp.getSeSizeAttr(),
            origOp.getSeDepthAttr(), origOp.getSeAttrAttr(), origOp.getDataStridesAttr(), origOp.getBasePtrsAttr());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::ZeroPointTableOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::ZeroPointTableOp origOp, VPU::ZeroPointTableOp::Adaptor&,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUZeroPointTableOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto zeroPointsDataAttr = origOp.getZeroPointTableDataAttr();
    VPUX_THROW_UNLESS(zeroPointsDataAttr, "ZeroPointTableOp at '{0}' has no zeroPointTableData", origOp->getLoc());

    const auto zeroPointsData = parseIntArrayAttr<int32_t>(zeroPointsDataAttr);
    const auto zeroPointsType = mlir::cast<vpux::NDTypeInterface>(origOp.getZeroPoints().getType()).getElementType();
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    // Create constant with the zero-point table data
    // Note: Template parameter (int8_t or uint8_t) controls how data is converted from int32_t,
    // but the constant type is always i8 regardless of signedness. Unsigned values like uint8_t(255)
    // are stored as int8_t(-1) with the same bit pattern (0xFF).
    auto createTensor = [&](auto storageType) {
        using StorageT = decltype(storageType);
        auto zeroPointsTyped = to_small_vector(llvm::map_range(zeroPointsData, [](int32_t val) {
            return static_cast<StorageT>(val);
        }));

        return VPU::createTensorFromTableData<StorageT>(rewriter, origOp->getLoc(), zeroPointsTyped,
                                                        outputType.getShape(), rewriter.getI8Type());
    };

    mlir::Value constOp = zeroPointsType.isSignedInteger() ? createTensor(int8_t{}) : createTensor(uint8_t{});
    VPUX_THROW_WHEN(constOp == nullptr, "Failed to create constant for ZeroPointTableOp at '{0}'", origOp->getLoc());

    rewriter.replaceOp(origOp, constOp);
    return mlir::success();
}

//
// bufferize VPU::DataPointerTableOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::DataPointerTableOp origOp,
                                      VPU::DataPointerTableOp::Adaptor&, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUDataPointerTableOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto dataPointerTableDataAttr = origOp.getDataPointerTableDataAttr();
    VPUX_THROW_UNLESS(dataPointerTableDataAttr, "DataPointerTableOp at '{0}' has no dataPointerTableData",
                      origOp->getLoc());

    auto dataPointerTableData = parseIntArrayAttr<int32_t>(dataPointerTableDataAttr);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto outputShape = outputType.getShape();

    // Create constant with the data-pointer table data
    mlir::Value constOp = VPU::createTensorFromTableData<int32_t>(rewriter, origOp->getLoc(), dataPointerTableData,
                                                                  outputShape, getSInt32Type(rewriter.getContext()));

    rewriter.replaceOp(origOp, constOp);
    return mlir::success();
}

//
// bufferize VPU::ShapeCastOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::ShapeCastOp origOp, VPU::ShapeCastOp::Adaptor& newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUShapeCastOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newOp = rewriter.create<VPUIP::ShapeCastOp>(origOp->getLoc(), newArgs.getInput(), newArgs.getShape());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());

    return mlir::success();
}

//
// bufferize VPU::LayoutCastOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::LayoutCastOp origOp, VPU::LayoutCastOp::Adaptor& newArgs,
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

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::GatherDMAOp origOp, VPU::GatherDMAOp::Adaptor& newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUGatherDMAOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto addressingMode = (origOp.getAddressingMode().has_value() &&
                                 (origOp.getAddressingMode().value() == VPU::GatherAddressingMode::ABSOLUTE))
                                        ? VPUIP::GatherAddressingMode::ABSOLUTE
                                        : VPUIP::GatherAddressingMode::INDEXED;

    if (mlir::isa<VPU::DistributedTensorType>(origOp.getOutput().getType())) {
        auto outputCMXBuffers = VPUIP::allocateBuffers(log, origOp.getLoc(), rewriter, origOp.getOutput(), true);
        auto newOp = rewriter.create<VPUIP::GatherDMAOp>(origOp.getLoc(), newArgs.getInput(), newArgs.getIndices(),
                                                         outputCMXBuffers[0], 0, 0, 0, addressingMode);
        mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp.getResult());
        return mlir::success();
    }

    auto ctx = origOp->getContext();
    const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
    // Hardware Limitation: In Gather addressing mode, indices must reside in CMX
    // Currently, this implementation only handles GatherDMAOp where the input is in DDR and the output is in CMX
    auto indices = newArgs.getIndices();
    auto indicesCMXBuffers = VPUIP::allocateBuffer(log, origOp.getLoc(), rewriter, indices, memSpaceCMX);
    auto indicesCMXCopy = rewriter.create<VPUIP::CopyOp>(origOp.getLoc(), indices, indicesCMXBuffers);

    auto outputCMXBuffers = VPUIP::allocateBuffer(log, origOp.getLoc(), rewriter, origOp.getOutput(), memSpaceCMX);
    auto newOp = rewriter.create<VPUIP::GatherDMAOp>(origOp.getLoc(), newArgs.getInput(), indicesCMXCopy.getOutput(),
                                                     outputCMXBuffers, 0, 0, 0, addressingMode);
    auto outputDDRBuffers = VPUIP::allocateBuffers(log, origOp.getLoc(), rewriter, origOp->getOpResults(),
                                                   /*individualBuffers =*/false);
    auto newResult =
            createCopyResult(newOp.getType(), newOp.getOutput(), outputDDRBuffers[0], rewriter, origOp.getLoc());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newResult);

    return mlir::success();
}

//
// bufferize VPU::UpsamplingOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::UpsamplingOp origOp, VPU::UpsamplingOp::Adaptor& newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUUpsamplingOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto outputBuffers = VPUIP::allocateBuffers(log, origOp->getLoc(), rewriter, origOp->getOpResults(),
                                                /*individualBuffers =*/false);
    auto newOp = rewriter.create<VPUIP::UpsamplingOp>(origOp->getLoc(), newArgs.getInput(), outputBuffers[0],
                                                      origOp.getUpsamplingFactorAttr(), origOp.getPadAttr());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

//
// bufferize VPU::ShapeOfOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::ShapeOfOp origOp, VPU::ShapeOfOp::Adaptor& newArgs,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUShapeOfOp", 0);

    auto ctx = origOp.getContext();

    const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);

    auto newOperand = newArgs.getOperands()[0];
    VPUX_THROW_UNLESS(mlir::isa<VPUIP::BoundedBufferType>(newOperand.getType()),
                      "Expected to have BoundedBufferType as input to ShapeOf, got: {0}", newOperand.getType());

    auto ungroupInput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(newOperand.getLoc(), newOperand);
    auto shapeValue = ungroupInput.getDynamicShape();
    auto shapeAlloc = VPUIP::allocateBuffer(log, shapeValue.getLoc(), rewriter, shapeValue, memSpaceCMX);

    auto copyOp = rewriter.create<VPUIP::CopyOp>(shapeValue.getLoc(), shapeValue, shapeAlloc);
    SmallVector<mlir::Value> swKernelOperands{copyOp.getOutput()};

    auto origResult = origOp->getResult(0);
    auto outputBuffer = VPUIP::allocateBuffer(log, origResult.getLoc(), rewriter, origResult, memSpaceCMX);
    SmallVector<mlir::Value> swKernelResults{outputBuffer};

    auto op = origOp.getOperation();
    auto module = getModuleOp(op);
    VPUIP::createRuntimeKernelDefinition(module, log.nest());

    auto layerOp = mlir::cast<VPU::LayerOpInterface>(op);
    auto swLayerOp = mlir::cast<VPUIP::SoftwareLayerOpInterface>(op);

    auto builtInFunction = VPUIP::createBuiltInFunction(module, layerOp, swKernelOperands, swKernelResults,
                                                        swLayerOp.getKernelInfo(), log.nest());

    const int64_t tileIndex = 0;
    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(origOp.getLoc(), swKernelOperands, swKernelResults,
                                                         builtInFunction, getIntAttr(ctx, tileIndex));

    vpux::VPUIP::initSwKernel(swKernelOp, swKernelOperands, swKernelResults, swLayerOp.getKernelInfo().args, log.nest(),
                              /*swKernelRunOp=*/nullptr);

    log.trace("Added kernel operation: {0}", swKernelOp);

    auto newResult = swKernelOp.getResult(0);
    auto resultAlloc = VPUIP::allocateBuffer(log, newResult.getLoc(), rewriter, newResult, nullptr);
    auto resultCopy = rewriter.create<VPUIP::CopyOp>(swKernelOp->getLoc(), newResult, resultAlloc);
    SmallVector<mlir::Value> newResults{resultCopy.getOutput()};

    log.trace("Replace origin op {0} with new outputs from SW Kernel {1}", origOp.getLoc(), newResults);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newResults);
    return mlir::success();
}

//
// bufferize VPU::EmptyOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, VPU::EmptyOp origOp, VPU::EmptyOp::Adaptor& /*newArgs*/,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-VPUEmptyOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto bufferType = vpux::getBufferType(origOp.getResult());
    auto alloc = VPUIP::allocateBuffersOfType(log, origOp.getLoc(), rewriter, bufferType);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, alloc);
    return mlir::success();
}

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, Core::ReinterpretCastOp origOp,
                                      Core::ReinterpretCastOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-CoreReinterpretCastOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto newOutType = vpux::getBufferType(origOp.getResult().getType());

    // Case: output is BoundedBuffer (e.g. memref<?x...> -> BoundedBuffer<data=memref<N...>, shape=...>).
    // In host compile, at this boundary, dynamic-shape bufferization needs to materialize a VPUIP::BoundedBuffer.
    // We do not want to model that as a single ReinterpretCast from one memref into both data and
    // shape, so emit an explicit data cast, shape alloc, and GroupBoundedBuffer.
    if (auto outBounded = mlir::dyn_cast<VPUIP::BoundedBufferType>(newOutType)) {
        const auto dataType = mlir::dyn_cast<mlir::MemRefType>(outBounded.getData());
        const auto shapeType = mlir::dyn_cast<mlir::MemRefType>(outBounded.getDynamicShape());
        if (dataType == nullptr || shapeType == nullptr) {
            return mlir::failure();
        }
        auto dataCast = rewriter.create<Core::ReinterpretCastOp>(appendLoc(origOp->getLoc(), "_data"), dataType,
                                                                 newArgs.getInput());
        auto shapeAlloc = rewriter.create<mlir::memref::AllocOp>(appendLoc(origOp->getLoc(), "_shape"), shapeType);
        auto grouped = rewriter.create<VPUIP::GroupBoundedBufferOp>(appendLoc(origOp->getLoc(), "_grouped"),
                                                                    dataCast.getResult(), shapeAlloc);
        mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, grouped->getResults());
        return mlir::success();
    }

    // Case: input is BoundedBuffer (e.g. BoundedBuffer<...> -> memref<?x...>)
    if (mlir::isa<VPUIP::BoundedBufferType>(newArgs.getInput().getType())) {
        auto ungroup = rewriter.create<VPUIP::UngroupBoundedBufferOp>(appendLoc(origOp->getLoc(), "_ungroup"),
                                                                      newArgs.getInput());
        auto dataCast = rewriter.create<Core::ReinterpretCastOp>(appendLoc(origOp->getLoc(), "_data"), newOutType,
                                                                 ungroup.getData());
        mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, dataCast->getResults());
        return mlir::success();
    }

    // Default: straightforward memref-to-memref reinterpret cast.
    auto newOp = rewriter.create<Core::ReinterpretCastOp>(origOp->getLoc(), newOutType, newArgs.getInput());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

bool vpux::NestedCallOpBufferizeModel::bufferizesToMemoryReadImpl(
        Core::NestedCallOp op, mlir::OpOperand& opOperand, const mlir::bufferization::AnalysisState& state) const {
    auto funcOp = vpux::getCalledFunction(op);

    if (getFuncOpAnalysisState(state, funcOp) != mlir::bufferization::func_ext::FuncOpAnalysisState::Analyzed) {
        return true;
    }

    const auto& funcState = getFuncOneShotAnalysisState(state);
    return funcState.readBbArgs.lookup(funcOp).contains(opOperand.getOperandNumber());
}

bool vpux::NestedCallOpBufferizeModel::bufferizesToMemoryWriteImpl(
        Core::NestedCallOp op, mlir::OpOperand& opOperand, const mlir::bufferization::AnalysisState& state) const {
    auto funcOp = vpux::getCalledFunction(op);

    if (getFuncOpAnalysisState(state, funcOp) != mlir::bufferization::func_ext::FuncOpAnalysisState::Analyzed) {
        // FuncOp not analyzed yet. Assume that OpOperand is written.
        return true;
    }

    const auto& funcState = getFuncOneShotAnalysisState(state);
    return funcState.writtenBbArgs.lookup(funcOp).contains(opOperand.getOperandNumber());
}

mlir::bufferization::AliasingValueList vpux::NestedCallOpBufferizeModel::getAliasingValuesImpl(
        Core::NestedCallOp op, mlir::OpOperand& opOperand, const mlir::bufferization::AnalysisState& state) const {
    auto funcOp = vpux::getCalledFunction(op);

    if (getFuncOpAnalysisState(state, funcOp) != mlir::bufferization::func_ext::FuncOpAnalysisState::Analyzed) {
        // FuncOp not analyzed yet. Any OpResult may be aliasing.
        return mlir::bufferization::detail::unknownGetAliasingValues(opOperand);  // Note: using 'detail' namespace!
    }

    const auto& funcState = getFuncOneShotAnalysisState(state);
    auto aliasingReturnVals = funcState.aliasingReturnVals.lookup(funcOp).lookup(opOperand.getOperandNumber());

    std::optional<int64_t> equivalent = {};
    if (aliasingReturnVals.size() == 1) {
        equivalent = getEquivalentFuncArgIdx(funcOp, funcState, aliasingReturnVals.front());
        VPUX_THROW_WHEN((equivalent.has_value() && *equivalent != opOperand.getOperandNumber()),
                        "inconsistent analysis state");
    }

    mlir::bufferization::AliasingValueList result;
    for (auto resultIdx : aliasingReturnVals) {
        result.addAlias({op->getOpResult(resultIdx),
                         equivalent.has_value() ? mlir::bufferization::BufferRelation::Equivalent
                                                : mlir::bufferization::BufferRelation::Unknown,
                         /*isDefinite=*/equivalent.has_value()});
    }
    return result;
}

mlir::LogicalResult vpux::NestedCallOpBufferizeModel::bufferizeImpl(
        Core::NestedCallOp op, mlir::RewriterBase& rewriter, const mlir::bufferization::BufferizationOptions& options,
        mlir::bufferization::BufferizationState& state, Core::NestedCallOp::Adaptor&) const {
    auto log = vpux::Logger::global().nest("one-shot-bufferize-CallOp", 0);
    log.trace("Got '{0}' at '{1}'", op->getName(), op->getLoc());

    auto funcOp = getCalledFunction(op);
    auto funcType = funcOp.getFunctionType();

    SmallVector<mlir::Type> resultTypes;
    resultTypes.reserve(op.getNumResults());
    for (auto result : op.getResults()) {
        auto returnType = result.getType();
        if (!mlir::isa<mlir::TensorType>(returnType)) {
            resultTypes.push_back(returnType);
            continue;
        }
        resultTypes.push_back(funcType.getResult(result.getResultNumber()));
    }

    SmallVector<mlir::Value> newOperands;
    newOperands.reserve(op->getOperands().size());
    for (auto& opOperand : op->getOpOperands()) {
        auto maybeBuffer = mlir::bufferization::getBuffer(rewriter, opOperand.get(), options, state);
        VPUX_THROW_WHEN(mlir::failed(maybeBuffer), "Bufferization process failed for operand '{0}'", opOperand.get());

        auto buffer = *maybeBuffer;
        auto memRefType = mlir::cast<mlir::MemRefType>(funcType.getInput(opOperand.getOperandNumber()));
        if (buffer.getType() != memRefType) {
            auto memrefDstType = mlir::cast<mlir::MemRefType>(memRefType);
            mlir::FailureOr<mlir::Value> replacement =
                    mlir::bufferization::castOrReallocMemRefValue(rewriter, buffer, memrefDstType, options);
            if (mlir::failed(replacement)) {
                return mlir::failure();
            }
            buffer = *replacement;
        }
        newOperands.push_back(buffer);
    }

    auto newCallOp = rewriter.create<Core::NestedCallOp>(op.getLoc(), op.getCallee(), resultTypes, newOperands);
    newCallOp->setAttrs(op->getAttrs());

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, newCallOp->getResults());

    return mlir::success();
}

void vpux::registerCoreBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, vpux::Core::CoreDialect*) {
        Core::ReinterpretCastOp::attachInterface<VpuGenericOneShotBufferizeModel<Core::ReinterpretCastOp>>(*ctx);
        Core::NestedCallOp::attachInterface<NestedCallOpBufferizeModel>(*ctx);
    });
}

//
// ConstDialect: bufferize Const::DeclareOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, Const::DeclareOp origOp, Const::DeclareOp::Adaptor&,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-ConstDeclareOp", 0);
    log.trace("Found Constant Operation '{0}'", origOp->getLoc());

    const auto newType = vpux::getBufferType(origOp.getType());
    auto newOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), newType, origOp.getContentAttr());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

void vpux::registerConstDeclareBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, vpux::Const::ConstDialect*) {
        Const::DeclareOp::attachInterface<VpuGenericOneShotBufferizeModel<Const::DeclareOp>>(*ctx);
    });
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
        VPU::StubOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::StubOp>>(*ctx);
        VPU::GroupSparseTensorOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::GroupSparseTensorOp>>(*ctx);
        VPU::UngroupSparseTensorOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::UngroupSparseTensorOp>>(*ctx);
        VPU::StorageElementTableOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::StorageElementTableOp>>(*ctx);
        VPU::DataPointerTableOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::DataPointerTableOp>>(*ctx);
        VPU::ZeroPointTableOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::ZeroPointTableOp>>(*ctx);
        VPU::ShapeCastOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::ShapeCastOp>>(*ctx);
        VPU::LayoutCastOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::LayoutCastOp>>(*ctx);
        VPU::UpsamplingOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::UpsamplingOp>>(*ctx);
        VPU::ShapeOfOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::ShapeOfOp>>(*ctx);
        VPU::EmptyOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::EmptyOp>>(*ctx);
    });
}

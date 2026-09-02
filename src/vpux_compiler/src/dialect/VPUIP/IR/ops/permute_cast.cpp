//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/distributed_buffer_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

//
// build
//

void VPUIP::PermuteCastOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type result,
                                 mlir::Value source, mlir::AffineMapAttr dst_order, mlir::AffineMapAttr mem_perm) {
    build(builder, state, result, source, dst_order, mem_perm, /*is_layout_cast=*/nullptr);
}

void VPUIP::PermuteCastOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type result,
                                 mlir::Value source, mlir::AffineMap dst_order, mlir::AffineMap mem_perm) {
    build(builder, state, result, source, mlir::AffineMapAttr::get(dst_order), mlir::AffineMapAttr::get(mem_perm),
          /*is_layout_cast=*/nullptr);
}

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::PermuteCastOp::getViewSource() {
    return getSource();
}

namespace {

bool isFromLayoutCast(VPUIP::PermuteCastOp op) {
    return op.getIsLayoutCastAttr() != nullptr;
}

bool areCommonPermuteCastTypesCompatible(VPUIP::PermuteCastOp op, vpux::NDTypeInterface inputNDType,
                                         vpux::NDTypeInterface outputNDType, LogCb logCb) {
    if (inputNDType.getNumElements() != outputNDType.getNumElements()) {
        logCb(formatv("PermuteCast input and output must have the same number of elements. inType {0}, outType {1}",
                      inputNDType, outputNDType));
        return false;
    }
    const auto inputRank = inputNDType.getRank();
    const auto outputRank = outputNDType.getRank();
    if (inputRank != outputRank) {
        logCb(formatv("PermuteCast input rank {0} does not match output rank {1}", inputRank, outputRank));
        return false;
    }
    if (inputRank != op.getDstOrder().getNumDims() || inputRank != op.getMemPerm().getNumDims()) {
        if (inputRank != op.getDstOrder().getNumDims()) {
            logCb(formatv("PermuteCast input rank {0} does not match 'dst_order' {1}", inputRank,
                          op.getDstOrder().getNumDims()));
        } else {
            logCb(formatv("PermuteCast input rank {0} does not match 'mem_perm' {1}", inputRank,
                          op.getMemPerm().getNumDims()));
        }
        return false;
    }

    // A pure view cannot invent non-compact strides from a compact input. Keep
    // the historical verifier boundary lenient in the other direction because
    // strided input -> compact output IR can already exist after propagation.
    // LayoutCast-origin back-inference remaps explicit input strides when it
    // constructs a new candidate type.
    const auto compactStrideReqs = StrideReqs::compact(inputRank);
    if (compactStrideReqs.checkStrides(inputNDType) && !compactStrideReqs.checkStrides(outputNDType)) {
        logCb(formatv("Compact input {0} and non-compact output {1} are inconsistent", inputNDType, outputNDType));
        return false;
    }

    return true;
}

//
// LayoutCast-origin PermuteCast
//

bool isSupportedLayoutCastType(VPUIP::PermuteCastOp op, vpux::NDTypeInterface inputNDType,
                               vpux::NDTypeInterface outputNDType, LogCb logCb) {
    // VPU.LayoutCast lowers to VPUIP.PermuteCast with a marker because it has
    // reinterpret-cast semantics: logical shape, element type, memory space and
    // distribution stay fixed, only the dims order changes.
    // Example: memref<1x3x32x32xf16, NWCH, DDR> can become
    // memref<1x3x32x32xf16, NHWC, DDR>; it is not the same rule as a real
    // memory-shape permutation.
    const auto dstOrder = DimsOrder::fromAffineMap(op.getDstOrder());
    if (op.getMemPerm() != op.getDstOrder()) {
        logCb(formatv("LayoutCast-origin PermuteCast must keep 'mem_perm' equal to 'dst_order'. mem_perm {0}, "
                      "dst_order {1}",
                      op.getMemPerm(), op.getDstOrder()));
        return false;
    }
    if (inputNDType.getShape() != outputNDType.getShape()) {
        logCb(formatv("LayoutCast-origin PermuteCast must preserve logical shape. inType {0}, outType {1}", inputNDType,
                      outputNDType));
        return false;
    }
    if (inputNDType.getElementType() != outputNDType.getElementType()) {
        logCb(formatv("LayoutCast-origin PermuteCast must preserve element type. inType {0}, outType {1}", inputNDType,
                      outputNDType));
        return false;
    }
    if (inputNDType.getMemSpace() != outputNDType.getMemSpace()) {
        logCb(formatv("LayoutCast-origin PermuteCast must preserve mem space. inType {0}, outType {1}", inputNDType,
                      outputNDType));
        return false;
    }
    if (outputNDType.getDimsOrder() != dstOrder) {
        logCb(formatv("LayoutCast-origin PermuteCast output order {0} does not match 'dst_order' {1}",
                      outputNDType.getDimsOrder(), dstOrder));
        return false;
    }

    auto inputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(mlir::cast<mlir::Type>(inputNDType));
    auto outputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(mlir::cast<mlir::Type>(outputNDType));
    if (inputDistType != nullptr && outputDistType != nullptr &&
        inputDistType.getDistribution() != outputDistType.getDistribution()) {
        logCb(formatv("LayoutCast-origin PermuteCast must preserve distribution. in = {0}, out = {1}",
                      inputDistType.getDistribution(), outputDistType.getDistribution()));
        return false;
    }

    return true;
}

std::optional<mlir::Type> changeDimsOrderPreservingDistribution(vpux::NDTypeInterface type, DimsOrder order) {
    if (type == nullptr || !type.hasRank() || order.numDims() != checked_cast<size_t>(type.getRank()) ||
        mlir::isa<VPUIP::ITIBufferType>(mlir::cast<mlir::Type>(type))) {
        return std::nullopt;
    }

    const auto rank = checked_cast<size_t>(type.getRank());
    const auto isCompact = StrideReqs::compact(rank).checkStrides(type);
    const auto newStrides = order.toLogicalOrder(type.getMemStrides());
    if (!isCompact) {
        const auto newMemShape = order.toMemoryOrder(type.getShape());
        const auto newMemStrides = order.toMemoryOrder(StridesRef(newStrides));
        if (!StrideReqs().checkStrides(newMemStrides, type.getElemTypeSize(), newMemShape)) {
            return std::nullopt;
        }
    }

    if (auto distBufType = mlir::dyn_cast<VPUIP::DistributedBufferType>(type)) {
        // LayoutCast-origin inference follows VPU.LayoutCast for compact
        // buffers: changing dims order rebuilds a compact layout.
        if (!isCompact) {
            // If propagation provides explicit strides, preserve the same
            // memory strides and express them in the requested logical order.
            // Example: input NWCH logical strides [N, C, H, W] first become
            // memory strides [N, W, C, H], then NHWC maps them back to
            // [N, C, H, W] for the output type.
            auto typeComps = TypeComponents().setDimsOrder(order).setStrides(newStrides);
            return mlir::cast<mlir::Type>(
                    distBufType.changeTypeComponentsForExplicitDistribution(typeComps, distBufType.getDistribution()));
        }

        // Rebuild distributed buffers through the quiet constructor so an
        // impossible rank/order/distribution combination simply fails this
        // candidate.
        auto ctx = distBufType.getContext();
        auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        auto newType = VPUIP::createDistributedBufferTypeOrNull(
                ctx, distBufType.getShape(), distBufType.getElementType(), orderAttr, distBufType.getMemSpace(),
                distBufType.getDistribution());
        if (!newType.has_value()) {
            return std::nullopt;
        }
        return mlir::cast<mlir::Type>(newType.value());
    }

    if (!isCompact) {
        auto typeComps = TypeComponents().setDimsOrder(order).setStrides(newStrides);
        return mlir::cast<mlir::Type>(type.changeTypeComponents(typeComps));
    }

    return mlir::cast<mlir::Type>(type.changeDimsOrder(order));
}

//
// Ordinary PermuteCast
//

bool isSupportedOrdinaryPermuteCastType(VPUIP::PermuteCastOp op, vpux::NDTypeInterface inputNDType,
                                        vpux::NDTypeInterface outputNDType, LogCb logCb) {
    auto inputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(mlir::cast<mlir::Type>(inputNDType));
    auto outputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(mlir::cast<mlir::Type>(outputNDType));
    if (inputDistType == nullptr || outputDistType == nullptr) {
        return true;
    }

    // Ordinary PermuteCast preserves the physical memory and changes the
    // logical view according to mem_perm. For distributed buffers the same
    // memory permutation must be applied to num_tiles/alignment/explicit
    // distribution metadata, otherwise per-cluster views would no longer
    // describe the result buffer.
    auto expectedOutputDistribution = VPU::applyPermutationOnDistributionInfoAttr(
            inputDistType, op.getMemPerm(), inputDistType.getDimsOrder(), outputDistType.getDimsOrder(),
            inputDistType.getShape(), outputDistType.getShape());
    if (mlir::failed(expectedOutputDistribution)) {
        logCb(formatv("PermuteCast unsupported input distribution: in = {0}", inputDistType.getDistribution()));
        return false;
    }

    const auto outputDistribution = outputDistType.getDistribution();
    if (outputDistribution != expectedOutputDistribution.value()) {
        logCb(formatv("PermuteCast input and output distributions are incompatible: in = {0}, out = {1},"
                      "expected = {2}",
                      inputDistType.getDistribution(), outputDistribution, expectedOutputDistribution.value()));
        return false;
    }

    return true;
}

std::optional<mlir::Type> inferPermuteCastOutput(VPUIP::PermuteCastOp op, vpux::NDTypeInterface newInputNDType) {
    // Ordinary PermuteCast keeps the memory shape relationship. Example with
    // compact NCHW input 1x3x32x32 and mem_perm NWHC: compute the output memory
    // shape by applying mem_perm, then convert it back through dst_order to get
    // the output logical shape.
    const auto memPerm = op.getMemPerm();
    const auto dstOrder = DimsOrder::fromAffineMap(op.getDstOrder());
    if (newInputNDType == nullptr || !newInputNDType.hasRank() ||
        dstOrder.numDims() != checked_cast<size_t>(newInputNDType.getRank()) ||
        mlir::isa<VPUIP::ITIBufferType>(mlir::cast<mlir::Type>(newInputNDType))) {
        return std::nullopt;
    }

    const auto inferredOutputNDType = vpux::inferNewTypeWithMemPerm(newInputNDType, memPerm, dstOrder);
    const auto inOrder = newInputNDType.getDimsOrder();
    const auto inShape = newInputNDType.getShape();
    const auto outShape = inferredOutputNDType.getShape();
    const auto elemType = inferredOutputNDType.getElementType();

    if (auto distBufType = mlir::dyn_cast<VPUIP::DistributedBufferType>(newInputNDType)) {
        auto newDistAttrOrFailure = VPU::applyPermutationOnDistributionInfoAttr(distBufType, memPerm, inOrder, dstOrder,
                                                                                inShape, ShapeRef(outShape));
        if (mlir::failed(newDistAttrOrFailure)) {
            return std::nullopt;
        }
        auto ctx = op.getContext();
        auto orderAttr = mlir::AffineMapAttr::get(dstOrder.toAffineMap(ctx));
        auto outType = VPUIP::createDistributedBufferTypeOrNull(
                ctx, ShapeRef(outShape), elemType, orderAttr, distBufType.getMemSpace(), newDistAttrOrFailure.value());
        if (!outType.has_value()) {
            return std::nullopt;
        }
        return mlir::cast<mlir::Type>(outType.value());
    }

    return mlir::cast<mlir::Type>(inferredOutputNDType);
}

std::optional<mlir::Type> inferPermuteCastInputElementType(VPUIP::PermuteCastOp op,
                                                           vpux::NDTypeInterface desiredOutputNDType,
                                                           vpux::NDTypeInterface origInputNDType) {
    // Reverse the per-axis quantization remap used by forward inference.
    // Example: if output scales are on logical C after mem_perm, map that axis
    // back through inverse memory permutation to the original input logical dim.
    auto elemType = desiredOutputNDType.getElementType();
    const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType);
    if (perAxisType == nullptr) {
        return elemType;
    }

    const auto outAxis = perAxisType.getQuantizedDimension();
    if (outAxis < 0 || outAxis >= desiredOutputNDType.getRank()) {
        return std::nullopt;
    }
    const auto outMemAxis = desiredOutputNDType.getDimsOrder().dimPos(Dim(outAxis));
    const auto memPermOrder = DimsOrder::fromAffineMap(op.getMemPerm());
    if (outMemAxis >= memPermOrder.numDims()) {
        return std::nullopt;
    }
    const auto inMemAxis = memPermOrder.dimAt(outMemAxis).ind();
    if (inMemAxis < 0 || checked_cast<size_t>(inMemAxis) >= origInputNDType.getDimsOrder().numDims()) {
        return std::nullopt;
    }
    const auto inAxis = origInputNDType.getDimsOrder().dimAt(inMemAxis);
    if (inAxis.ind() < 0 || inAxis.ind() >= origInputNDType.getRank()) {
        return std::nullopt;
    }
    elemType = changeAxis(perAxisType, checked_cast<int32_t>(inAxis.ind()));

    if (!vpux::isSupportedElemTypeQuantization(elemType, origInputNDType.getShape())) {
        return std::nullopt;
    }
    return elemType;
}

std::optional<mlir::Type> reverseInferPermuteCastInput(VPUIP::PermuteCastOp op,
                                                       vpux::NDTypeInterface desiredOutputNDType) {
    auto origInputNDType = mlir::cast<vpux::NDTypeInterface>(op.getSource().getType());
    auto origOutputNDType = mlir::cast<vpux::NDTypeInterface>(op.getResult().getType());
    if (origInputNDType.getNumElements() != desiredOutputNDType.getNumElements()) {
        return std::nullopt;
    }

    const auto inputElemType = inferPermuteCastInputElementType(op, desiredOutputNDType, origInputNDType);
    if (!inputElemType.has_value()) {
        return std::nullopt;
    }

    if (auto desiredOutputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(desiredOutputNDType)) {
        auto inversePerm = mlir::inversePermutation(op.getMemPerm());
        auto newDistAttrOrFailure = VPU::applyPermutationOnDistributionInfoAttr(
                desiredOutputDistType, inversePerm, origOutputNDType.getDimsOrder(), origInputNDType.getDimsOrder(),
                origOutputNDType.getShape(), origInputNDType.getShape());
        if (mlir::failed(newDistAttrOrFailure)) {
            return std::nullopt;
        }

        auto ctx = op.getContext();
        auto orderAttr = mlir::AffineMapAttr::get(origInputNDType.getDimsOrder().toAffineMap(ctx));
        auto inType = VPUIP::createDistributedBufferTypeOrNull(ctx, origInputNDType.getShape(), inputElemType.value(),
                                                               orderAttr, desiredOutputDistType.getMemSpace(),
                                                               newDistAttrOrFailure.value());
        if (!inType.has_value()) {
            return std::nullopt;
        }
        return mlir::cast<mlir::Type>(inType.value());
    }

    if (!desiredOutputNDType.hasRank() ||
        origInputNDType.getDimsOrder().numDims() != checked_cast<size_t>(desiredOutputNDType.getRank()) ||
        mlir::isa<VPUIP::ITIBufferType>(mlir::cast<mlir::Type>(desiredOutputNDType))) {
        return std::nullopt;
    }
    auto inType = desiredOutputNDType.changeShapeElemType(origInputNDType.getShape(), inputElemType.value());
    inType = inType.changeDimsOrder(origInputNDType.getDimsOrder());
    return mlir::cast<mlir::Type>(inType);
}

//
// PermuteCast kind dispatch
//

bool shouldInferAsLayoutCast(VPUIP::PermuteCastOp op) {
    if (isFromLayoutCast(op)) {
        return true;
    }

    // Keep old IR and recreated ops working when the type relation still
    // unambiguously matches LayoutCast semantics. Prefer ordinary PermuteCast
    // when both interpretations reproduce the original result.
    auto origInputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(op.getSource().getType());
    if (origInputNDType == nullptr) {
        return false;
    }

    const auto permuteCastOutputType = inferPermuteCastOutput(op, origInputNDType);
    if (permuteCastOutputType.has_value() && permuteCastOutputType.value() == op.getResult().getType()) {
        return false;
    }

    const auto layoutCastOutputType =
            changeDimsOrderPreservingDistribution(origInputNDType, DimsOrder::fromAffineMap(op.getDstOrder()));
    return layoutCastOutputType.has_value() && layoutCastOutputType.value() == op.getResult().getType();
}

bool isSupportedPermuteCastType(VPUIP::PermuteCastOp op, mlir::Type inputType, mlir::Type outputType,
                                LogCb logCb = emptyLogCb) {
    auto inputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(inputType);
    auto outputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(outputType);
    if (inputNDType == nullptr || outputNDType == nullptr) {
        logCb(formatv("PermuteCast input and output must be ND types. inType {0}, outType {1}", inputType, outputType));
        return false;
    }

    if (!areCommonPermuteCastTypesCompatible(op, inputNDType, outputNDType, logCb)) {
        return false;
    }

    return isFromLayoutCast(op) ? isSupportedLayoutCastType(op, inputNDType, outputNDType, logCb)
                                : isSupportedOrdinaryPermuteCastType(op, inputNDType, outputNDType, logCb);
}

}  // namespace

//
// BackInferViewTypeOpInterface
//

std::optional<mlir::Type> VPUIP::PermuteCastOp::inferOutputTypeFromInput(mlir::Type newInputType) {
    auto newInputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(newInputType);
    if (newInputNDType == nullptr) {
        return std::nullopt;
    }
    const auto outputType =
            shouldInferAsLayoutCast(*this)
                    ? changeDimsOrderPreservingDistribution(newInputNDType, DimsOrder::fromAffineMap(getDstOrder()))
                    : inferPermuteCastOutput(*this, newInputNDType);
    if (!outputType.has_value() || !isSupportedPermuteCastType(*this, newInputType, outputType.value())) {
        return std::nullopt;
    }
    return outputType;
}

std::optional<mlir::Type> VPUIP::PermuteCastOp::inferInputTypeFromOutput(mlir::Type desiredOutputType) {
    auto desiredOutputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(desiredOutputType);
    if (desiredOutputNDType == nullptr) {
        return std::nullopt;
    }
    auto origInputNDType = mlir::cast<vpux::NDTypeInterface>(getSource().getType());
    const auto inputType = shouldInferAsLayoutCast(*this) ? changeDimsOrderPreservingDistribution(
                                                                    desiredOutputNDType, origInputNDType.getDimsOrder())
                                                          : reverseInferPermuteCastInput(*this, desiredOutputNDType);
    if (!inputType.has_value() || !isSupportedPermuteCastType(*this, inputType.value(), desiredOutputType)) {
        return std::nullopt;
    }
    return inputType;
}

//
// fold
//

mlir::OpFoldResult vpux::VPUIP::PermuteCastOp::fold(FoldAdaptor adaptor) {
    if (getSource().getType() == getResult().getType() && getMemPerm().isIdentity()) {
        return getSource();
    }

    if (auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getSource())) {
        if (isFromLayoutCast(*this)) {
            return attr.transform().layoutCast(DimsOrder::fromAffineMap(getDstOrder())).get();
        }

        // This is a fallback solution. In some cases we get VPUIP::PermuteCastOps that should not
        // be allowed. However, the verifier doesn't check this.
        // TODO: #-141102 Remove this fallback solution as soon as the correct verifier is implemented.
        mlir::SmallVector<mlir::Type> inferredReturnTypes;
        VPU::inferPermuteReturnTypes(getSource(), getMemPerm(), getDstOrder(), inferredReturnTypes);
        if (inferredReturnTypes.front() != getResult().getType()) {
            auto restored = static_cast<Const::ContentAttr>(attr);
            if (restored.getType().getShape() != getShape(getResult())) {
                restored = restored.transform().reshape(getShape(getResult())).get();
            }
            return restored.transform().reorder(DimsOrder::fromAffineMap(getDstOrder())).get();
        }

        // PermuteCastOp ensures that it is always a trivial permutation. That's why we can just add MemPermuteAttr
        // which will not perform any data movements.
        auto result =
                attr.transform()
                        .memPermute(DimsOrder::fromAffineMap(getDstOrder()), DimsOrder::fromAffineMap(getMemPerm()))
                        .get();
        return result;
    }

    return nullptr;
}

mlir::LogicalResult vpux::VPUIP::PermuteCastOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };
    return mlir::success(isSupportedPermuteCastType(*this, getSource().getType(), getResult().getType(), logCb));
}

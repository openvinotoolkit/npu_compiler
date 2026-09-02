//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/Utils/StaticValueUtils.h>

using namespace vpux;

//
// DPUWorkloadOp::verify
//

mlir::LogicalResult vpux::VPU::DPUWorkloadOp::verify() {
    // Although ParentOneOf allows scf.for/scf.if as the immediate parent (for dynamic
    // workload materialization), DPU.Workload must still reside inside an NCE op's
    // workloads region. Walk ancestors to confirm.
    auto* current = getOperation()->getParentOp();
    while (current != nullptr) {
        if (mlir::isa<VPU::NCEOpInterface>(current)) {
            return mlir::success();
        }
        current = current->getParentOp();
    }
    return emitOpError("must be nested inside an NCE op's workloads region, but no "
                       "NCE op ancestor found");
}

//
// DPUTaskOp
//

void vpux::VPU::DPUWorkloadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                     mlir::ArrayRef<mlir::OpFoldResult> outOffsets,
                                     mlir::ArrayRef<mlir::OpFoldResult> outSizes,
                                     mlir::ArrayRef<mlir::OpFoldResult> pad, VPU::MPEMode mpeMode) {
    SmallVector<int64_t> staticOffsets, staticSizes, staticPad;
    SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicPad;
    mlir::dispatchIndexOpFoldResults(outOffsets, dynamicOffsets, staticOffsets);
    mlir::dispatchIndexOpFoldResults(outSizes, dynamicSizes, staticSizes);
    mlir::dispatchIndexOpFoldResults(pad, dynamicPad, staticPad);

    build(builder, state, dynamicOffsets, dynamicSizes, staticOffsets, staticSizes, /*inOffsets=*/mlir::ValueRange{},
          /*inSizes=*/mlir::ValueRange{},
          /*inStaticOffsets=*/nullptr, /*inStaticSizes=*/nullptr, mpeMode, dynamicPad, staticPad,
          /*cluster_id=*/nullptr);
}

void vpux::VPU::DPUWorkloadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                     mlir::ArrayRef<mlir::OpFoldResult> outOffsets,
                                     mlir::ArrayRef<mlir::OpFoldResult> outSizes,
                                     mlir::ArrayRef<mlir::OpFoldResult> pad, VPU::MPEModeAttr mpeMode,
                                     mlir::IntegerAttr clusterId) {
    SmallVector<int64_t> staticOffsets, staticSizes, staticPad;
    SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicPad;
    mlir::dispatchIndexOpFoldResults(outOffsets, dynamicOffsets, staticOffsets);
    mlir::dispatchIndexOpFoldResults(outSizes, dynamicSizes, staticSizes);
    mlir::dispatchIndexOpFoldResults(pad, dynamicPad, staticPad);

    auto* ctx = builder.getContext();

    build(builder, state, dynamicOffsets, dynamicSizes, mlir::DenseI64ArrayAttr::get(ctx, staticOffsets),
          mlir::DenseI64ArrayAttr::get(ctx, staticSizes), /*inOffsets=*/mlir::ValueRange{},
          /*inSizes=*/mlir::ValueRange{},
          /*inStaticOffsets=*/nullptr, /*inStaticSizes=*/nullptr, mpeMode, dynamicPad,
          mlir::DenseI64ArrayAttr::get(ctx, staticPad), clusterId);
}

void vpux::VPU::DPUWorkloadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                     mlir::DenseI64ArrayAttr outOffsets, mlir::DenseI64ArrayAttr outSizes,
                                     VPU::PaddingAttr pad, VPU::MPEModeAttr mpeMode, mlir::IntegerAttr clusterId) {
    SmallVector<int64_t> padAttr = {pad.getLeft().getInt(), pad.getRight().getInt(), pad.getTop().getInt(),
                                    pad.getBottom().getInt()};
    build(builder, state, /* outOffsets */ mlir::ValueRange{}, /* outSizes */ mlir::ValueRange{}, outOffsets, outSizes,
          /*inOffsets=*/mlir::ValueRange{},
          /*inSizes=*/mlir::ValueRange{},
          /*inStaticOffsets=*/nullptr, /*inStaticSizes=*/nullptr, mpeMode, /* pad */ mlir::ValueRange{},
          mlir::DenseI64ArrayAttr::get(builder.getContext(), padAttr), clusterId);
}

void vpux::VPU::DPUWorkloadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                     mlir::DenseI64ArrayAttr outOffsets, mlir::DenseI64ArrayAttr outSizes,
                                     VPU::PaddingAttr pad, VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId) {
    SmallVector<int64_t> padAttr = {pad.getLeft().getInt(), pad.getRight().getInt(), pad.getTop().getInt(),
                                    pad.getBottom().getInt()};
    build(builder, state, /* outOffsets */ mlir::ValueRange{}, /* outSizes */ mlir::ValueRange{}, outOffsets, outSizes,
          /*inOffsets=*/mlir::ValueRange{},
          /*inSizes=*/mlir::ValueRange{},
          /*inStaticOffsets=*/nullptr, /*inStaticSizes=*/nullptr, mpeMode, /* pad */ mlir::ValueRange{},
          mlir::DenseI64ArrayAttr::get(builder.getContext(), padAttr), clusterId);
}

void vpux::VPU::DPUWorkloadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                     mlir::DenseI64ArrayAttr outOffsets, mlir::DenseI64ArrayAttr outSizes,
                                     mlir::DenseI64ArrayAttr inOffsets, mlir::DenseI64ArrayAttr inSizes,
                                     VPU::PaddingAttr pad, VPU::MPEModeAttr mpeMode, mlir::IntegerAttr clusterId) {
    SmallVector<int64_t> padAttr = {pad.getLeft().getInt(), pad.getRight().getInt(), pad.getTop().getInt(),
                                    pad.getBottom().getInt()};
    build(builder, state, /* outOffsets */ mlir::ValueRange{}, /* outSizes */ mlir::ValueRange{}, outOffsets, outSizes,
          /*inOffsets=*/mlir::ValueRange{},
          /*inSizes=*/mlir::ValueRange{}, inOffsets, inSizes, mpeMode, /* pad */ mlir::ValueRange{},
          mlir::DenseI64ArrayAttr::get(builder.getContext(), padAttr), clusterId);
}

void vpux::VPU::DPUWorkloadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                     mlir::ArrayRef<mlir::OpFoldResult> outOffsets,
                                     mlir::ArrayRef<mlir::OpFoldResult> outSizes,
                                     mlir::ArrayRef<mlir::OpFoldResult> inOffsets,
                                     mlir::ArrayRef<mlir::OpFoldResult> inSizes, mlir::ArrayRef<mlir::OpFoldResult> pad,
                                     VPU::MPEModeAttr mpeMode, mlir::IntegerAttr clusterId) {
    SmallVector<int64_t> staticOutOffsets, staticOutSizes, staticInOffsets, staticInSizes, staticPad;
    SmallVector<mlir::Value> dynamicOutOffsets, dynamicOutSizes, dynamicInOffsets, dynamicInSizes, dynamicPad;
    mlir::dispatchIndexOpFoldResults(outOffsets, dynamicOutOffsets, staticOutOffsets);
    mlir::dispatchIndexOpFoldResults(outSizes, dynamicOutSizes, staticOutSizes);
    mlir::dispatchIndexOpFoldResults(inOffsets, dynamicInOffsets, staticInOffsets);
    mlir::dispatchIndexOpFoldResults(inSizes, dynamicInSizes, staticInSizes);
    mlir::dispatchIndexOpFoldResults(pad, dynamicPad, staticPad);

    auto* ctx = builder.getContext();
    build(builder, state, dynamicOutOffsets, dynamicOutSizes, mlir::DenseI64ArrayAttr::get(ctx, staticOutOffsets),
          mlir::DenseI64ArrayAttr::get(ctx, staticOutSizes), dynamicInOffsets, dynamicInSizes,
          mlir::DenseI64ArrayAttr::get(ctx, staticInOffsets), mlir::DenseI64ArrayAttr::get(ctx, staticInSizes), mpeMode,
          dynamicPad, mlir::DenseI64ArrayAttr::get(ctx, staticPad), clusterId);
}

/*
 * Return the padding attribute of the DPUWorkloadOp by extracting the static padding values and constructing a new
 * PaddingAttr.
 */
VPU::PaddingAttr vpux::VPU::DPUWorkloadOp::getPadAttribute() {
    auto staticPad = getStaticPad();
    return VPU::getPaddingAttr(getContext(), staticPad[0], staticPad[1], staticPad[2], staticPad[3]);
}

/*
 * Return the mixed output offsets by combining the static and dynamic output offsets into a single SmallVector of
 * OpFoldResults.
 */
SmallVector<mlir::OpFoldResult> vpux::VPU::DPUWorkloadOp::getMixedOutputOffsets() {
    mlir::Builder builder(getContext());
    return mlir::getMixedValues(getStaticOutOffsets(), getOutOffsets(), builder);
}

/*
 * Return the mixed output sizes by combining the static and dynamic output sizes into a single SmallVector of
 * OpFoldResults.
 */
SmallVector<mlir::OpFoldResult> vpux::VPU::DPUWorkloadOp::getMixedOutputSizes() {
    mlir::Builder builder(getContext());
    return mlir::getMixedValues(getStaticOutSizes(), getOutSizes(), builder);
}

/*
 * Return the mixed input offsets by combining the static and dynamic input offsets into a single SmallVector of
 * OpFoldResults. If there are no static input offsets, return std::nullopt.
 */
std::optional<SmallVector<mlir::OpFoldResult>> vpux::VPU::DPUWorkloadOp::getMixedInputOffsets() {
    if (getStaticInOffsetsAttr() == nullptr) {
        return std::nullopt;
    }

    mlir::Builder builder(getContext());
    return mlir::getMixedValues(getStaticInOffsetsAttr(), getInOffsets(), builder);
}

/*
 * Return the mixed input sizes by combining the static and dynamic input sizes into a single SmallVector of
 * OpFoldResults. If there are no static input sizes, return std::nullopt.
 */
std::optional<SmallVector<mlir::OpFoldResult>> vpux::VPU::DPUWorkloadOp::getMixedInputSizes() {
    if (getStaticInSizesAttr() == nullptr) {
        return std::nullopt;
    }

    mlir::Builder builder(getContext());
    return mlir::getMixedValues(getStaticInSizesAttr(), getInSizes(), builder);
}

/*
 * Return the constant output offsets by extracting the constant values from the mixed output offsets.
 */
SmallVector<int64_t> vpux::VPU::DPUWorkloadOp::getConstOutputOffsets() {
    auto wlOffsets = mlir::getConstantIntValues(getMixedOutputOffsets());
    VPUX_THROW_WHEN(!wlOffsets.has_value(), "Cannot get constant output offsets from DPUWorkloadOp '{0}'", getLoc());
    return wlOffsets.value();
}

/*
 * Return the constant output sizes by extracting the constant values from the mixed output sizes.
 */
SmallVector<int64_t> vpux::VPU::DPUWorkloadOp::getConstOutputSizes() {
    auto wlSizes = mlir::getConstantIntValues(getMixedOutputSizes());
    VPUX_THROW_WHEN(!wlSizes.has_value(), "Cannot get constant output sizes from DPUWorkloadOp '{0}'", getLoc());
    return wlSizes.value();
}

/*
 * Return the constant input offsets by extracting the constant values from the mixed input offsets.
 * If there are no constant input offsets, return std::nullopt.
 */
std::optional<SmallVector<int64_t>> vpux::VPU::DPUWorkloadOp::getConstInputOffsets() {
    auto mixedInOffsets = getMixedInputOffsets();
    if (!mixedInOffsets.has_value()) {
        return std::nullopt;
    }

    auto wlOffsets = mlir::getConstantIntValues(mixedInOffsets.value());
    VPUX_THROW_WHEN(!wlOffsets.has_value(), "Cannot get constant input offsets from DPUWorkloadOp '{0}'", getLoc());
    return wlOffsets.value();
}

/*
 * Return the constant input sizes by extracting the constant values from the mixed input sizes.
 * If there are no constant input sizes, return std::nullopt.
 */
std::optional<SmallVector<int64_t>> vpux::VPU::DPUWorkloadOp::getConstInputSizes() {
    auto mixedInSizes = getMixedInputSizes();
    if (!mixedInSizes.has_value()) {
        return std::nullopt;
    }

    auto wlSizes = mlir::getConstantIntValues(mixedInSizes.value());
    VPUX_THROW_WHEN(!wlSizes.has_value(), "Cannot get constant input sizes from DPUWorkloadOp '{0}'", getLoc());
    return wlSizes.value();
}

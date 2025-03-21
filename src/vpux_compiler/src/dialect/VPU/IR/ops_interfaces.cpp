//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

//
// LayerOpInterface
//

mlir::LogicalResult vpux::VPU::verifyLayer(mlir::Operation* op) {
    if (VPU::verifyOpLayout(op).failed()) {
        return mlir::failure();
    }

    return IE::verifyLayer(op);
}

//
// SparseOpInterface
//

bool vpux::VPU::supportsSparseInputs(mlir::Operation* op) {
    const auto compressedInput = [](mlir::Value operand) {
        auto inputShape = operand.getType().cast<vpux::NDTypeInterface>().getShape();
        if (inputShape[Dims4D::Act::C] < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) {
            return true;
        }
        return false;
    };
    if (compressedInput(op->getOperand(0))) {
        return false;
    }
    if (mlir::isa<VPU::NCEEltwiseOp>(op) && compressedInput(op->getOperand(1))) {
        return false;
    }

    if (auto sparseOp = mlir::dyn_cast<VPU::SparseOpInterface>(op)) {
        return VPU::bitEnumContainsAny(sparseOp.sparsitySupport(), VPU::SparsitySupport::SPARSE_INPUTS);
    }
    return false;
}

bool vpux::VPU::supportsSparseOutputs(mlir::Operation* op) {
    if (auto sparseOp = mlir::dyn_cast<VPU::SparseOpInterface>(op)) {
        return VPU::bitEnumContainsAny(sparseOp.sparsitySupport(), VPU::SparsitySupport::SPARSE_OUTPUTS);
    }
    return false;
}

bool vpux::VPU::supportsSparseWeights(mlir::Operation* op) {
    if (auto sparseOp = mlir::dyn_cast<VPU::SparseOpInterface>(op)) {
        return VPU::bitEnumContainsAny(sparseOp.sparsitySupport(), VPU::SparsitySupport::SPARSE_WEIGHTS);
    }
    return false;
}

bool vpux::VPU::supportsSparseData(mlir::Operation* op) {
    return supportsSparseInputs(op) && supportsSparseOutputs(op);
}

mlir::LogicalResult vpux::VPU::details::validateWorkloadsRegion(mlir::Location loc, mlir::Region& workloads) {
    for (auto& workloadOp : workloads.getOps()) {
        if (!mlir::isa<DPUWorkloadOp>(workloadOp)) {
            return errorAt(loc, "Got unsupported Operation '{0}' in 'workloads' region", workloadOp.getName());
        }
    }

    return mlir::success();
}

mlir::Operation* vpux::VPU::details::addWorkload(mlir::Region& workloads, mlir::OpBuilder& builder, mlir::Location loc,
                                                 ShapeRef offsets, ShapeRef sizes, PaddingAttr pad, MPEMode mpeMode,
                                                 mlir::IntegerAttr clusterId) {
    if (workloads.empty()) {
        workloads.emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&workloads.front());

    const auto offsetsAttr = getIntArrayAttr(builder.getContext(), offsets);
    const auto sizesAttr = getIntArrayAttr(builder.getContext(), sizes);

    return builder.create<DPUWorkloadOp>(loc, offsetsAttr, sizesAttr, pad, mpeMode, clusterId);
}

mlir::LogicalResult vpux::VPU::details::verifyInputTypeOp(mlir::Operation* op, vpux::NDTypeInterface inputType) {
    // TODO: #123810 split verifier into arch-specific parts
    bool supportsInputActCompression = false;
    auto alignment = VPU::NCEInvariant::getAlignment(inputType.getElementType());
    if (mlir::isa<VPU::NCECompressConvolutionOp>(op)) {
        alignment = vpux::VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM;
        supportsInputActCompression = true;
    }

    return mlir::success(
            vpux::VPU::NCEInvariant::isInputActTypeSupported(inputType, alignment, supportsInputActCompression));
}

//
// TilingBuilderOpInterface
//

mlir::Value vpux::VPU::makeTile(mlir::OpBuilder& builder, mlir::Location baseLoc, mlir::Value origVal,
                                const TileInfo& tile, StringRef valName) {
    if (tile.shape == getShape(origVal)) {
        return origVal;
    }

    const auto loc = appendLoc(baseLoc, "{0} tile {1}", valName, tile.offsets);

    auto sliceOp = builder.create<VPU::SliceOp>(loc, origVal, tile.offsets, tile.shape);
    return sliceOp.getResult();
}

//
// TilingInfoOpInterface
//

mlir::LogicalResult vpux::VPU::verifyTilingInfo(mlir::Operation* op) {
    if (!mlir::isa<VPU::TilingBuilderOpInterface>(op)) {
        return errorAt(op, "Operation '{0}' provides TilingInfoOpInterface, but not TilingBuilderOpInterface",
                       op->getName());
    }

    if (op->getNumResults() != 1) {
        return errorAt(op, "Unsupported operation '{0}', it must have one and only one result", op->getName());
    }

    return mlir::success();
}

//
// EltwiseOp
//

mlir::LogicalResult vpux::VPU::verifyEltwiseOp(mlir::Operation* op) {
    if (!mlir::isa<VPU::LayerOpInterface>(op)) {
        return errorAt(op, "EltwiseOp trait is applied to non layer operation");
    }

    if (op->getNumResults() != 1) {
        return errorAt(op, "Operation with multiple results can't be EltwiseOp");
    }

    if (op->hasAttr("auto_broadcast")) {
        auto autoBroadcast = op->getAttr("auto_broadcast").dyn_cast<IE::AutoBroadcastTypeAttr>();
        if (autoBroadcast == nullptr) {
            return errorAt(op, "Auto broadcast attribute cannot be cast");
        }
        auto broadcast = autoBroadcast.getValue();

        SmallVector<ArrayRef<int64_t>> inputShapes;
        for (auto operand : op->getOperands()) {
            const auto shape = operand.getType().cast<vpux::NDTypeInterface>().getShape().raw();
            inputShapes.push_back(shape);
        }

        const auto outputShape = IE::broadcastEltwiseShape(inputShapes, broadcast, op->getLoc());

        if (mlir::failed(outputShape)) {
            return errorAt(op, "Eltwise inputs cannot be broadcast");
        }
    }

    return mlir::success();
}

//
// NCEOpInterface
//

mlir::LogicalResult vpux::VPU::verifyNCEOp(mlir::Operation* op) {
    if (!mlir::isa<VPU::NCEOpInterface>(op)) {
        return errorAt(op, "Operation '{0}' is not NCE", op->getName());
    }

    auto nceOp = mlir::cast<VPU::NCEOpInterface>(op);
    if (vpux::VPU::details::validateWorkloadsRegion(nceOp->getLoc(), nceOp.getWorkloads()).failed()) {
        return mlir::failure();
    }

    auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op);
    if (!iface) {
        return errorAt(op, "NCE Operation '{0}' must attach AlignedChannelsOpInterface", op->getName());
    }

    return iface.verifyChannels();
}

//
// isPureViewLike
//

bool vpux::VPU::isPureViewOp(mlir::Operation* op) {
    return mlir::isa<VPU::ViewLikeOpInterface, mlir::ViewLikeOpInterface, vpux::MultiViewOpInterface,
                     vpux::GroupedViewOpInterface, VPU::GroupedViewLikeOpInterface>(op);
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/ops_interfaces.cpp.inc>

//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_tiling_interface_utils.hpp"

#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>

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
        auto inputShape = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getShape();
        return inputShape[Dims4D::Act::C] < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
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
    auto alignedOp = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op);
    if (alignedOp == nullptr) {
        return mlir::success();
    }

    // TODO: #123810 split verifier into arch-specific parts
    bool supportsInputActCompression = false;
    auto alignment = alignedOp.getInputChannelAlignment();
    if (mlir::isa<VPU::NCECompressConvolutionOp>(op)) {
        alignment = vpux::VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM;
        supportsInputActCompression = true;
    }

    return mlir::success(
            vpux::VPU::NCEInvariant::isInputActTypeSupported(inputType, alignment, supportsInputActCompression));
}

mlir::LogicalResult vpux::VPU::details::verifyInputQuantization(mlir::Operation* op) {
    for (auto operand : op->getOperands()) {
        auto inputType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
        auto elemType = inputType.getElementType();
        if (auto inputQType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
            bool is16BitsQuantization = inputQType.getStorageType().isInteger(16);
            if (is16BitsQuantization) {
                return mlir::failure();
            }
        }
    }
    return mlir::success();
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
        auto autoBroadcast = mlir::dyn_cast<vpux::IE::AutoBroadcastTypeAttr>(op->getAttr("auto_broadcast"));
        if (autoBroadcast == nullptr) {
            return errorAt(op, "Auto broadcast attribute cannot be cast");
        }
        auto broadcast = autoBroadcast.getValue();

        SmallVector<ArrayRef<int64_t>> inputShapes;
        for (auto operand : op->getOperands()) {
            const auto shape = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getShape().raw();
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

    if (vpux::VPU::details::verifyInputQuantization(op).failed()) {
        return errorAt(op, "Invalid quantization type");
    }
    return iface.verifyChannels();
}

//
// ClusteredOpInterface
//

vpux::NDTypeInterface vpux::VPU::getDistributedTypeForOpOperand(mlir::Operation* op, mlir::OpOperand& operand,
                                                                bool hasExplicitDistributedAttr,
                                                                SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return nullptr;
    }

    if (mlir::isa<VPU::SWOpInterface>(op)) {
        return getSwDistributedTypeForOpOperand(clusteredOp, operand, siblingsAnalysis, hasExplicitDistributedAttr);
    }

    VPUX_THROW("Can't generate distributed-type for operand");
}

vpux::NDTypeInterface vpux::VPU::getDistributedTypeForOpResult(mlir::Operation* op, mlir::Value result,
                                                               VPU::MultiClusterStrategy strategy,
                                                               SiblingOpsAnalysis& siblingsAnalysis,
                                                               bool hasExplicitDistributedAttr) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return nullptr;
    }

    return getDistributedOutputTensorType(clusteredOp, mlir::cast<vpux::NDTypeInterface>(result.getType()),
                                          siblingsAnalysis, strategy, hasExplicitDistributedAttr);
}

bool vpux::VPU::supportSwOpLoweringAsDMA(mlir::Operation* op) {
    VPUX_THROW_UNLESS(mlir::isa_and_nonnull<VPU::SWOpInterface>(op), "Unexpected operation type");
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<VPU::ConvertOp>([](auto convertOp) {
                return isConvertSupportedOnDMA<VPU::ConvertOp>(convertOp);
            })
            .Default([](mlir::Operation*) {
                return false;
            });
}

//
// isPureViewLike
//

bool vpux::VPU::isPureViewOp(mlir::Operation* op) {
    return mlir::isa<VPU::ViewLikeOpInterface, mlir::ViewLikeOpInterface, vpux::MultiViewOpInterface,
                     vpux::GroupedViewOpInterface, VPU::GroupedViewLikeOpInterface>(op);
}

//
// TilingInfoOpInterface for SW
//

template <class MainOpType>
class SwLayerTilingInfoOpModel final :
        public SwLayerTilingInfoOpModelBase<SwLayerTilingInfoOpModel<MainOpType>, MainOpType> {};

// Register all tiling-supported SW op here
// Common interface for all archKinds
void vpux::VPU::registerSWTilingInfoOpInterfaceCommon(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::ConvolutionOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ConvolutionOp>>(*ctx);
        VPU::GroupConvolutionOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GroupConvolutionOp>>(*ctx);
        VPU::MaxPoolOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MaxPoolOp>>(*ctx);
        VPU::AddOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AddOp>>(*ctx);
        VPU::SubtractOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SubtractOp>>(*ctx);
        VPU::AndOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AndOp>>(*ctx);
        VPU::InterpolateOp::attachInterface<SwLayerTilingInfoOpModel<VPU::InterpolateOp>>(*ctx);
        VPU::MatMulOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MatMulOp>>(*ctx);
        VPU::FakeQuantizeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::FakeQuantizeOp>>(*ctx);
        VPU::FakeConvertOp::attachInterface<SwLayerTilingInfoOpModel<VPU::FakeConvertOp>>(*ctx);
        VPU::QuantizeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::QuantizeOp>>(*ctx);
        VPU::DequantizeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DequantizeOp>>(*ctx);
        VPU::DynamicQuantizeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DynamicQuantizeOp>>(*ctx);
        VPU::DynamicDequantizeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DynamicDequantizeOp>>(*ctx);
        VPU::GatherOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GatherOp>>(*ctx);
        VPU::GatherElementsOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GatherElementsOp>>(*ctx);
        VPU::GridSampleOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GridSampleOp>>(*ctx);
        VPU::GatherDMAOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GatherDMAOp>>(*ctx);
        VPU::GatherNDOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GatherNDOp>>(*ctx);
        VPU::MaxPool8Op::attachInterface<SwLayerTilingInfoOpModel<VPU::MaxPool8Op>>(*ctx);
        VPU::ConvertOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ConvertOp>>(*ctx);
        VPU::SigmoidOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SigmoidOp>>(*ctx);
        VPU::HSwishOp::attachInterface<SwLayerTilingInfoOpModel<VPU::HSwishOp>>(*ctx);
        VPU::HSigmoidOp::attachInterface<SwLayerTilingInfoOpModel<VPU::HSigmoidOp>>(*ctx);
        VPU::LeakyReluOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LeakyReluOp>>(*ctx);
        VPU::PReluOp::attachInterface<SwLayerTilingInfoOpModel<VPU::PReluOp>>(*ctx);
        VPU::MishOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MishOp>>(*ctx);
        VPU::EluOp::attachInterface<SwLayerTilingInfoOpModel<VPU::EluOp>>(*ctx);
        VPU::ClampOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ClampOp>>(*ctx);
        VPU::ReLUOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReLUOp>>(*ctx);
        VPU::SqrtOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SqrtOp>>(*ctx);
        VPU::ExpOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ExpOp>>(*ctx);
        VPU::TanhOp::attachInterface<SwLayerTilingInfoOpModel<VPU::TanhOp>>(*ctx);
        VPU::DivideOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DivideOp>>(*ctx);
        VPU::FloorOp::attachInterface<SwLayerTilingInfoOpModel<VPU::FloorOp>>(*ctx);
        VPU::MemPermuteOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MemPermuteOp>>(*ctx);
        VPU::AvgPoolOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AvgPoolOp>>(*ctx);
        VPU::AvgPool16Op::attachInterface<SwLayerTilingInfoOpModel<VPU::AvgPool16Op>>(*ctx);
        VPU::PermuteQuantizeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::PermuteQuantizeOp>>(*ctx);
        VPU::LogOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LogOp>>(*ctx);
        VPU::PowerOp::attachInterface<SwLayerTilingInfoOpModel<VPU::PowerOp>>(*ctx);
        VPU::FloorModOp::attachInterface<SwLayerTilingInfoOpModel<VPU::FloorModOp>>(*ctx);
        VPU::ModOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ModOp>>(*ctx);
        VPU::EqualOp::attachInterface<SwLayerTilingInfoOpModel<VPU::EqualOp>>(*ctx);
        VPU::LessOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LessOp>>(*ctx);
        VPU::LessEqualOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LessEqualOp>>(*ctx);
        VPU::NotEqualOp::attachInterface<SwLayerTilingInfoOpModel<VPU::NotEqualOp>>(*ctx);
        VPU::GreaterOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GreaterOp>>(*ctx);
        VPU::GreaterEqualOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GreaterEqualOp>>(*ctx);
        VPU::LogicalOrOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LogicalOrOp>>(*ctx);
        VPU::LogicalXorOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LogicalXorOp>>(*ctx);
        VPU::LogicalNotOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LogicalNotOp>>(*ctx);
        VPU::BitwiseAndOp::attachInterface<SwLayerTilingInfoOpModel<VPU::BitwiseAndOp>>(*ctx);
        VPU::BitwiseOrOp::attachInterface<SwLayerTilingInfoOpModel<VPU::BitwiseOrOp>>(*ctx);
        VPU::BitwiseXorOp::attachInterface<SwLayerTilingInfoOpModel<VPU::BitwiseXorOp>>(*ctx);
        VPU::BitwiseNotOp::attachInterface<SwLayerTilingInfoOpModel<VPU::BitwiseNotOp>>(*ctx);
        VPU::RoundOp::attachInterface<SwLayerTilingInfoOpModel<VPU::RoundOp>>(*ctx);
        VPU::SelectOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SelectOp>>(*ctx);
        VPU::ErfOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ErfOp>>(*ctx);
        VPU::DetectionOutputDecodeBoxesOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DetectionOutputDecodeBoxesOp>>(
                *ctx);
        VPU::DetectionOutputNmsCaffeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DetectionOutputNmsCaffeOp>>(*ctx);
        VPU::DetectionOutputSortOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DetectionOutputSortOp>>(*ctx);
        VPU::SinOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SinOp>>(*ctx);
        VPU::SinhOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SinhOp>>(*ctx);
        VPU::SignOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SignOp>>(*ctx);
        VPU::CoshOp::attachInterface<SwLayerTilingInfoOpModel<VPU::CoshOp>>(*ctx);
        VPU::TanOp::attachInterface<SwLayerTilingInfoOpModel<VPU::TanOp>>(*ctx);
        VPU::ReduceL1Op::attachInterface<SwLayerTilingInfoOpModel<VPU::ReduceL1Op>>(*ctx);
        VPU::ReduceL2Op::attachInterface<SwLayerTilingInfoOpModel<VPU::ReduceL2Op>>(*ctx);
        VPU::ReduceLogicalAndOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReduceLogicalAndOp>>(*ctx);
        VPU::ReduceLogicalOrOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReduceLogicalOrOp>>(*ctx);
        VPU::ReduceMaxOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReduceMaxOp>>(*ctx);
        VPU::ReduceMeanOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReduceMeanOp>>(*ctx);
        VPU::ReduceMinOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReduceMinOp>>(*ctx);
        VPU::ReduceProdOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReduceProdOp>>(*ctx);
        VPU::ReduceSumOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReduceSumOp>>(*ctx);
        VPU::ReduceMeanSquareOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReduceMeanSquareOp>>(*ctx);
        VPU::ReverseOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReverseOp>>(*ctx);
        VPU::ReverseSequenceOp::attachInterface<SwLayerTilingInfoOpModel<VPU::ReverseSequenceOp>>(*ctx);
        VPU::SwishOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SwishOp>>(*ctx);
        VPU::NegativeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::NegativeOp>>(*ctx);
        VPU::CeilingOp::attachInterface<SwLayerTilingInfoOpModel<VPU::CeilingOp>>(*ctx);
        VPU::AbsOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AbsOp>>(*ctx);
        VPU::SoftMaxOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SoftMaxOp>>(*ctx);
        VPU::LogSoftmaxOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LogSoftmaxOp>>(*ctx);
        VPU::TopKOp::attachInterface<SwLayerTilingInfoOpModel<VPU::TopKOp>>(*ctx);
        VPU::StridedSliceOp::attachInterface<SwLayerTilingInfoOpModel<VPU::StridedSliceOp>>(*ctx);
        VPU::SpaceToDepthOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SpaceToDepthOp>>(*ctx);
        VPU::DepthToSpaceOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DepthToSpaceOp>>(*ctx);
        VPU::TileOp::attachInterface<SwLayerTilingInfoOpModel<VPU::TileOp>>(*ctx);
        VPU::DynamicTileOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DynamicTileOp>>(*ctx);
        VPU::NormalizeL2Op::attachInterface<SwLayerTilingInfoOpModel<VPU::NormalizeL2Op>>(*ctx);
        VPU::YuvToRgbOp::attachInterface<SwLayerTilingInfoOpModel<VPU::YuvToRgbOp>>(*ctx);
        VPU::SquaredDifferenceOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SquaredDifferenceOp>>(*ctx);
        VPU::GeluOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GeluOp>>(*ctx);
        VPU::GRUSequenceOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GRUSequenceOp>>(*ctx);
        VPU::GRUSequenceLastPartOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GRUSequenceLastPartOp>>(*ctx);
        VPU::SoftPlusOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SoftPlusOp>>(*ctx);
        VPU::MVNOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MVNOp>>(*ctx);
        VPU::MVN1MeanVarOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MVN1MeanVarOp>>(*ctx);
        VPU::MVN6Op::attachInterface<SwLayerTilingInfoOpModel<VPU::MVN6Op>>(*ctx);
        VPU::DFTOp::attachInterface<SwLayerTilingInfoOpModel<VPU::DFTOp>>(*ctx);
        VPU::RDFTUncutOp::attachInterface<SwLayerTilingInfoOpModel<VPU::RDFTUncutOp>>(*ctx);
        VPU::IDFTOp::attachInterface<SwLayerTilingInfoOpModel<VPU::IDFTOp>>(*ctx);
        VPU::IRDFTLastAxisOp::attachInterface<SwLayerTilingInfoOpModel<VPU::IRDFTLastAxisOp>>(*ctx);
        VPU::HardSigmoidOp::attachInterface<SwLayerTilingInfoOpModel<VPU::HardSigmoidOp>>(*ctx);
        VPU::MaximumOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MaximumOp>>(*ctx);
        VPU::MinimumOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MinimumOp>>(*ctx);
        VPU::PadOp::attachInterface<SwLayerTilingInfoOpModel<VPU::PadOp>>(*ctx);
        VPU::AccumulateOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AccumulateOp>>(*ctx);
        VPU::RMSOp::attachInterface<SwLayerTilingInfoOpModel<VPU::RMSOp>>(*ctx);
        VPU::SDPAOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SDPAOp>>(*ctx);
        VPU::SDPAExtendedOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SDPAExtendedOp>>(*ctx);
        VPU::RandomUniformOp::attachInterface<SwLayerTilingInfoOpModel<VPU::RandomUniformOp>>(*ctx);
        VPU::AcoshOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AcoshOp>>(*ctx);
        VPU::AcosOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AcosOp>>(*ctx);
        VPU::AsinhOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AsinhOp>>(*ctx);
        VPU::AsinOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AsinOp>>(*ctx);
        VPU::AtanhOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AtanhOp>>(*ctx);
        VPU::AtanOp::attachInterface<SwLayerTilingInfoOpModel<VPU::AtanOp>>(*ctx);
        VPU::SeluOp::attachInterface<SwLayerTilingInfoOpModel<VPU::SeluOp>>(*ctx);
        VPU::CosOp::attachInterface<SwLayerTilingInfoOpModel<VPU::CosOp>>(*ctx);
        VPU::GRUGatesOp::attachInterface<SwLayerTilingInfoOpModel<VPU::GRUGatesOp>>(*ctx);
        VPU::LSTMGatesOp::attachInterface<SwLayerTilingInfoOpModel<VPU::LSTMGatesOp>>(*ctx);
        VPU::RollOp::attachInterface<SwLayerTilingInfoOpModel<VPU::RollOp>>(*ctx);
        VPU::MVN1NormalizeOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MVN1NormalizeOp>>(*ctx);
        VPU::RoPEOp::attachInterface<SwLayerTilingInfoOpModel<VPU::RoPEOp>>(*ctx);
        VPU::CumSumOp::attachInterface<SwLayerTilingInfoOpModel<VPU::CumSumOp>>(*ctx);
        VPU::MultiplyOp::attachInterface<SwLayerTilingInfoOpModel<VPU::MultiplyOp>>(*ctx);
        VPU::FlashSDPAOp::attachInterface<SwLayerTilingInfoOpModel<VPU::FlashSDPAOp>>(*ctx);
    });
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/ops_interfaces.cpp.inc>

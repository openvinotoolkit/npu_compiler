//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include <vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp>
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>

using namespace vpux;

//
// LayerOpInterface
//

mlir::LogicalResult vpux::IE::verifyLayer(mlir::Operation* op) {
    if (op->getOperands().empty()) {
        return errorAt(op, "Layer Operation has no operands");
    }
    if (op->getResults().empty()) {
        return errorAt(op, "Layer Operation has no results");
    }

    const auto verifyType = [&](mlir::Type type, StringRef name, unsigned ind) {
        if (mlir::isa<mlir::MemRefType>(type)) {
            return errorAt(op, "Layer Operation has MemRef {0} #{1}", name, ind);
        }

        if (auto mainType = mlir::dyn_cast<vpux::NDTypeInterface>(type)) {
            if (validateQuantElemType(op->getLoc(), mainType).failed()) {
                return mlir::failure();
            }
        }

        return mlir::success();
    };

    for (auto& arg : op->getOpOperands()) {
        if (verifyType(arg.get().getType(), "operand", arg.getOperandNumber()).failed()) {
            return mlir::failure();
        }
    }
    for (auto res : op->getOpResults()) {
        if (verifyType(res.getType(), "result", res.getResultNumber()).failed()) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::inferTensorTypes(InferTypeComponentsCb componentsCb, mlir::MLIRContext* ctx,
                                               std::optional<mlir::Location> loc, mlir::ValueRange operands,
                                               mlir::DictionaryAttr attrs, mlir::OpaqueProperties props,
                                               mlir::RegionRange regions, SmallVectorImpl<mlir::Type>& inferredTypes) {
    SmallVector<mlir::ShapedTypeComponents> components;
    if (mlir::failed(componentsCb(ctx, loc, operands, attrs, props, regions, components))) {
        return mlir::failure();
    }

    for (const auto& desc : components) {
        VPUX_THROW_UNLESS(desc.hasRank(), "Unranked TensorType is not supported");

        const auto type = mlir::RankedTensorType::get(desc.getDims(), desc.getElementType(), desc.getAttribute());
        inferredTypes.push_back(type);
    }

    return mlir::success();
}

//
// LayerWithPostOpInterface
//

IE::PostOpAttr vpux::IE::attributizePostOp(mlir::Operation* postOp) {
    return llvm::TypeSwitch<mlir::Operation*, IE::PostOpAttr>(postOp)
            .Case<IE::ReLUOp>([](auto reluOp) {
                return IE::ReluAttr::get(reluOp.getContext());
            })
            .Case<IE::ClampOp>([](auto clampOp) {
                return IE::ClampAttr::get(clampOp.getContext(), clampOp.getMinAttr(), clampOp.getMaxAttr());
            })
            .Case<IE::LeakyReluOp>([](auto leakyReluOp) {
                return IE::LeakyReluAttr::get(leakyReluOp.getContext(), leakyReluOp.getNegativeSlopeAttr());
            })
            .Case<IE::PReluOp>([](auto pReluOp) -> IE::PostOpAttr {
                const auto ctx = pReluOp.getContext();
                const auto slopesConst = pReluOp.getNegativeSlope().template getDefiningOp<Const::DeclareOp>();
                VPUX_THROW_WHEN(slopesConst == nullptr, "Cannon attributize PRelu operation with non-constant slopes.");

                const auto slopesContent = slopesConst.getContentAttr().fold();
                if (slopesContent.isSplat()) {
                    const auto slope = slopesContent.template getSplatValue<double>();
                    return IE::LeakyReluAttr::get(ctx, getFPAttr(ctx, slope));
                }

                const auto slopes = slopesContent.template getValues<double>();
                return IE::PReluAttr::get(ctx, getFPArrayAttr(ctx, slopes));
            })
            .Case<IE::TanhOp>([](auto tanhOp) {
                return IE::TanhAttr::get(tanhOp.getContext());
            })
            .Case<IE::SigmoidOp>([](auto sigmoidOp) {
                return IE::SigmoidAttr::get(sigmoidOp.getContext());
            })
            .Case<IE::SwishOp>([](auto swishOp) {
                const auto beta = swishOp.getBetaValueAttr();
                VPUX_THROW_WHEN(beta == nullptr, "Cannot attributize Swish operation with non-constant beta.");
                return IE::SwishAttr::get(swishOp.getContext(), beta);
            })
            .Case<IE::GeluOp>([](auto geluOp) {
                return IE::GeluAttr::get(geluOp.getContext());
            })
            .Case<IE::ExpOp>([](auto expOp) {
                return IE::ExpAttr::get(expOp.getContext());
            })
            .Case<IE::HSwishOp>([](auto hswishOp) {
                return IE::HSwishAttr::get(hswishOp.getContext());
            })
            .Default([](auto unknownOp) {
                VPUX_THROW("Failed to attributize operation: {0}", unknownOp->getName());
                return nullptr;
            });
}

//
// LayoutInfoOpInterface
//

void vpux::IE::LayerLayoutInfo::setInput(size_t ind, const DimsOrder& info) {
    const auto prevInfo = getInput(ind);
    VPUX_THROW_UNLESS(info.numDims() == prevInfo.numDims(), "New order '{0}' doesn't match original rank '{1}'", info,
                      prevInfo.numDims());

    LayerDataInfo<DimsOrder>::setInput(ind, info);
}

void vpux::IE::LayerLayoutInfo::setOutput(size_t ind, const DimsOrder& info) {
    const auto prevInfo = getOutput(ind);
    VPUX_THROW_UNLESS(info.numDims() == prevInfo.numDims(), "New order '{0}' doesn't match original rank '{1}'", info,
                      prevInfo.numDims());

    LayerDataInfo<DimsOrder>::setOutput(ind, info);
}

mlir::LogicalResult vpux::IE::verifyLayout(mlir::Operation*) {
    // Tracking number [E#84955]
    return mlir::success();
}

IE::LayerLayoutInfo vpux::IE::getLayoutInfo(mlir::Operation* op) {
    SmallVector<DimsOrder> inputOrders;
    inputOrders.reserve(op->getNumOperands());
    for (const auto& val : op->getOperands()) {
        inputOrders.push_back(DimsOrder::fromValue(val));
    }

    SmallVector<DimsOrder> outputOrders;
    outputOrders.reserve(op->getNumResults());
    for (const auto& val : op->getResults()) {
        outputOrders.push_back(DimsOrder::fromValue(val));
    }

    return IE::LayerLayoutInfo(std::move(inputOrders), std::move(outputOrders));
}

void vpux::IE::fillDefaultLayoutInfo(IE::LayerLayoutInfo& info) {
    for (auto i : irange(info.getNumInputs())) {
        info.setInput(i, DimsOrder::fromNumDims(info.getInput(i).numDims()));
    }

    for (auto i : irange(info.getNumOutputs())) {
        info.setOutput(i, DimsOrder::fromNumDims(info.getOutput(i).numDims()));
    }
}

void vpux::IE::fillDefaultLayoutInfo(LayerLayoutInfo& info, FuncRef<bool(size_t)> inputFilter,
                                     FuncRef<bool(size_t)> outputFilter) {
    for (auto i : irange(info.getNumInputs()) | filtered(inputFilter)) {
        info.setInput(i, DimsOrder::fromNumDims(info.getInput(i).numDims()));
    }

    for (auto i : irange(info.getNumOutputs()) | filtered(outputFilter)) {
        info.setOutput(i, DimsOrder::fromNumDims(info.getOutput(i).numDims()));
    }
}

//
// EltwiseOp
//

mlir::LogicalResult vpux::IE::verifyEltwiseOp(mlir::Operation* op) {
    if (!mlir::isa<IE::LayerOpInterface>(op)) {
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

vpux::IE::LayerDataInfo<mlir::Type> vpux::IE::getElemTypeInfo(mlir::Operation* op) {
    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(op->getNumOperands());
    for (const auto& val : op->getOperands()) {
        inputTypes.push_back(mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType());
    }

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(op->getNumResults());
    for (const auto& val : op->getResults()) {
        outputTypes.push_back(mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType());
    }

    return vpux::IE::LayerDataInfo<mlir::Type>(std::move(inputTypes), std::move(outputTypes));
}

//
// isPureViewLike
//

bool vpux::IE::isPureViewOp(mlir::Operation* op) {
    return mlir::isa<IE::ViewLikeOpInterface, mlir::ViewLikeOpInterface>(op);
}

//
// Generated
//

#include <vpux/compiler/dialect/IE/ops_interfaces.cpp.inc>

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/types.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// LayerOpInterface
//

namespace {

// Returns the number of operands that are the result of other layers
// For this ops:
// %6 = VPUIP.SWKernel inputs(%1 : memref, %2 : memref) outputs(%3 : memref) waits(%4 : !VPUIP.Barrier) updates(%5 :
// !VPUIP.Barrier)) numOperands() == 5 <==> %1, %2, %3, %4, %5 getLastMemRefPosition() == 3 <==> %1, %2 and %3
ptrdiff_t getLastMemRefPosition(mlir::ValueRange vals) {
    return std::find_if(vals.begin(), vals.end(),
                        [](mlir::Value val) {
                            return !val.getType().isa<mlir::BaseMemRefType, VPUIP::ITIBufferType>();
                        }) -
           vals.begin();
}

mlir::Type getElementType(mlir::Type type) {
    VPUX_THROW_UNLESS(type.isa<vpux::NDTypeInterface>(), "Could not extract element type from '{0}'", type);
    const auto elemType = type.cast<vpux::NDTypeInterface>().getElementType();
    if (auto qType = elemType.dyn_cast<mlir::quant::QuantizedType>()) {
        return qType.getStorageType();
    }
    return elemType;
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::verifyLayer(mlir::Operation* op) {
    auto swKernel = mlir::dyn_cast<VPUIP::SwKernelOp>(op);
    if (swKernel && isCacheHandlingOp(swKernel)) {
        return mlir::success();
    }

    if (op->getOperands().empty()) {
        return errorAt(op, "RunTime Layer Operation has no operands");
    }
    if (op->getResults().empty()) {
        return errorAt(op, "RunTime Layer Operation has no results");
    }
    const auto verifyType = [&](mlir::Type type, StringRef name, unsigned ind) {
        if (type.isa<mlir::RankedTensorType>()) {
            return errorAt(op, "RunTime Layer Operation has Tensor {0} #{1}", name, ind);
        }
        if (auto mainType = type.dyn_cast<mlir::ShapedType>()) {
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

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    if (inNum < outNum) {
        return errorAt(op,
                       "The number of operands must always be greater than or equal to the number of results, since "
                       "they include buffers for the results : inNum={0} outNum={1}",
                       inNum, outNum);
    }

    return mlir::success();
}

mlir::OperandRange vpux::VPUIP::getLayerInputs(mlir::Operation* op) {
    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOperands().take_front(inNum - outNum);
}

mlir::OperandRange vpux::VPUIP::getLayerOutputs(mlir::Operation* op) {
    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOperands().slice(inNum - outNum, outNum);
}

MutableArrayRef<mlir::OpOperand> vpux::VPUIP::getLayerInOpOperands(mlir::Operation* op) {
    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOpOperands().take_front(inNum - outNum);
}

MutableArrayRef<mlir::OpOperand> vpux::VPUIP::getLayerOutOpOperands(mlir::Operation* op) {
    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOpOperands().slice(inNum - outNum, outNum);
}

mlir::Value vpux::VPUIP::getLayerViewSource(mlir::Operation* op, ptrdiff_t resultInd) {
    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    VPUX_THROW_UNLESS(resultInd < outNum, "Result index '{0}' is out of range '{1}'", resultInd, outNum);
    return op->getOperand(checked_cast<unsigned>(inNum - outNum + resultInd));
}

mlir::LogicalResult vpux::VPUIP::inferLayerReturnTypes(mlir::ValueRange operands, size_t numResults,
                                                       SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto inNum = getLastMemRefPosition(operands);

    VPUX_THROW_UNLESS(numResults <= checked_cast<size_t>(inNum),
                      "Call inferLayerReturnTypes for non RT Layer Operation: {0} {1}", numResults, inNum);

    inferredReturnTypes.reserve(numResults);
    for (const auto val : operands.slice(inNum - numResults, numResults)) {
        inferredReturnTypes.push_back(val.getType());
    }

    return mlir::success();
}

void vpux::VPUIP::getLayerEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects) {
    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    VPUX_THROW_WHEN(layer == nullptr, "Got non layer operation '{0}' at '{1}' in getLayerEffects", op->getName(),
                    op->getLoc());

    auto swKernel = mlir::dyn_cast<VPUIP::SwKernelOp>(op);
    if (swKernel != nullptr && isCacheHandlingOp(swKernel)) {
        effects.emplace_back(mlir::MemoryEffects::Read::get());
        effects.emplace_back(mlir::MemoryEffects::Write::get());
    } else {
        for (const auto input : layer.getInputs()) {
            effects.emplace_back(mlir::MemoryEffects::Read::get(), input);
        }

        for (const auto output : layer.getOutputs()) {
            effects.emplace_back(mlir::MemoryEffects::Write::get(), output);
        }
    }
}

//
// TaskOpInterface
//

IndexedSymbolAttr vpux::VPUIP::getExecutorAttr(mlir::Operation* op, VPU::ExecutorKind kind) {
    return IndexedSymbolAttr::get(op->getContext(), stringifyEnum(kind));
}

IndexedSymbolAttr vpux::VPUIP::getTaskOpExecutor(mlir::Operation* op) {
    auto task = mlir::dyn_cast<VPUIP::TaskOpInterface>(op);
    VPUX_THROW_WHEN(task == nullptr, "Operation '{0}' is not a VPUIP Task", op->getName());
    return VPUIP::getExecutorAttr(op, task.getExecutorKind());
}

//
// Legacy4D
//

mlir::LogicalResult vpux::VPUIP::verifyLegacy4D(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement RT Layer interface", op->getName());
    }

    for (const auto& val : layer.getOpOperands()) {
        const auto shape = getShape(val.get());
        const auto order = DimsOrder::fromValue(val.get());

        if (shape.size() != 3 && shape.size() != 4) {
            return errorAt(op, "Got unsupported shape '{0}', only 3D/4D are supported", shape);
        }

        if (shape.size() == 3) {
            if (order != DimsOrder::CHW && order != DimsOrder::HWC) {
                return errorAt(op, "Got unsupported input DimsOrder '{0}', only CHW and HWC are supported", order);
            }
        } else if (shape.size() == 4) {
            if (order != DimsOrder::NCHW && order != DimsOrder::NHWC) {
                return errorAt(op, "Got unsupported input DimsOrder '{0}', only NCHW and NHWC are supported", order);
            }

            if (shape.front() != 1) {
                return errorAt(op, "Batch size != 1 is not supported");
            }
        }
    }

    return mlir::success();
}

//
// isPureViewLike
//

bool vpux::VPUIP::isPureViewOp(mlir::Operation* op) {
    return mlir::isa<mlir::ViewLikeOpInterface, vpux::MultiViewOpInterface, vpux::GroupedViewOpInterface>(op) &&
           !mlir::isa<VPUIP::LayerOpInterface>(op);
}

//
// SameShape
//

mlir::LogicalResult vpux::VPUIP::verifySameShape(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    const auto mainShape = getShape(firstInput);

    for (const auto& val : layer.getOpOperands()) {
        const auto shape = getShape(val.get());

        if (shape != mainShape) {
            return errorAt(op, "Operation's input/output shapes mismatch: {0} != {1}", mainShape, shape);
        }
    }

    return mlir::success();
}

//
// SameElementType
//

mlir::LogicalResult vpux::VPUIP::verifySameElementType(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    auto mainElemType = getElementType(firstInput.getType());

    for (const auto& val : layer.getOpOperands()) {
        auto elemType = getElementType(val.get().getType());

        if (elemType != mainElemType) {
            return errorAt(op, "Operation's input/output element types mismatch");
        }
    }

    return mlir::success();
}

//
// SameDimsOrder
//

mlir::LogicalResult vpux::VPUIP::verifySameDimsOrder(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    const auto mainOrder = DimsOrder::fromValue(firstInput);

    for (const auto& val : layer.getOpOperands()) {
        const auto order = DimsOrder::fromValue(val.get());

        if (order != mainOrder) {
            return errorAt(op, "Operation's input/output layout mismatch");
        }
    }

    return mlir::success();
}

//
// SameInOutDimsOrder
//

mlir::LogicalResult vpux::VPUIP::verifySameInOutDimsOrder(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation {0} is not layer", op->getName());

    const auto input = layer.getInputs()[0];
    const auto output = layer.getOutputs()[0];

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromValue(output);

    if (inOrder != outOrder) {
        return errorAt(op->getLoc(), "Operation must have the same input and output order. inL={0}, outL={1}", inOrder,
                       outOrder);
    }

    return mlir::success();
}

//
// SameInOutSpecificDimsOrder
//

mlir::LogicalResult vpux::VPUIP::verifySameInOutSpecificDimsOrder(mlir::Operation* op,
                                                                  ArrayRef<DimsOrder> supportedLayouts) {
    if (verifySameInOutDimsOrder(op).failed()) {
        return mlir::failure();
    }

    auto layerOp = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);

    const auto input = layerOp.getInputs()[0];
    const auto inOrder = DimsOrder::fromValue(input);

    const auto isSupported = std::count(supportedLayouts.begin(), supportedLayouts.end(), inOrder);
    if (!isSupported) {
        return errorAt(op->getLoc(), "Operation does not support {0} layout", inOrder);
    }

    return mlir::success();
}

//
// SameOperandsAndResultElementType
//

mlir::LogicalResult vpux::VPUIP::verifySameOperandsAndResultElementType(mlir::Operation* op) {
    if (op->getOperands().empty() || op->getResults().empty()) {
        return errorAt(op, "Operation '{0}' should have at least one input and output", op->getName());
    }

    auto elementType = getElementType(op->getResult(0).getType());

    // Verify result element type matches first result's element type.
    for (auto result : llvm::drop_begin(op->getResults(), 1)) {
        if (getElementType(result.getType()) != elementType) {
            return errorAt(op, "Expected result element type '{0}' to be '{1}'", getElementType(result.getType()),
                           elementType);
        }
    }

    // Verify operand's element type matches first result's element type.
    for (auto operand : op->getOperands()) {
        if (getElementType(operand.getType()) != elementType) {
            return errorAt(op, "Expected operand element type '{0}' to be '{1}'", getElementType(operand.getType()),
                           elementType);
        }
    }
    return mlir::success();
}

//
// DMATypeOpInterface
//

std::optional<VPUIP::DmaChannelType> vpux::VPUIP::getChannelType(mlir::Operation* op) {
    // Configure DMA channel only for NPU40XX for now
    const auto arch = VPU::getArch(op);
    if (arch < VPU::ArchKind::NPU40XX) {
        return std::nullopt;
    }

    auto srcMemKind = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getMemoryKind();

    switch (srcMemKind) {
    case VPU::MemoryKind::DDR:
        return VPUIP::DmaChannelType::DDR;
    case VPU::MemoryKind::CMX_NN:
        return VPUIP::DmaChannelType::CMX;
    default:
        return std::nullopt;
    }
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/ops_interfaces.cpp.inc>

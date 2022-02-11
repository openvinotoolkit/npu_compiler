//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"
#include "vpux/compiler/dialect/VPURT/types.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/Interfaces/ViewLikeInterface.h>

using namespace vpux;

//
// LayerOpInterface
//

namespace {

// Returns the number of operands that are the result of other layers
// For this ops:
// %6 = VPUIP.SomeTaskUPA inputs(%1 : memref, %2 : memref) outputs(%3 : memref) waits(%4 : !VPUIP.Barrier) updates(%5 :
// !VPUIP.Barrier)) numOperands() == 5 <==> %1, %2, %3, %4, %5 getLastMemRefPosition() == 3  <==> %1, %2 and %3
ptrdiff_t getLastMemRefPosition(mlir::ValueRange vals) {
    return std::find_if(
                   vals.begin(), vals.end(),
                   [](mlir::Value val) {
                       return !val.getType()
                                       .isa<mlir::MemRefType, VPURT::SparseBufferType, VPUIP::DistributedBufferType>();
                   }) -
           vals.begin();
}

}  // namespace

mlir::LogicalResult vpux::IERT::verifyLayer(mlir::Operation* op) {
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

mlir::OperandRange vpux::IERT::getLayerInputs(mlir::Operation* op) {
    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOperands().take_front(inNum - outNum);
}

mlir::OperandRange vpux::IERT::getLayerOutputs(mlir::Operation* op) {
    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOperands().slice(inNum - outNum, outNum);
}

MutableArrayRef<mlir::OpOperand> vpux::IERT::getLayerInOpOperands(mlir::Operation* op) {
    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOpOperands().take_front(inNum - outNum);
}

MutableArrayRef<mlir::OpOperand> vpux::IERT::getLayerOutOpOperands(mlir::Operation* op) {
    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOpOperands().slice(inNum - outNum, outNum);
}

mlir::Value vpux::IERT::getLayerViewSource(mlir::Operation* op, ptrdiff_t resultInd) {
    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    VPUX_THROW_UNLESS(resultInd < outNum, "Result index '{0}' is out of range '{1}'", resultInd, outNum);
    return op->getOperand(checked_cast<unsigned>(inNum - outNum + resultInd));
}

mlir::LogicalResult vpux::IERT::inferLayerReturnTypes(mlir::ValueRange operands, size_t numResults,
                                                      SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto inNum = getLastMemRefPosition(operands);

    VPUX_THROW_UNLESS(numResults <= checked_cast<size_t>(inNum),
                      "Call inferLayerReturnTypes for non RT Layer Operation");

    inferredReturnTypes.reserve(numResults);
    for (const auto val : operands.slice(inNum - numResults, numResults)) {
        inferredReturnTypes.push_back(val.getType());
    }

    return mlir::success();
}

void vpux::IERT::getLayerEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects) {
    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    VPUX_THROW_WHEN(layer == nullptr, "Got non layer operation '{0}' at '{1}' in getLayerEffects", op->getName(),
                    op->getLoc());

    for (const auto input : layer.getInputs()) {
        effects.emplace_back(mlir::MemoryEffects::Read::get(), input);
    }

    for (const auto output : layer.getOutputs()) {
        effects.emplace_back(mlir::MemoryEffects::Write::get(), output);
    }
}

//
// SameShape
//

mlir::LogicalResult vpux::IERT::verifySameShape(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    const auto mainShape = getShape(firstInput);

    for (const auto& val : layer.getOpOperands()) {
        const auto shape = getShape(val.get());

        if (shape != mainShape) {
            return errorAt(op, "Operation's input/output shapes mismatch");
        }
    }

    return mlir::success();
}

//
// SameElementType
//

mlir::LogicalResult vpux::IERT::verifySameElementType(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    auto mainElemType = firstInput.getType().cast<mlir::ShapedType>().getElementType();
    if (auto qType = mainElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        mainElemType = qType.getStorageType();
    }

    for (const auto& val : layer.getOpOperands()) {
        auto elemType = val.get().getType().cast<mlir::ShapedType>().getElementType();
        if (auto qType = elemType.dyn_cast<mlir::quant::QuantizedType>()) {
            elemType = qType.getStorageType();
        }

        if (elemType != mainElemType) {
            return errorAt(op, "Operation's input/output element types mismatch");
        }
    }

    return mlir::success();
}

//
// SameDimsOrder
//

mlir::LogicalResult vpux::IERT::verifySameDimsOrder(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
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

mlir::LogicalResult vpux::IERT::verifySameInOutDimsOrder(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
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

void vpux::IERT::inferLayoutInfoSameInOutDimsOrder(IE::LayerLayoutInfo& info) {
    const auto filter = [](size_t ind) {
        return ind != 0;
    };
    IE::fillDefaultLayoutInfo(info, filter, filter);

    info.setOutput(0, info.getInput(0));
}

//
// SameInOutSpecificDimsOrder
//

mlir::LogicalResult vpux::IERT::verifySameInOutSpecificDimsOrder(mlir::Operation* op,
                                                                 ArrayRef<DimsOrder> supportedLayouts) {
    if (verifySameInOutDimsOrder(op).failed()) {
        return mlir::failure();
    }

    auto layerOp = mlir::dyn_cast<IERT::LayerOpInterface>(op);

    const auto input = layerOp.getInputs()[0];
    const auto inOrder = DimsOrder::fromValue(input);

    const auto isSupported = std::count(supportedLayouts.begin(), supportedLayouts.end(), inOrder);
    if (!isSupported) {
        return errorAt(op->getLoc(), "Operation does not support {0} layout", inOrder);
    }

    return mlir::success();
}

void vpux::IERT::inferLayoutInfoSameInOutSpecificDimsOrder(IE::LayerLayoutInfo& info,
                                                           ArrayRef<DimsOrder> supportedLayouts) {
    const auto filter = [](size_t ind) {
        return ind != 0;
    };
    IE::fillDefaultLayoutInfo(info, filter, filter);

    const auto mainOrder = info.getInput(0);

    if (llvm::is_contained(supportedLayouts, mainOrder)) {
        info.setOutput(0, mainOrder);
        return;
    }

    const auto supportedOrderIt = llvm::find_if(supportedLayouts, [mainOrder](DimsOrder order) {
        return order.numDims() == mainOrder.numDims();
    });

    VPUX_THROW_UNLESS(supportedOrderIt != supportedLayouts.end(),
                      "Layouts supported by the operation '{0}' do not match the rank '{1}' of the input shape",
                      supportedLayouts, mainOrder.numDims());

    const auto supportedOrder = *supportedOrderIt;
    info.setInput(0, supportedOrder);
    info.setOutput(0, supportedOrder);
}

//
// isPureViewLike
//

bool vpux::IERT::isPureViewOp(mlir::Operation* op) {
    return mlir::isa<mlir::ViewLikeOpInterface>(op) && !mlir::isa<IERT::LayerOpInterface>(op);
}

//
// Generated
//

#include <vpux/compiler/dialect/IERT/generated/ops_interfaces.cpp.inc>

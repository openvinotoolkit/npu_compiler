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

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/attributes/structs.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

namespace vpux {
namespace IE {

//
// Dims4D
//

struct Dims4D final {
    // Convolution2D/Pooling2D activations

    struct Act final {
        static const Dim N;
        static const Dim C;
        static const Dim H;
        static const Dim W;

        static constexpr size_t numSpatialDims = 2;

        static Dim getSpatialDim(size_t index) {
            VPUX_THROW_UNLESS(index < 2, "Dims4D::Act: Wrong spatial dimension index '{0}'", index);
            return Dim(index + 2);
        }
    };

    // Convolution2D filter

    struct Filter final {
        static const Dim OC;
        static const Dim IC;
        static const Dim KY;
        static const Dim KX;

        static constexpr size_t numSpatialDims = 2;

        static Dim getSpatialDim(size_t index) {
            VPUX_THROW_UNLESS(index < 2, "Dims4D::Filter: Wrong spatial dimension index '{0}'", index);
            return Dim(index + 2);
        }
    };
};

//
// DataOrderInfo
//

class DataOrderInfo final {
public:
    DataOrderInfo(size_t numInputs, size_t numOutputs) {
        _inputOrders.resize(numInputs);
        _outputOrders.resize(numOutputs);
    }

public:
    void setInput(size_t argNum, DimsOrder order) {
        VPUX_THROW_UNLESS(argNum < _inputOrders.size(), "Argument number {0} is out of range {1}", argNum,
                          _inputOrders.size());
        _inputOrders[argNum] = order;
    }
    void setOutput(size_t argNum, DimsOrder order) {
        VPUX_THROW_UNLESS(argNum < _outputOrders.size(), "Argument number {0} is out of range {1}", argNum,
                          _outputOrders.size());
        _outputOrders[argNum] = order;
    }

    bool hasInput(size_t argNum) const {
        VPUX_THROW_UNLESS(argNum < _inputOrders.size(), "Argument number {0} is out of range {1}", argNum,
                          _inputOrders.size());
        return _inputOrders[argNum].hasValue();
    }
    bool hasOutput(size_t argNum) const {
        VPUX_THROW_UNLESS(argNum < _outputOrders.size(), "Argument number {0} is out of range {1}", argNum,
                          _outputOrders.size());
        return _outputOrders[argNum].hasValue();
    }

    DimsOrder getInput(size_t argNum) const {
        VPUX_THROW_UNLESS(hasInput(argNum), "No value for argument {0}", argNum);
        return _inputOrders[argNum].getValue();
    }
    DimsOrder getOutput(size_t argNum) const {
        VPUX_THROW_UNLESS(hasOutput(argNum), "No value for argument {0}", argNum);
        return _outputOrders[argNum].getValue();
    }

public:
    void printFormat(llvm::raw_ostream& stream) const;

private:
    SmallVector<Optional<DimsOrder>> _inputOrders;
    SmallVector<Optional<DimsOrder>> _outputOrders;
};

void fillDataInfo(DataOrderInfo& info, size_t inNum, size_t outNum, DimsOrder mainOrder);

//
// LayerOpInterface
//

mlir::LogicalResult verifyLayer(mlir::Operation* op);

using InferTypeComponentsCb = FuncRef<mlir::LogicalResult(mlir::MLIRContext*, Optional<mlir::Location>,
                                                          mlir::ValueRange, mlir::DictionaryAttr, mlir::RegionRange,
                                                          SmallVectorImpl<mlir::ShapedTypeComponents>&)>;

mlir::LogicalResult inferTensorTypes(InferTypeComponentsCb componentsCb, mlir::MLIRContext* ctx,
                                     Optional<mlir::Location> loc, mlir::ValueRange operands,
                                     mlir::DictionaryAttr attrs, mlir::RegionRange regions,
                                     SmallVectorImpl<mlir::Type>& inferredTypes);

bool isCompatibleShapeAndElemType(mlir::TypeRange lhs, mlir::TypeRange rhs);

//
// LayerWithPostOpInterface
//

template <typename ConcreteOp>
Optional<mlir::OperationName> getLayerPostOp(ConcreteOp mainOp) {
    if (auto postOpInfo = mainOp.post_opAttr()) {
        return mlir::OperationName(postOpInfo.name().getValue(), mainOp->getContext());
    }

    return None;
}

template <typename ConcreteOp>
mlir::DictionaryAttr getLayerPostOpAttrs(ConcreteOp mainOp) {
    if (auto postOpInfo = mainOp.post_opAttr()) {
        return postOpInfo.attrs();
    }

    return nullptr;
}

template <typename ConcreteOp>
void setLayerPostOp(ConcreteOp mainOp, mlir::Operation* postOp) {
    VPUX_THROW_UNLESS(mainOp.post_opAttr() == nullptr, "Operation '{0}' at '{1}' already has a PostOp '{2}'",
                      mainOp->getName(), mainOp->getLoc(), mainOp.post_opAttr());
    VPUX_THROW_UNLESS(postOp->getNumOperands() == 1,
                      "Only single input operation can be attached as PostOp via attributes");

    const auto postOpName = mlir::StringAttr::get(mainOp->getContext(), postOp->getName().getStringRef());
    const auto postOpInfo = IE::PostOp::get(postOpName, postOp->getAttrDictionary(), mainOp->getContext());
    mainOp.post_opAttr(postOpInfo);
}

//
// LayoutInfoOpInterface
//

DataOrderInfo getDataOrderInfo(mlir::Operation* op);

//
// EltwiseOp
//

mlir::LogicalResult verifyEltwiseOp(mlir::Operation* op);

template <typename ConcreteOp>
class EltwiseOp : public mlir::OpTrait::TraitBase<ConcreteOp, EltwiseOp> {
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifyEltwiseOp(op);
    }
};

}  // namespace IE
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/ops_interfaces.hpp.inc>

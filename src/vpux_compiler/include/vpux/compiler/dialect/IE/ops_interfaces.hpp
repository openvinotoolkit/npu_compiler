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

#include "vpux/utils/core/format.hpp"
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
// LayerDataInfo
//

template <class InfoT>
class LayerDataInfo {
public:
    template <class Range1, class Range2>
    LayerDataInfo(Range1&& inputInfo, Range2&& outputInfo)
            : _inputInfo(std::forward<Range1>(inputInfo)), _outputInfo(std::forward<Range1>(outputInfo)) {
    }

public:
    bool hasChanges() const {
        return _hasChanges;
    }

public:
    size_t getNumInputs() const {
        return _inputInfo.size();
    }
    size_t getNumOutputs() const {
        return _outputInfo.size();
    }

    const InfoT& getInput(size_t ind) const {
        VPUX_THROW_UNLESS(ind < _inputInfo.size(), "Input index '{0}' is out of range '{1}'", ind, _inputInfo.size());
        return _inputInfo[ind];
    }
    const InfoT& getOutput(size_t ind) const {
        VPUX_THROW_UNLESS(ind < _outputInfo.size(), "Output index '{0}' is out of range '{1}'", ind,
                          _outputInfo.size());
        return _outputInfo[ind];
    }

public:
    virtual void setInput(size_t ind, const InfoT& info) {
        VPUX_THROW_UNLESS(ind < _inputInfo.size(), "Input index '{0}' is out of range '{1}'", ind, _inputInfo.size());

        if (info != _inputInfo[ind]) {
            _hasChanges = true;
        }

        _inputInfo[ind] = info;
    }

    virtual void setOutput(size_t ind, const InfoT& info) {
        VPUX_THROW_UNLESS(ind < _outputInfo.size(), "Output index '{0}' is out of range '{1}'", ind,
                          _outputInfo.size());

        if (info != _outputInfo[ind]) {
            _hasChanges = true;
        }

        _outputInfo[ind] = info;
    }

    void fill(const InfoT& info) {
        for (size_t i = 0; i < getNumInputs(); ++i) {
            setInput(i, info);
        }

        for (size_t i = 0; i < getNumOutputs(); ++i) {
            setOutput(i, info);
        }
    }

public:
    void printFormat(llvm::raw_ostream& stream) const {
        stream << "LayerDataInfo {";

        for (size_t i = 0; i < _inputInfo.size(); ++i) {
            printTo(stream, " in[{0}] = {1}", i, _inputInfo[i]);
        }

        stream << ";";
        for (size_t i = 0; i < _outputInfo.size(); ++i) {
            printTo(stream, " out[{0}] = {1}", i, _outputInfo[i]);
        }

        stream << " }";
    }

private:
    SmallVector<InfoT> _inputInfo;
    SmallVector<InfoT> _outputInfo;
    bool _hasChanges = false;
};

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

class LayerLayoutInfo final : public LayerDataInfo<DimsOrder> {
public:
    using LayerDataInfo<DimsOrder>::LayerDataInfo;

public:
    void setInput(size_t ind, const DimsOrder& info) final;
    void setOutput(size_t ind, const DimsOrder& info) final;
};

LayerLayoutInfo getLayoutInfo(mlir::Operation* op);
void fillDefaultLayoutInfo(LayerLayoutInfo& info);
void fillDefaultLayoutInfo(LayerLayoutInfo& info, FuncRef<bool(size_t)> inputFilter,
                           FuncRef<bool(size_t)> outputFilter);

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

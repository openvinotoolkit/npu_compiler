//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

namespace vpux {
namespace VPU {

// Forward declaration to avoid including VPU headers
enum class ExecutorKind : uint64_t;

}  // namespace VPU
namespace IE {

//
// LayerDataInfo
//

template <class InfoT>
class LayerDataInfo {
public:
    template <class Range1, class Range2>
    LayerDataInfo(Range1&& inputInfo, Range2&& outputInfo)
            : _inputInfo(std::forward<Range1>(inputInfo)), _outputInfo(std::forward<Range2>(outputInfo)) {
    }
    virtual ~LayerDataInfo() = default;

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

    ArrayRef<mlir::Type> getInputs() const {
        return _inputInfo;
    }
    ArrayRef<mlir::Type> getOutputs() const {
        return _outputInfo;
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
    void resetChanges() {
        _hasChanges = false;
    }

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

using InferTypeComponentsCb = FuncRef<mlir::LogicalResult(
        mlir::MLIRContext*, std::optional<mlir::Location>, mlir::ValueRange, mlir::DictionaryAttr,
        mlir::OpaqueProperties, mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>&)>;

mlir::LogicalResult inferTensorTypes(InferTypeComponentsCb componentsCb, mlir::MLIRContext* ctx,
                                     std::optional<mlir::Location> loc, mlir::ValueRange operands,
                                     mlir::DictionaryAttr attrs, mlir::OpaqueProperties props,
                                     mlir::RegionRange regions, SmallVectorImpl<mlir::Type>& inferredTypes);

//
// LayerWithPostOpInterface
//

PostOpAttr attributizePostOp(mlir::Operation* postOp);

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

mlir::LogicalResult verifyLayout(mlir::Operation*);
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
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return IE::verifyEltwiseOp(op);
    }
};

//
// ElemTypeInfoOpInterface
//

void inferElemTypeInfo(mlir::Operation* op, LayerDataInfo<mlir::Type>& info);
void inferElemTypeInfoUp(mlir::Operation* op, LayerDataInfo<mlir::Type>& info);
LayerDataInfo<mlir::Type> getElemTypeInfo(mlir::Operation* op);

//
// isPureViewOp
//

bool isPureViewOp(mlir::Operation* op);

}  // namespace IE
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/IE/ops_interfaces.hpp.inc>

//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/odu_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

namespace vpux {
namespace VPU {

template <typename ConvType>
SmallVector<int64_t> getKernelSize(ConvType op) {
    const auto kernelShape = Shape(op.getConstRawFilterShape());
    auto KY = kernelShape[Dims4D::Filter::KY];
    auto KX = kernelShape[Dims4D::Filter::KX];
    if (kernelShape.size() == DimsGroups5D::Filter::numDims) {
        KY = kernelShape[DimsGroups5D::Filter::KY];
        KX = kernelShape[DimsGroups5D::Filter::KX];
    }
    return {KY, KX};
}

//
// NCEConvolution-like op models
//

template <typename ConcreteModel, typename ConcreteOp>
class NCEConvolutionOpBaseModel : public VPU::NCEOpInterface::ExternalModel<ConcreteModel, ConcreteOp> {
public:
    SmallVector<int64_t> getKernelSizeVal(mlir::Operation* op) const {
        return getKernelSize<ConcreteOp>(mlir::cast<ConcreteOp>(op));
    }
    SmallVector<int64_t> getStridesVal(mlir::Operation* op) const {
        return parseIntArrayAttr<int64_t>(mlir::cast<ConcreteOp>(op).getStrides());
    }
    mlir::Value getWeightsTableOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightsTable();
    }
    VPU::MPEMode getMpeMode(mlir::Operation* op, mlir::Type inElemType, mlir::Type outElemType, ShapeRef shape) const {
        return static_cast<const ConcreteModel*>(this)->getMpeModeImpl(op, inElemType, outElemType, shape);
    }
};

template <typename ConcreteModel, typename ConcreteOp>
class NCEConvolutionOpModel : public NCEConvolutionOpBaseModel<ConcreteModel, ConcreteOp> {
public:
    mlir::Value getWeightsOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getFilter();
    }
    mlir::Value getWeightTableDataPtrOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableDataPtr();
    }
    mlir::Value getWeightTableSpPtrOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableSpPtr();
    }
    mlir::Value getWeightTableScaleOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableScale();
    }
    mlir::Value getWeightTableBiasOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableBias();
    }
    mlir::Value getWeightZeroPointsOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightZeroPoints();
    }
    vpux::VPU::PaddingAttr getPad(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getPad();
    }
};

template <typename ConcreteModel, typename ConcreteOp>
class NCECompressConvolutionOpModel : public NCEConvolutionOpBaseModel<ConcreteModel, ConcreteOp> {
public:
    mlir::Value getWeightsOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getFilter();
    }
    vpux::VPU::PaddingAttr getPad(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getPad();
    }
};

template <typename ConcreteModel, typename ConcreteOp>
class NCEInterpolateOpModel : public NCEConvolutionOpBaseModel<ConcreteModel, ConcreteOp> {
public:
    mlir::Value getWeightsOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeights();
    }
    mlir::Value getWeightTableDataPtrOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableDataPtr();
    }
    mlir::Value getWeightTableSpPtrOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableSpPtr();
    }
    mlir::Value getWeightTableScaleOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableScale();
    }
    mlir::Value getWeightTableBiasOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableBias();
    }
    mlir::Value getWeightZeroPointsOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightZeroPoints();
    }
    vpux::VPU::PaddingAttr getPad(mlir::Operation* op) const {
        return vpux::VPU::getPaddingAttr(mlir::cast<ConcreteOp>(op).getContext(), 0, 0, 0, 0);
    }
};

template <typename ConcreteModel, typename ConcreteOp>
class NCEMatMulOpModel : public NCEConvolutionOpBaseModel<ConcreteModel, ConcreteOp> {
public:
    mlir::Value getWeightsOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeights();
    }
    mlir::Value getWeightTableDataPtrOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableDataPtr();
    }
    mlir::Value getWeightTableSpPtrOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableSpPtr();
    }
    mlir::Value getWeightTableScaleOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableScale();
    }
    mlir::Value getWeightTableBiasOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableBias();
    }
    mlir::Value getWeightZeroPointsOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightZeroPoints();
    }
    vpux::VPU::PaddingAttr getPad(mlir::Operation* op) const {
        return vpux::VPU::getPaddingAttr(mlir::cast<ConcreteOp>(op).getContext(), 0, 0, 0, 0);
    }
};

//
// NCEPool-like op models
//

template <typename ConcreteModel, typename ConcreteOp>
class NCEPoolOpBaseModel : public VPU::NCEOpInterface::ExternalModel<ConcreteModel, ConcreteOp> {
public:
    SmallVector<int64_t> getKernelSizeVal(mlir::Operation* op) const {
        return parseIntArrayAttr<int64_t>(mlir::cast<ConcreteOp>(op).getKernelSize());
    }
    SmallVector<int64_t> getStridesVal(mlir::Operation* op) const {
        return parseIntArrayAttr<int64_t>(mlir::cast<ConcreteOp>(op).getStrides());
    }
    vpux::VPU::PaddingAttr getPad(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getPad();
    }
    VPU::MPEMode getMpeMode(mlir::Operation* op, mlir::Type inElemType, mlir::Type outElemType, ShapeRef shape) const {
        return static_cast<const ConcreteModel*>(this)->getMpeModeImpl(op, inElemType, outElemType, shape);
    }
};

template <typename ConcreteModel, typename ConcreteOp>
class NCEAveragePoolOpModel : public NCEPoolOpBaseModel<ConcreteModel, ConcreteOp> {
public:
    mlir::Value getWeightsTableOperand(mlir::Operation*) const {
        return nullptr;
    }
    mlir::Value getWeightTableDataPtrOperand(mlir::Operation*) const {
        return nullptr;
    }
    mlir::Value getWeightTableSpPtrOperand(mlir::Operation*) const {
        return nullptr;
    }
    mlir::Value getWeightTableScaleOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableScale();
    }
    mlir::Value getWeightTableBiasOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightTableBias();
    }
    mlir::Value getWeightZeroPointsOperand(mlir::Operation*) const {
        return nullptr;
    }
};

template <typename ConcreteModel, typename ConcreteOp>
class NCEMaxPoolOpModel : public NCEPoolOpBaseModel<ConcreteModel, ConcreteOp> {
public:
    mlir::Value getWeightsTableOperand(mlir::Operation* op) const {
        return mlir::cast<ConcreteOp>(op).getWeightsTable();
    }

    SmallVector<vpux::VPU::ODUDimScale> getODUScaling(mlir::Operation* op) const {
        auto concreteOp = mlir::cast<ConcreteOp>(op);
        const auto cfg = concreteOp.getS2dd2sConfigAttr();
        if (!cfg) {
            return {};
        }
        const int64_t rank = mlir::cast<vpux::NDTypeInterface>(concreteOp.getOutput().getType()).getRank();
        const auto transformInfo = VPU::getODUS2DD2STransformInfo(cfg, rank);
        if (!transformInfo.has_value()) {
            return {};
        }
        return VPU::getODUS2DD2SScaling(*transformInfo, rank);
    }
};

//
// NCEEltwise op model
//

template <typename ConcreteModel, typename ConcreteOp>
class NCEEltwiseOpModel : public VPU::NCEOpInterface::ExternalModel<ConcreteModel, ConcreteOp> {
public:
    mlir::Value getWeightsTableOperand(mlir::Operation*) const {
        return nullptr;
    }
    mlir::Value getWeightTableDataPtrOperand(mlir::Operation*) const {
        return nullptr;
    }
    mlir::Value getWeightTableSpPtrOperand(mlir::Operation*) const {
        return nullptr;
    }
    mlir::Value getWeightTableScaleOperand(mlir::Operation*) const {
        return nullptr;
    }
    mlir::Value getWeightTableBiasOperand(mlir::Operation*) const {
        return nullptr;
    }
    mlir::Value getWeightZeroPointsOperand(mlir::Operation*) const {
        return nullptr;
    }
    SmallVector<int64_t> getKernelSizeVal(mlir::Operation*) const {
        return {1, 1};
    }
    SmallVector<int64_t> getStridesVal(mlir::Operation*) const {
        return {1, 1};
    }
    vpux::VPU::PaddingAttr getPad(mlir::Operation* op) const {
        return vpux::VPU::getPaddingAttr(mlir::cast<ConcreteOp>(op).getContext(), 0, 0, 0, 0);
    }
    VPU::MPEMode getMpeMode(mlir::Operation* op, mlir::Type inElemType, mlir::Type outElemType, ShapeRef shape) const {
        return static_cast<const ConcreteModel*>(this)->getMpeModeImpl(op, inElemType, outElemType, shape);
    }
};

//
// NCEReduce op model
//

template <typename ConcreteModel, typename ConcreteOp>
class NCEReduceOpModel : public VPU::NCEOpInterface::ExternalModel<ConcreteModel, ConcreteOp> {
public:
    SmallVector<int64_t> getKernelSizeVal(mlir::Operation*) const {
        return {1, 1};
    }
    SmallVector<int64_t> getStridesVal(mlir::Operation*) const {
        return {1, 1};
    }
    vpux::VPU::PaddingAttr getPad(mlir::Operation* op) const {
        return vpux::VPU::getPaddingAttr(mlir::cast<ConcreteOp>(op).getContext(), 0, 0, 0, 0);
    }
    VPU::MPEMode getMpeMode(mlir::Operation* op, mlir::Type inElemType, mlir::Type outElemType, ShapeRef shape) const {
        return static_cast<const ConcreteModel*>(this)->getMpeModeImpl(op, inElemType, outElemType, shape);
    }
    mlir::Value getWeightsTableOperand(mlir::Operation*) const {
        return nullptr;
    }
};

}  // namespace VPU
}  // namespace vpux

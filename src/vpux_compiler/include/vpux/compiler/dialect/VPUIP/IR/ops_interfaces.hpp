//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/IR/indexed_symbol_attr.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux::IE {
class LayerLayoutInfo;
}  // namespace vpux::IE
namespace vpux::VPU {
enum class ExecutorKind : uint64_t;
}  // namespace vpux::VPU
namespace VPUNN {
class VPUCostModel;
}  // namespace VPUNN

namespace vpux {
namespace VPUIP {

//
// LayerOpInterface
//

mlir::LogicalResult verifyLayer(mlir::Operation* op);

mlir::OperandRange getLayerInputs(mlir::Operation* op);
mlir::OperandRange getLayerOutputs(mlir::Operation* op);

MutableArrayRef<mlir::OpOperand> getLayerInOpOperands(mlir::Operation* op);
MutableArrayRef<mlir::OpOperand> getLayerOutOpOperands(mlir::Operation* op);

mlir::Value getLayerViewSource(mlir::Operation* op, ptrdiff_t resultInd);

mlir::LogicalResult inferLayerReturnTypes(mlir::ValueRange operands, size_t numResults,
                                          SmallVectorImpl<mlir::Type>& inferredReturnTypes);

using MemoryEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;
void getLayerEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects);

//
// TaskOpInterface
//

using MemoryEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;
void getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects);

IndexedSymbolAttr getExecutorAttr(mlir::Operation* op, VPU::ExecutorKind kind);

IndexedSymbolAttr getTaskOpExecutor(mlir::Operation* op);

//
// Legacy4D
//

mlir::LogicalResult verifyLegacy4D(mlir::Operation* op);

template <typename ConcreteOp>
class Legacy4D : public mlir::OpTrait::TraitBase<ConcreteOp, Legacy4D> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifyLegacy4D(op);
    }
};

//
// SoftwareLayerOpInterface
//

struct KernelInfo final {
    SmallVector<mlir::Attribute> args;
    SmallString entryName;
    SmallString sourceFileName;
    SmallString layerName;

    KernelInfo(const SmallVector<mlir::Attribute>& opArgs, const SmallString& entryPointName,
               const SmallString& sourceName, const SmallString& kernelName)
            : args(opArgs), entryName(entryPointName), sourceFileName(sourceName), layerName(kernelName) {
    }

    KernelInfo(const SmallVector<mlir::Attribute>& opArgs, const SmallString& entryPointName,
               const SmallString& sourceName)
            : args(opArgs), entryName(entryPointName), sourceFileName(sourceName), layerName(entryPointName) {
    }
    KernelInfo(const SmallVector<mlir::Attribute>& opArgs, const SmallString& kernelName)
            : args(opArgs), entryName(kernelName), sourceFileName(kernelName), layerName(kernelName) {
        sourceFileName.append(".cpp");
    }
};

//
// isPureViewOp
//

bool isPureViewOp(mlir::Operation* op);

//
// SameShape
//

mlir::LogicalResult verifySameShape(mlir::Operation* op);

template <typename ConcreteOp>
class SameShape : public mlir::OpTrait::TraitBase<ConcreteOp, SameShape> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameShape(op);
    }
};

//
// SameElementType
//

mlir::LogicalResult verifySameElementType(mlir::Operation* op);

template <typename ConcreteOp>
class SameElementType : public mlir::OpTrait::TraitBase<ConcreteOp, SameElementType> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameElementType(op);
    }
};

//
// SameDimsOrder
//

mlir::LogicalResult verifySameDimsOrder(mlir::Operation* op);

template <typename ConcreteOp>
class SameDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameDimsOrder(op);
    }
};

//
// SameInOutDimsOrder
//

mlir::LogicalResult verifySameInOutDimsOrder(mlir::Operation* op);
void inferLayoutInfoSameInOutDimsOrder(IE::LayerLayoutInfo& info);

template <typename ConcreteOp>
class SameInOutDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutDimsOrder(op);
    }

    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        inferLayoutInfoSameInOutDimsOrder(info);
    }
};

mlir::LogicalResult verifySameInOutSpecificDimsOrder(mlir::Operation* op, ArrayRef<DimsOrder> supportedLayouts);

//
// SameOperandsAndResultElementType
//

mlir::LogicalResult verifySameOperandsAndResultElementType(mlir::Operation* op);

template <typename ConcreteOp>
class SameOperandsAndResultElementType : public mlir::OpTrait::TraitBase<ConcreteOp, SameOperandsAndResultElementType> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameOperandsAndResultElementType(op);
    }
};

//
// DMATypeOpInterface
//

std::optional<VPUIP::DmaChannelType> getChannelType(mlir::Operation* op);

}  // namespace VPUIP
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/ops_interfaces.hpp.inc>

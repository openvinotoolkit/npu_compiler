//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/convert_op_types.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dynamic_shape_propagation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace vpux::VPU {
#define GEN_PASS_DECL_DYNAMICDIMSMASKTOBOUNDEDTENSORS
#define GEN_PASS_DEF_DYNAMICDIMSMASKTOBOUNDEDTENSORS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class DynamicDimsMaskToBoundedTensors final :
        public VPU::impl::DynamicDimsMaskToBoundedTensorsBase<DynamicDimsMaskToBoundedTensors> {
public:
    explicit DynamicDimsMaskToBoundedTensors(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    Logger _log;
};

void DynamicDimsMaskToBoundedTensors::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();

    module.walk([&](VPU::BoundsRepresentationInterface op) {
        op.setBoundsRepresentation(VPU::BoundsRepresentation::BOUNDS);
    });

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([&](NDTypeInterface ndType) {
        if (const auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(ndType)) {
            if (!mlir::isa<Core::DynamicDimsMaskTensorType>(sparseType.getData())) {
                return ndType;
            }
        } else if (!mlir::isa<Core::DynamicDimsMaskTensorType>(ndType)) {
            return ndType;
        }

        const auto bounds = ndType.getShape();
        const auto dynamicDims = getDynamicDimsMask(mlir::cast<mlir::Type>(ndType));
        Shape shape;
        shape.reserve(bounds.size());
        for (auto [bound, isDynamic] : llvm::zip(bounds, dynamicDims)) {
            shape.push_back(isDynamic ? mlir::ShapedType::kDynamic : bound);
        }

        auto typeComponents = vpux::TypeComponents().setShape(shape).setBounds(Bounds(bounds.raw()));
        auto outType = ndType.changeTypeComponents(typeComponents);
        return outType;
    });
    mlir::ConversionTarget target(ctx);
    VPU::configureDynamismConversionPassEnvironment(_log, module, typeConverter, target);

    if (mlir::failed(vpux::IE::runConvertOpTypes(module, typeConverter, target, _log))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createDynamicDimsMaskToBoundedTensorsPass(Logger log) {
    return std::make_unique<DynamicDimsMaskToBoundedTensors>(log);
}

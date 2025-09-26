//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/convert_op_types.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/net/network_info_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

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
#define GEN_PASS_DECL_BOUNDEDTENSORSTODYNAMICDIMSMASK
#define GEN_PASS_DEF_BOUNDEDTENSORSTODYNAMICDIMSMASK
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class BoundedTensorsToDynamicDimsMask final :
        public VPU::impl::BoundedTensorsToDynamicDimsMaskBase<BoundedTensorsToDynamicDimsMask> {
public:
    explicit BoundedTensorsToDynamicDimsMask(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    Logger _log;
};

void BoundedTensorsToDynamicDimsMask::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();

    module.walk([&](VPU::BoundsRepresentationInterface op) {
        op.setBoundsRepresentation(VPU::BoundsRepresentation::DYNAMIC_DIMS_MASK);
    });

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([&](NDTypeInterface ndType) {
        if (const auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(ndType)) {
            if (!mlir::isa<Core::BoundedTensorType>(sparseType.getData())) {
                return ndType;
            }
        } else if (!mlir::isa<Core::BoundedTensorType>(ndType)) {
            return ndType;
        }

        const auto shape = ndType.getShape();
        const auto dimsMask = to_small_vector(shape | transformed([&](auto dim) -> int64_t {
                                                  return (dim == mlir::ShapedType::kDynamic) ? 1 : 0;
                                              }));

        const auto bounds = getBoundedShape(ndType);
        auto typeComponents =
                vpux::TypeComponents().setShape(ShapeRef(bounds)).setDynamicDimsMask(DynamicDimsMask(dimsMask));

        auto outType = ndType.changeTypeComponents(typeComponents);
        return outType;
    });

    const auto convert = [](mlir::OpBuilder& builder, mlir::RankedTensorType type, mlir::ValueRange inputs,
                            mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        auto newLocation = appendLoc(loc, "casted");

        auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(newLocation, type, inputs[0]);
        return castOp->getResult(0);
    };

    typeConverter.addSourceMaterialization(convert);
    typeConverter.addTargetMaterialization(convert);
    typeConverter.addArgumentMaterialization(convert);

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    // Mark dialects used by ShaveCodeGen as legal.
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::math::MathDialect, mlir::linalg::LinalgDialect>();

    target.markUnknownOpDynamicallyLegal(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::CallOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp>();
    // TODO: The scf/affine/tensor dialects are explicitly marked as legal because, in the case of the HostCompile
    // pipeline, this pass is executed on the main function, which contains host-side code as well. Ideally, this pass
    // should not operate on the main function in the HostCompile pipeline. This will be refactored in the future.
    // Track: E#168311
    auto hostCompileMode = (config::getCompilationMode(module) == config::CompilationMode::HostCompile);
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalOp<mlir::tensor::ExtractSliceOp>();
    if (hostCompileMode) {
        target.addLegalDialect<mlir::tensor::TensorDialect>();
        target.addLegalDialect<mlir::affine::AffineDialect>();
        target.addLegalOp<mlir::tensor::DimOp>();
        target.addLegalOp<mlir::cf::AssertOp>();
        target.addLegalOp<mlir::UnrealizedConversionCastOp>();
        target.addLegalOp<mlir::func::ReturnOp>();
        target.addLegalOp<mlir::tensor::InsertSliceOp>();
    }

    // We lookup the software module directly to avoid creating it.
    static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};
    auto swModule = module.lookupSymbol<mlir::ModuleOp>(vpuSwModuleName);

    const auto entryFuncOp = vpux::net::findEntryPointFunc(module, _log);
    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp funcOp) {
        if (hostCompileMode && (funcOp == entryFuncOp)) {
            _log.trace("Skipping function {0} in HostCompile mode", funcOp.getName());
            return true;
        }
        if (swModule != nullptr && funcOp->getParentOfType<mlir::ModuleOp>() == swModule && !funcOp.isExternal()) {
            _log.trace("Skipping ShaveCodeGen function {0}", funcOp.getName());
            return true;
        }
        return typeConverter.isSignatureLegal(funcOp.getFunctionType());
    });

    if (swModule != nullptr) {
        // Recursively mark the software module external as legal to prevent interactions
        // with outlined ShaveCodeGen functions. All external functions should be ShaveCodeGen-specific.
        target.markOpRecursivelyLegal<mlir::func::FuncOp>([&](mlir::func::FuncOp funcOp) {
            return funcOp->getParentOfType<mlir::ModuleOp>() == swModule && !funcOp.isExternal();
        });
    }

    if (mlir::failed(vpux::IE::runConvertOpTypes(module, typeConverter, target, _log))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDynamicTensorBoundsToStaticShapePass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createBoundedTensorsToDynamicDimsMaskPass(Logger log) {
    return std::make_unique<BoundedTensorsToDynamicDimsMask>(log);
}

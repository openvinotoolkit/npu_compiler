//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/allocate_buffers_for_net_results.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <functional>

namespace vpux {
#define GEN_PASS_DECL_ADDBUFFERSFORNETRESULTS
#define GEN_PASS_DEF_ADDBUFFERSFORNETRESULTS
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// AddBuffersForNetResults
//

class AddBuffersForNetResults final : public impl::AddBuffersForNetResultsBase<AddBuffersForNetResults> {
public:
    explicit AddBuffersForNetResults(bool useMemrefForHostFunctionBufferization, Logger log)
            : _useMemrefForHostFunctionBufferization(useMemrefForHostFunctionBufferization) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;

private:
    void safeRunOnModule() final;
    bool _useMemrefForHostFunctionBufferization;
};

mlir::LogicalResult AddBuffersForNetResults::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }
    _useMemrefForHostFunctionBufferization = useMemrefForHostFunctionBufferization.getValue();
    return mlir::success();
}

//
// safeRunOnFunc
//

void AddBuffersForNetResults::safeRunOnModule() {
    auto module = getOperation();
    auto entryPointFuncOp = net::getMainFunc(module);

    mlir::DenseSet<mlir::CallOpInterface> callOps;
    module.walk([&](mlir::CallOpInterface callOp) {
        callOps.insert(callOp);
    });
    mlir::DenseSet<mlir::func::FuncOp> funcOps;
    mlir::DenseSet<mlir::func::FuncOp> hostFuncOps;
    module.walk([&](mlir::func::FuncOp funcOp) {
        auto closestModuleParentOp = funcOp->getParentOfType<mlir::ModuleOp>();
        if (_useMemrefForHostFunctionBufferization &&
            (funcOp == entryPointFuncOp || config::isPureHostCompileFunc(funcOp))) {
            hostFuncOps.insert(funcOp);
            return mlir::WalkResult::skip();
        }

        auto moduleName = closestModuleParentOp.getSymName().value_or("");
        if (moduleName == "VPU.SW" || funcOp.isExternal()) {
            /*
            Example of external functions:
            module @VPU.SW {
                func.func private @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
                    attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
            }
            */
            _log.trace("Function '@{0}' at {1} is skipped as it's either external or part of @VPU.SW Module",
                       funcOp.getSymName(), funcOp.getLoc());
            return mlir::WalkResult::skip();
        }
        funcOps.insert(funcOp);
        return mlir::WalkResult::advance();
    });

    VPUIP::updateFuncBoundariesForNetResults(funcOps, _log);
    if (_useMemrefForHostFunctionBufferization) {
        VPUIP::updateFuncBoundariesForNetResults<mlir::memref::CopyOp>(hostFuncOps, _log);
    }
    // Update call sites only after every function boundary (kernels and host entry functions) has been
    // extended, so each result's caller-supplied output buffer already exists and can be reused directly.
    VPUIP::updateCallsForNetResults(callOps, _log);
}

}  // namespace

//
// createAddBuffersForNetResults
//

std::unique_ptr<mlir::Pass> vpux::createAddBuffersForNetResults(bool useMemrefForHostFunctionBufferization,
                                                                Logger log) {
    return std::make_unique<AddBuffersForNetResults>(useMemrefForHostFunctionBufferization, log);
}

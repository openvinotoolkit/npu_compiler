//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Operation.h>

#pragma once

namespace vpux {
namespace VPU {

bool checkForQuantization(mlir::Operation* op, mlir::Type type);
bool checkForQuantization(mlir::Operation* op, mlir::Operation* postOp);
bool hasPerChannelQuantizedOutput(mlir::Operation* op);
bool hasPerChannelQuantizedOutput(mlir::Value output);

template <typename ConcreteModel, typename MainOpType>
class LayerWithPostOpModelBase : public IE::LayerWithPostOpInterface::ExternalModel<ConcreteModel, MainOpType> {
public:
    bool isSupportedClampProperties(mlir::Operation* mainOp, double minValue, double maxValue, mlir::Type type,
                                    const LogCb& logCb) const {
        if (minValue > maxValue) {
            logCb(llvm::formatv("Clamp range [{0}, {1}] of {2} at '{3}' is disjoint", minValue, maxValue,
                                mainOp->getName(), mainOp->getLoc()));
            return false;
        }

        if (config::getCompilationMode(mainOp) == config::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!ConcreteModel::isSupportedHWClampOp(mainOp, minValue, maxValue, type, logCb)) {
            return false;
        }

        return isSupportedOnNCE(mainOp, logCb);
    }

    bool isSupportedClampOp(mlir::Operation* mainOp, mlir::Operation* maybeClampOp, const LogCb& logCb) const {
        auto clampOp = mlir::cast<vpux::IE::ClampOp>(maybeClampOp);

        auto minValue = clampOp.getMinAttr().getValueAsDouble();
        auto maxValue = clampOp.getMaxAttr().getValueAsDouble();
        if (minValue > maxValue) {
            logCb(llvm::formatv("Clamp range [{0}, {1}] at '{2}' is disjoint", minValue, maxValue, clampOp->getLoc()));
            return false;
        }

        if (config::getCompilationMode(clampOp) == config::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!ConcreteModel::isSupportedHWClampOp(mainOp, clampOp, logCb)) {
            return false;
        }

        return isSupportedOnNCE(mainOp, logCb);
    }

    bool isSupportedPostOp(mlir::Operation* mainOp, mlir::Operation* postOp, const LogCb& logCb) const {
        const auto postOpAttr = IE::attributizePostOp(postOp);
        if (postOpAttr == nullptr) {
            logCb(llvm::formatv("{0} at '{1}' cannot be fused as a post-op attribute", postOp->getName(),
                                postOp->getLoc()));
            return false;
        }

        return isSupportedPostOpImpl(mainOp, postOpAttr, postOp->getResult(0), logCb);
    }

    bool isSupportedPostOpAttr(mlir::Operation* mainOp, IE::PostOpAttr postOpAttr, const LogCb& logCb) const {
        if (postOpAttr == nullptr) {
            logCb(llvm::formatv("post-op attribute cannot be fused into {0}", mainOp->getName()));
            return false;
        }

        return isSupportedPostOpImpl(mainOp, postOpAttr, mainOp->getResult(0), logCb);
    }

private:
    bool isSupportedPostOpImpl(mlir::Operation* mainOp, IE::PostOpAttr postOpAttr, mlir::Value fusedOutput,
                               const LogCb& logCb) const {
        if (config::getCompilationMode(mainOp) == config::CompilationMode::ReferenceSW) {
            return false;
        }

        // NOTE: A PPE task evaluates the post-op first and applies the fused clamp afterwards. Fusing a post-op into an
        //       operation that already has a clamp reorders the two. Only when the activation function is monotonically
        //       non-decreasing does the following hold: f(clamp_[min, max](x)) == clamp_[f(min), f(max)](f(x))
        const auto mainOpClamp = mlir::cast<IE::LayerWithPostOpInterface>(mainOp).getClampAttr();
        if (mainOpClamp != nullptr && !IE::isMonotonicActivationFunctionNonDecreasing(postOpAttr)) {
            logCb(llvm::formatv("Post-op '{0}' is not monotonically non-decreasing, it cannot be moved before the "
                                "clamp of {1} at '{2}'",
                                postOpAttr, mainOp->getName(), mainOp->getLoc()));
            return false;
        }

        if (!ConcreteModel::isSupportedHWPostOp(mainOp, postOpAttr, fusedOutput, logCb)) {
            return false;
        }

        return isSupportedOnNCE(mainOp, logCb);
    }

    // NOTE: Some ops (e.g. dilated GroupConvolution) are not directly legal for a plain NCE task, but
    //       can still be lowered to a DPU task through an alternative HW mechanism, such as the
    //       Storage Element (SE) feature.
    static bool isSupportedOnNCE(mlir::Operation* mainOp, const LogCb& logCb) {
        if (VPU::NCEInvariant::isSupported(mlir::cast<MainOpType>(mainOp)).succeeded()) {
            return true;
        }

        auto seOpIfc = mlir::dyn_cast<IE::SEOpInterface>(mainOp);
        if (seOpIfc == nullptr) {
            return false;
        }

        return seOpIfc.isSupported(logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true,
                                   /*checkBatch=*/true);
    }
};

}  // namespace VPU
}  // namespace vpux

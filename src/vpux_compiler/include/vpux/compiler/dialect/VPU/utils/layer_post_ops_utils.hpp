//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Operation.h>
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/utils/logger/logger.hpp"

#pragma once

namespace vpux {
namespace VPU {

bool checkForQuantization(mlir::Operation* op, mlir::Type type);
bool checkForQuantization(mlir::Operation* op, mlir::Operation* postOp);
bool hasPerChannelQuantizedOutput(mlir::Operation* op);

template <typename ConcreteModel, typename MainOpType>
class LayerWithPostOpModelBase : public IE::LayerWithPostOpInterface::ExternalModel<ConcreteModel, MainOpType> {
public:
    bool isSupportedClampProperties(mlir::Operation* mainOp, double minValue, double maxValue, mlir::Type type,
                                    const LogCb& logCb) const {
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

        if (config::getCompilationMode(clampOp) == config::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!ConcreteModel::isSupportedHWClampOp(mainOp, clampOp, logCb)) {
            return false;
        }

        return isSupportedOnNCE(mainOp, logCb);
    }

    bool isSupportedPostOp(mlir::Operation* mainOp, mlir::Operation* postOp, const LogCb& logCb) const {
        if (config::getCompilationMode(postOp) == config::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!ConcreteModel::isSupportedHWPostOp(mainOp, postOp, logCb)) {
            return false;
        }

        return isSupportedOnNCE(mainOp, logCb);
    }

private:
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

//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Operation.h>
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/utils/logger/logger.hpp"

#pragma once

namespace vpux {
namespace VPU {

bool checkForQuantization(mlir::Operation* op, mlir::Operation* postOp);
bool hasPerChannelQuantizedOutput(mlir::Operation* op);
void setHWClampOp(mlir::Operation* mainOp, mlir::Operation* activationOp);
bool isSupportedHWClampOp(mlir::Operation* mainOp, mlir::Operation* clampOp, const LogCb& logCb);

template <typename ConcreteModel, typename MainOpType>
class LayerWithClampOpModel : public IE::LayerWithPostOpInterface::ExternalModel<ConcreteModel, MainOpType> {
public:
    bool isSupportedClampOp(mlir::Operation* mainOp, mlir::Operation* clampOp, const LogCb& logCb) const {
        if (config::getCompilationMode(clampOp) == config::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!VPU::isSupportedHWClampOp(mainOp, clampOp, logCb)) {
            return false;
        }

        return VPU::NCEInvariant::isSupported(mlir::cast<MainOpType>(mainOp)).succeeded();
    }

    void setLayerClampOp(mlir::Operation* mainOp, mlir::Operation* activationOp) const {
        VPU::setHWClampOp(mainOp, activationOp);
    }
};

}  // namespace VPU
}  // namespace vpux

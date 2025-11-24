//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/IR/ops_interfaces.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

namespace {

//
// ExecutorOpModel for NPU37XX
//

class InterpolateExecutorOpModel final :
        public IE::ExecutorOpInterface::ExternalModel<InterpolateExecutorOpModel, IE::InterpolateOp> {
public:
    mlir::SmallVector<VPU::ExecutorKind> getPreferredExecutors(mlir::Operation* origOp) const {
        auto op = mlir::cast<IE::InterpolateOp>(origOp);
        const auto inputShape = getShape(op.getInput());

        // Only support 4D Input shape - if not 4D, prefer SHAVE
        if (inputShape.size() != 4) {
            return {VPU::ExecutorKind::SHAVE_ACT, VPU::ExecutorKind::DPU};
        }

        return {VPU::ExecutorKind::DPU, VPU::ExecutorKind::SHAVE_ACT};
    }
};

}  // namespace

//
// registerExecutorOpInterfaces
//

void vpux::IE::arch37xx::registerExecutorOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::InterpolateOp::attachInterface<InterpolateExecutorOpModel>(*ctx);
    });
}

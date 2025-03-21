//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/unroll_batch.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

namespace {

//
// UnrollBatchOpModel
//

template <class MainOpType>
class UnrollBatchOpModel final :
        public IE::UnrollBatchOpInterface::ExternalModel<UnrollBatchOpModel<MainOpType>, MainOpType> {
public:
    bool doesNeedToUnroll(mlir::Operation* op) const {
        return IE::doesOpNeedToUnroll(op);
    }
};

template <class EltwiseOpType>
class UnrollBatchEltwiseOpModel final :
        public IE::UnrollBatchOpInterface::ExternalModel<UnrollBatchEltwiseOpModel<EltwiseOpType>, EltwiseOpType> {
public:
    bool doesNeedToUnroll(mlir::Operation* op) const {
        return IE::doesEltwiseNeedToUnroll(op);
    }

    size_t getNumberInputs(mlir::Operation* /*op*/) const {
        return 2;
    }
};

class UnrollBatchMemPermuteOpModel final :
        public IE::UnrollBatchOpInterface::ExternalModel<UnrollBatchMemPermuteOpModel, IE::MemPermuteOp> {
public:
    bool doesNeedToUnroll(mlir::Operation* op) const {
        return IE::doesMemPermuteNeedToUnroll(mlir::cast<IE::MemPermuteOp>(op));
    }
};

}  // namespace

//
// registerUnrollBatchOpInterfaces
//

void vpux::VPU::arch37xx::registerUnrollBatchOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::FullyConnectedOp::attachInterface<UnrollBatchOpModel<IE::FullyConnectedOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<UnrollBatchOpModel<IE::GroupConvolutionOp>>(*ctx);
        IE::ExpOp::attachInterface<UnrollBatchOpModel<IE::ExpOp>>(*ctx);
        IE::SigmoidOp::attachInterface<UnrollBatchOpModel<IE::SigmoidOp>>(*ctx);
        IE::InterpolateOp::attachInterface<UnrollBatchOpModel<IE::InterpolateOp>>(*ctx);
        IE::ConvolutionOp::attachInterface<UnrollBatchOpModel<IE::ConvolutionOp>>(*ctx);
        IE::MaxPoolOp::attachInterface<UnrollBatchOpModel<IE::MaxPoolOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<UnrollBatchOpModel<IE::AvgPoolOp>>(*ctx);
        IE::MemPermuteOp::attachInterface<UnrollBatchMemPermuteOpModel>(*ctx);
        IE::AddOp::attachInterface<UnrollBatchEltwiseOpModel<IE::AddOp>>(*ctx);
        IE::MultiplyOp::attachInterface<UnrollBatchEltwiseOpModel<IE::MultiplyOp>>(*ctx);
    });
}

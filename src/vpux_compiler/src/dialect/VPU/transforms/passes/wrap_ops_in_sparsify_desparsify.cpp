//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_WRAPOPSINSPARSIFYDESPARSIFYPAIRS
#define GEN_PASS_DEF_WRAPOPSINSPARSIFYDESPARSIFYPAIRS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

VPU::ActivationSparsityProfile getSparsityProfile(const std::string& sparsityProfile) {
    const auto profile = VPU::symbolizeActivationSparsityProfile(sparsityProfile);
    VPUX_THROW_UNLESS(profile.has_value(), "Unsupported activation sparsity profile '{0}'", sparsityProfile);
    return profile.value();
}

//
// WrapOpsInSparsifyDesparsifyPairsPass
//

class WrapOpsInSparsifyDesparsifyPairsPass final :
        public VPU::impl::WrapOpsInSparsifyDesparsifyPairsBase<WrapOpsInSparsifyDesparsifyPairsPass> {
public:
    WrapOpsInSparsifyDesparsifyPairsPass() = default;
    explicit WrapOpsInSparsifyDesparsifyPairsPass(VPU::EnableActivationSparsityMode enableActivationSparsityMode,
                                                  VPU::ActivationSparsityProfile actSparsityProfile, Logger log)
            : _enableActivationSparsityMode(enableActivationSparsityMode), _sparsityProfile(actSparsityProfile) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    VPU::EnableActivationSparsityMode _enableActivationSparsityMode = VPU::EnableActivationSparsityMode::FALSE;
    VPU::ActivationSparsityProfile _sparsityProfile{VPU::ActivationSparsityProfile::S0};
    VPU::SparsityProfileCreateFunc _sparsityProfileCreateCb;
};

mlir::LogicalResult WrapOpsInSparsifyDesparsifyPairsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (enableActivationSparsityMode.hasValue()) {
        _enableActivationSparsityMode = VPU::getActSparsityMode(enableActivationSparsityMode.getValue());
    }

    if (sparsityProfile.hasValue()) {
        _sparsityProfile = getSparsityProfile(sparsityProfile.getValue());
    }

    return mlir::success();
}

//
// safeRunOnFunc
//

void WrapOpsInSparsifyDesparsifyPairsPass::safeRunOnFunc() {
    using namespace VPU;
    using namespace VPU::NCESparsity;

    auto func = getOperation();
    auto rtStatsHelper = RuntimeSparsityStatsProvider(func, _log);

    // Enable activation sparsity if the option is passed
    // For the AUTO option, only enable if statistics are present inside the module
    if (_enableActivationSparsityMode == VPU::EnableActivationSparsityMode::FALSE) {
        return;
    }
    if (_enableActivationSparsityMode == VPU::EnableActivationSparsityMode::AUTO &&
        !rtStatsHelper.containsStatistics()) {
        return;
    }

    auto arch = VPU::getArch(func);
    auto constraint = VPU::getSparsityConstraint(arch);

    const auto outputWrapper = [&](mlir::Operation* producerOp, mlir::Location loc) {
        mlir::OpBuilder builder(producerOp);
        builder.setInsertionPointAfter(producerOp);

        auto result = producerOp->getResult(0);
        const auto shape = result.getType().cast<vpux::NDTypeInterface>().getShape();
        const auto channels = shape[Dims4D::Act::C];
        if (!constraint.areChannelsFitForSESize(channels)) {
            return;
        }
        const auto sparsifyOp = builder.create<VPU::SparsifyOp>(loc, result);
        const auto desparsifyOp = builder.create<VPU::DesparsifyOp>(loc, sparsifyOp->getResult(0));
        result.replaceAllUsesExcept(desparsifyOp->getResult(0), sparsifyOp);
        _log.trace("Added Desparsify->Sparsify chain for '{0}' output", producerOp->getLoc());
    };
    const auto inputWrapper = [&](mlir::Operation* consumerOp, unsigned int operandId, mlir::Location loc) {
        mlir::OpBuilder builder(consumerOp);

        if (_enableActivationSparsityMode == VPU::EnableActivationSparsityMode::AUTO &&
            !rtStatsHelper.likelySparsityConsumer(consumerOp, operandId)) {
            return;
        }

        const auto producer = consumerOp->getOperand(operandId);
        const auto shape = producer.getType().cast<vpux::NDTypeInterface>().getShape();
        const auto channels = shape[Dims4D::Act::C];
        if (!constraint.areChannelsFitForSESize(producer.getType(), channels)) {
            return;
        }
        const auto sparsifyOp = builder.create<VPU::SparsifyOp>(loc, producer);
        auto newResult = sparsifyOp->getResult(0);
        if (_sparsityProfile == ActivationSparsityProfile::S1) {
            const auto desparsifyOp = builder.create<VPU::DesparsifyOp>(loc, newResult);
            newResult = desparsifyOp->getResult(0);
            _log.trace("Added Desparsify->Sparsify chain for input #{0} of '{1}'", operandId, consumerOp->getLoc());
        } else {
            _log.trace("Added Sparsify for input #{0} of '{1}'", operandId, consumerOp->getLoc());
        }
        consumerOp->setOperand(operandId, newResult);
    };

    func->walk([&](VPU::SparseOpInterface sparsifiableOp) {
        const auto loc = sparsifiableOp->getLoc();
        if (!sparsifiableOp->getResult(0).getType().isa<VPU::SparseTensorType>() &&
            supportsSparseOutputs(sparsifiableOp)) {
            outputWrapper(sparsifiableOp, loc);
        }
        if (!sparsifiableOp->getOperand(0).getType().isa<VPU::SparseTensorType>() &&
            supportsSparseInputs(sparsifiableOp)) {
            inputWrapper(sparsifiableOp, DEFAULT_SPARSIFIABLE_INPUT_OPERAND_ID, loc);
            if (mlir::isa<VPU::NCEEltwiseOp>(sparsifiableOp)) {
                // TODO: handle eltwise weight sparsity
                inputWrapper(sparsifiableOp, ELTWISE_SPARSIFIABLE_SECOND_INPUT_OPERAND_ID, loc);
            }
        }
    });
}

}  // namespace

//
// createWrapOpsInSparsifyDesparsifyPairsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createWrapOpsInSparsifyDesparsifyPairsPass() {
    return std::make_unique<WrapOpsInSparsifyDesparsifyPairsPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createWrapOpsInSparsifyDesparsifyPairsPass(
        VPU::EnableActivationSparsityMode enableActivationSparsityMode,
        VPU::ActivationSparsityProfile actSparsityProfile, Logger log) {
    return std::make_unique<WrapOpsInSparsifyDesparsifyPairsPass>(enableActivationSparsityMode, actSparsityProfile,
                                                                  log);
}

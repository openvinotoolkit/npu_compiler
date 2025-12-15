//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/type/float16.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_LEGALIZEEPSILONUSAGE
#define GEN_PASS_DEF_LEGALIZEEPSILONUSAGE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

template <class OpType>
void processOp(OpType op) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInputs()[0].getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
    if (!inputType.getElementType().isF16() || !outputType.getElementType().isF16()) {
        return;
    }

    constexpr double minEpsilon = static_cast<double>(std::numeric_limits<type::float16>::smallest_mixed_precision_eps);
    auto epsilon = op.getEps().convertToDouble();

    if (epsilon < minEpsilon) {
        auto newEpsilonAttr = getFPAttr(op->getContext(), minEpsilon);
        op.setEpsAttr(newEpsilonAttr);
    }
}

//
// LegalizeEpsilonUsagePass
//

class LegalizeEpsilonUsagePass final : public IE::impl::LegalizeEpsilonUsageBase<LegalizeEpsilonUsagePass> {
public:
    explicit LegalizeEpsilonUsagePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void LegalizeEpsilonUsagePass::safeRunOnFunc() {
    auto func = getOperation();

    // Fix epsilon values that are too small in low-precision IE.NormalizeL2Ops
    func->walk([&](IE::NormalizeL2Op normL2Op) {
        processOp(normL2Op);
    });

    // Fix epsilon values that are too small in low-precision IE.RMSOps
    func->walk([&](IE::RMSOp rmsOp) {
        processOp(rmsOp);
    });
}

}  // namespace

//
// createLegalizeEpsilonUsagePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createLegalizeEpsilonUsagePass(Logger log) {
    return std::make_unique<LegalizeEpsilonUsagePass>(log);
}

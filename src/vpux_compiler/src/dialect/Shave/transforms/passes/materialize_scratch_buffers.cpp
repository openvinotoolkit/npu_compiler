//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/Shave/custom_layer_config.hpp"
#include "vpux/compiler/dialect/Shave/transforms/passes.hpp"
#include "vpux/compiler/dialect/Shave/utils/config_ir_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::Shave {
#define GEN_PASS_DECL_MATERIALIZESCRATCHBUFFERS
#define GEN_PASS_DEF_MATERIALIZESCRATCHBUFFERS
#include "vpux/compiler/dialect/Shave/passes.hpp.inc"
}  // namespace vpux::Shave

using namespace vpux;

namespace {

//
// MaterializeScratchBuffers
//

// Processes a single ExternalKernelOp: creates EmptyOps for each scratch buffer
// described in the op's IOConfig and attaches them as $scratch_buffers operands.
// Returns mlir::WalkResult::advance() always; failures are non-fatal (logged).
//
// Operand layout after this function:
//   [ $inputs ... | $scratch_buffers ... ]
// The "operand_segment_sizes" attribute (required by AttrSizedOperandSegments)
// is updated explicitly to match.
static void processScratchBuffers(Shave::ExternalKernelOp extKernelOp, Logger& log) {
    // Idempotency guard: skip if scratch buffers were already materialised.
    if (!extKernelOp.getScratchBuffers().empty()) {
        return;
    }

    auto configOp = extKernelOp.getCustomOpConfig();
    if (!configOp) {
        return;
    }

    auto ioConfigOps = configOp.getOps<Shave::CustomOpIOConfigOp>();
    if (ioConfigOps.empty()) {
        return;
    }

    Shave::CustomOpIOConfigOp ioConfigOp = *ioConfigOps.begin();
    auto ioConfig = Shave::parseIOConfig(ioConfigOp.getOperation());
    if (ioConfig.scratchConfigs.empty()) {
        return;
    }

    log.debug("Materializing {0} scratch buffer(s) for ExternalKernelOp '{1}'", ioConfig.scratchConfigs.size(),
              extKernelOp.getUniqueId());

    mlir::OpBuilder builder(extKernelOp);
    builder.setInsertionPoint(extKernelOp);
    auto loc = extKernelOp->getLoc();
    auto* ctx = extKernelOp->getContext();

    SmallVector<mlir::Value> scratchValues;
    SmallVector<mlir::Attribute> scratchLocationAttrs;
    scratchValues.reserve(ioConfig.scratchConfigs.size());
    scratchLocationAttrs.reserve(ioConfig.scratchConfigs.size());

    for (const auto& sc : ioConfig.scratchConfigs) {
        auto scratchType = mlir::RankedTensorType::get({1, 1, 1, static_cast<int64_t>(sc.size)}, getUInt8Type(ctx));
        scratchValues.push_back(VPU::createEmptyAuxiliaryBuffer(builder, loc, scratchType));
        scratchLocationAttrs.push_back(Shave::ScratchLocationAttr::get(ctx, sc.location));
    }

    // Assign the $scratch_buffers operand group via the ODS-generated
    // MutableOperandRange. assign() updates both the operand list and the
    // AttrSizedOperandSegments "operand_segment_sizes" bookkeeping attribute
    // atomically, avoiding any window of inconsistency between the two.
    // Using setOperands + manual setAttr would leave the verifier invariant
    // (sum(segment_sizes) == total_operands) temporarily broken.
    extKernelOp.getScratchBuffersMutable().assign(scratchValues);

    // Record the placement policy for each scratch buffer.
    // AdjustMemorySpaceForSHVOps reads this to decide CMX vs DDR vs AUTO.
    // Note: this pass creates all EmptyOps with null memSpace (DDR semantics
    // at the VPU level). The actual CMX/DDR/AUTO placement is applied later
    // by AdjustMemorySpaceForSHVOps via moveExternalKernelIntoCMX.
    extKernelOp.setScratchLocationsAttr(mlir::ArrayAttr::get(ctx, scratchLocationAttrs));
}

class MaterializeScratchBuffers final : public Shave::impl::MaterializeScratchBuffersBase<MaterializeScratchBuffers> {
public:
    explicit MaterializeScratchBuffers(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void MaterializeScratchBuffers::safeRunOnFunc() {
    auto func = getOperation();

    // Collect all ExternalKernelOps before processing to avoid any risk of
    // iterator invalidation caused by operand mutations during the walk.
    SmallVector<Shave::ExternalKernelOp> targets;
    func.walk([&](Shave::ExternalKernelOp op) {
        targets.push_back(op);
    });

    for (auto op : targets) {
        processScratchBuffers(op, _log);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::Shave::createMaterializeScratchBuffersPass(Logger log) {
    return std::make_unique<MaterializeScratchBuffers>(std::move(log));
}

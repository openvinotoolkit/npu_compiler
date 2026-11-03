//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <memory>
#include <utility>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_DETECTIDUOOBACCESSES
#define GEN_PASS_DEF_DETECTIDUOOBACCESSES
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// DetectIDUOOBAccesses
//

class DetectIDUOOBAccesses final : public VPUIP::impl::DetectIDUOOBAccessesBase<DetectIDUOOBAccesses> {
public:
    explicit DetectIDUOOBAccesses(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

protected:
    void safeRunOnFunc() final {
        auto funcOp = getOperation();
        AliasesInfo aliasesInfo{funcOp};

        const auto cmxSize = config::getAvailableMemory(getModuleOp(funcOp), VPU::MemoryKind::CMX_NN).size().count();
        const auto threshold = cmxSize;

        funcOp.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
            _log.trace("Got NCE operation '{0}'", nceOp->getLoc());
            if (nceOp.getOutput() == nullptr || nceOp.getWeights() == nullptr) {
                _log.nest().trace("Operation has no output or weights buffer");
                return;
            }
            if (nceOp.getTaskType() != VPUIP::NCETaskType::CONV && nceOp.getTaskType() != VPUIP::NCETaskType::DWCONV) {
                _log.nest().trace("Skipping non-convolution operation");
                return;
            }
            if (mlir::isa<VPUIP::DistributedBufferType>(nceOp.getWeights().getType())) {
                _log.nest().trace("Distributed weights types are not supported by the pass");
                return;
            }

            auto outputShape = mlir::cast<NDTypeInterface>(nceOp.getOutput().getType()).getShape();
            if (outputShape.size() != 4 ||
                outputShape[Dims4D::Act::C] % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT == 0) {
                _log.nest().trace("Operation does not use ODU autopad");
                return;
            }
            auto weightsShape = mlir::cast<NDTypeInterface>(nceOp.getWeights().getType()).getShape();
            if (weightsShape.size() != 4 ||
                weightsShape[Dims4D::Filter::OC] % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT == 0) {
                _log.nest().trace("Operation has the weights channel dimension aligned to a multiple of 16");
                return;
            }

            const auto& roots = aliasesInfo.getRoots(nceOp.getWeights());
            if (roots.size() != 1) {
                _log.nest().trace("Expected a single root buffer for the weights, got {0}", roots.size());
                return;
            }
            const auto root = roots.front();
            auto bufferOp = mlir::dyn_cast_if_present<VPURT::DeclareBufferOp>(root.getDefiningOp());
            if (bufferOp == nullptr) {
                _log.nest().trace("Could not find buffer operations for weights");
                return;
            }
            if (bufferOp.getSection() != VPURT::BufferSection::CMX_NN) {
                _log.nest().trace("Weights are not in CMX");
                return;
            }
            const auto offset = bufferOp.getByteOffset();

            const auto oc = weightsShape[Dims4D::Filter::OC];
            const auto alignedOc = alignValUp(oc, VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT);
            Shape padBefore{0, 0, 0, 0};
            Shape padAfter{alignedOc - oc, 0, 0, 0};
            auto alignedWeightsType = mlir::cast<NDTypeInterface>(nceOp.getWeights().getType());
            alignedWeightsType = alignedWeightsType.pad(padBefore, padAfter);
            const auto alignedWeightsSize = alignedWeightsType.getTotalAllocSize().count();
            _log.nest().trace("Weights offset: {0}; aligned weights size: {1}; aligned weights type: {2}", offset,
                              alignedWeightsSize, alignedWeightsType);

            if (offset + alignedWeightsSize > threshold) {
                _log.nest().info(
                        "Weights of NCE operation '{0}' need out-of-bounds read accesses. Address range "
                        "[{1}; {2}), CMX end boundary {3}. Weights OC dimension: {4}, aligned OC dimension: {5}",
                        nceOp->getLoc(), offset, offset + alignedWeightsSize, threshold,
                        weightsShape[Dims4D::Filter::OC], alignedOc);
            } else {
                _log.nest().trace("No out-of-bounds accesses detected");
                const auto remainingSize = threshold - (offset + alignedWeightsSize);
                if (remainingSize < 1024) {  // 1 KB
                    _log.nest().info("Weights of NCE operation '{0}' are near the CMX boundary: {1} bytes left",
                                     nceOp->getLoc(), remainingSize);
                }
            }
        });
    }
};

}  // namespace

//
// createDetectIDUOOBAccessesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDetectIDUOOBAccessesPass(Logger log) {
    return std::make_unique<DetectIDUOOBAccesses>(std::move(log));
}

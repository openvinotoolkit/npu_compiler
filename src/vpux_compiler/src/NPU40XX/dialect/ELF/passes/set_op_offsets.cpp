//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <vpux_elf/types/vpu_extensions.hpp>

#include <mlir/IR/SymbolTable.h>

namespace vpux::ELF::arch40xx {
#define GEN_PASS_DECL_SETOPOFFSETS
#define GEN_PASS_DEF_SETOPOFFSETS
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF::arch40xx

using namespace vpux;

namespace {

class SetOpOffsetsPass : public ELF::arch40xx::impl::SetOpOffsetsBase<SetOpOffsetsPass> {
public:
    explicit SetOpOffsetsPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    explicit SetOpOffsetsPass(Logger log, bool enableComputeTaskBufferOffsets)
            : _log(log), _computeTaskBufferOffsets(enableComputeTaskBufferOffsets) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    Logger _log;
    bool _computeTaskBufferOffsets = false;
};

mlir::LogicalResult SetOpOffsetsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (computeTaskBufferOffsets.hasValue()) {
        _computeTaskBufferOffsets = computeTaskBufferOffsets.getValue();
    }

    return mlir::success();
}

void SetOpOffsetsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    mlir::MLIRContext* ctx = &getContext();
    const auto arch = VPU::getArch(netFunc);

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    ELF::SymbolReferenceMap symRefMap(elfMain, true);

    auto u64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    SmallVector<ELF::ElfSectionInterface> metadataSecVec;

    for (auto section : elfMain.getOps<ELF::ElfSectionInterface>()) {
        auto block = section.getBlock();

        if (section.getSectionType().value_or(ELF::SectionTypeAttr::SHT_NULL) ==
            ELF::SectionTypeAttr::VPU_SHT_CMX_METADATA) {
            metadataSecVec.push_back(section);
            continue;
        }

        uint64_t tracker = 0;
        for (auto& operation : block->getOperations()) {
            auto offsetAttr = mlir::IntegerAttr::get(u64Type, mlir::APInt(64, tracker, false));
            if (auto binaryOperation = mlir::dyn_cast_or_null<ELF::BinaryOpInterface>(&operation)) {
                tracker += binaryOperation.getBinarySizeCached(symRefMap, arch);
                binaryOperation.setMemoryOffset(offsetAttr);
            }
        }
    }

    // In standard flow, offsets for task buffers should already be set up at this stage
    // In WLM flow, they are not set up yet and need to be computed in this pass, thus the computeTaskBufferOffsets flag
    // should be set
    if (_computeTaskBufferOffsets) {
        // In case of networks e.g. SOL or LIT tests with only DMA tasks we do not have a CMX Metadata Section
        // For such cases we don't need to setup anything at this stage. It could either be empty or has to be exactly 1
        // For WLM we use DMAs to copy task descriptors to the metadata buffer in CMX, but if the DMA task desc has to
        // be in CMX before it can run we're stuck
        if (metadataSecVec.empty()) {
            return;
        }
        VPUX_THROW_WHEN(metadataSecVec.size() != 1, "Only 1 metadata section is permitted!");
        auto metadataSecBlock = metadataSecVec[0].getBlock();

        size_t offsetTracker = 0;
        size_t tileTracker = 0;
        for (auto taskBufferOp : metadataSecBlock->getOps<VPUASM::DeclareTaskBufferOp>()) {
            auto offsetAttr = mlir::IntegerAttr::get(u64Type, mlir::APInt(64, offsetTracker, false));
            if (tileTracker != taskBufferOp.getTileIndex()) {
                tileTracker = taskBufferOp.getTileIndex();
                offsetTracker = 0;
                offsetAttr = mlir::IntegerAttr::get(u64Type, mlir::APInt(64, offsetTracker, false));
                offsetTracker += taskBufferOp.getBinarySize(arch);
            } else {
                offsetTracker += taskBufferOp.getBinarySize(arch);
            }
            taskBufferOp.setMemoryOffset(offsetAttr);
        }
    }
}

}  // namespace

//
// createAddELFSymbolTablePass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createSetOpOffsetsPass(Logger log, bool enableComputeTaskBufferOffsets) {
    return std::make_unique<SetOpOffsetsPass>(log, enableComputeTaskBufferOffsets);
}

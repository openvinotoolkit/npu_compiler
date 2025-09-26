//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_COMPRESSDMARESERVEMEM
#define GEN_PASS_DEF_COMPRESSDMARESERVEMEM
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
//  CompressDmaReserveMemPass
//

class CompressDmaReserveMemPass final : public VPU::impl::CompressDmaReserveMemBase<CompressDmaReserveMemPass> {
public:
    explicit CompressDmaReserveMemPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void CompressDmaReserveMemPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    _log.trace("Compressed DMA reserved memory - size: '{0}'", ACT_COMPRESSION_RESERVED_MEM_SIZE);

    config::setCompressDmaReservedMemory(module, memSpaceAttr, ACT_COMPRESSION_RESERVED_MEM_SIZE);
}

}  // namespace

//
// createCompressDmaReserveMemPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCompressDmaReserveMemPass(Logger log) {
    return std::make_unique<CompressDmaReserveMemPass>(log);
}

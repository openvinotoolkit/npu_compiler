//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_SWKERNELINSTRUCTIONPREFETCHRESERVEMEMFORDUMMYKERNELS
#define GEN_PASS_DEF_SWKERNELINSTRUCTIONPREFETCHRESERVEMEMFORDUMMYKERNELS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// SWKernelInstructionPrefetchReserveMemForDummyKernels
//
class SWKernelInstructionPrefetchReserveMemForDummyKernels final :
        public VPU::impl::SWKernelInstructionPrefetchReserveMemForDummyKernelsBase<
                SWKernelInstructionPrefetchReserveMemForDummyKernels> {
public:
    explicit SWKernelInstructionPrefetchReserveMemForDummyKernels(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void SWKernelInstructionPrefetchReserveMemForDummyKernels::safeRunOnModule() {
    auto module = getOperation();

    auto* ctx = module->getContext();
    auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    config::setDummySwKernelsForInstructionPrefetchReservedMemory(module, memSpaceAttr,
                                                                  vpux::VPUIP::MAX_SW_KERNEL_DUMMY_KERNELS_DATA_SIZE);
}

}  // namespace

//
// createSWKernelInstructionPrefetchReserveMemForDummyKernelsPass
//
std::unique_ptr<mlir::Pass> vpux::VPU::createSWKernelInstructionPrefetchReserveMemForDummyKernelsPass(Logger log) {
    return std::make_unique<SWKernelInstructionPrefetchReserveMemForDummyKernels>(log);
}

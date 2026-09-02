//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_CMXMETADATARESERVEMEM
#define GEN_PASS_DEF_CMXMETADATARESERVEMEM
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
//  CMXMetadataReserveMemPass
//

class CMXMetadataReserveMemPass final : public VPU::impl::CMXMetadataReserveMemBase<CMXMetadataReserveMemPass> {
public:
    explicit CMXMetadataReserveMemPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void CMXMetadataReserveMemPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    if (config::getArch(moduleOp) < config::ArchKind::NPU40XX) {
        return;
    }
    auto metadataSize = static_cast<uint32_t>(CMX_METADATA_SIZE.count());

    _log.trace("Network Metadata reserved CMX memory - size: '{0}'", metadataSize);
    config::setCMXMetadataReservedMemory(moduleOp, metadataSize, ELF::VPUX_METADATA_ALIGNMENT);
}

}  // namespace

//
// createCMXMetadataReserveMemPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCMXMetadataReserveMemPass(Logger log) {
    return std::make_unique<CMXMetadataReserveMemPass>(log);
}

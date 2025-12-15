//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// ConfigureBarrierOp
//

void vpux::NPUReg50XX::ConfigureBarrierOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto barrierDescriptor = getProperties().getDescriptor();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuBarrierCountConfig) == barrierDescriptor.size(),
                      "HW VpuBarrierCountConfig size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuBarrierCountConfig), barrierDescriptor.size());

    auto serializedBarrierDescriptor = barrierDescriptor.getStorage();
    binDataSection.appendData(serializedBarrierDescriptor.data(), getBinarySize(config::ArchKind::NPU50XX));
}

size_t vpux::NPUReg50XX::ConfigureBarrierOp::getBinarySize(config::ArchKind) {
    return sizeof(nn_public::VpuBarrierCountConfig);
}

size_t vpux::NPUReg50XX::ConfigureBarrierOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(nn_public::VpuBarrierCountConfig);
}

void vpux::NPUReg50XX::ConfigureBarrierOp::build(mlir::OpBuilder&, mlir::OperationState& state,
                                                 mlir::StringAttr symName,
                                                 vpux::NPUReg50XX::Descriptors::VpuBarrierCountConfig&& descriptor) {
    auto& props = state.getOrAddProperties<Properties>();

    props.sym_name = symName;
    props.descriptor = std::move(descriptor);
}

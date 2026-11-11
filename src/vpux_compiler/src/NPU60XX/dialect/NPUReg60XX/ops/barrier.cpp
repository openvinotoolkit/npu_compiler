//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"

#include <npu_60xx_nnrt.hpp>

using namespace vpux;
using namespace npu60xx;

//
// ConfigureBarrierOp
//

void vpux::NPUReg60XX::ConfigureBarrierOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto barrierDescriptor = getProperties().getDescriptor();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuBarrierCountConfig) == barrierDescriptor.size(),
                      "HW VpuBarrierCountConfig size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuBarrierCountConfig), barrierDescriptor.size());

    auto serializedBarrierDescriptor = barrierDescriptor.getStorage();
    binDataSection.appendData(serializedBarrierDescriptor.data(), getBinarySize(config::ArchKind::NPU60XX));
}

size_t vpux::NPUReg60XX::ConfigureBarrierOp::getBinarySize(config::ArchKind) {
    return sizeof(nn_public::VpuBarrierCountConfig);
}

size_t vpux::NPUReg60XX::ConfigureBarrierOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(nn_public::VpuBarrierCountConfig);
}

void vpux::NPUReg60XX::ConfigureBarrierOp::build(mlir::OpBuilder&, mlir::OperationState& state,
                                                 mlir::StringAttr symName,
                                                 vpux::NPUReg60XX::Descriptors::VpuBarrierCountConfig&& descriptor) {
    auto& props = state.getOrAddProperties<Properties>();

    props.sym_name = symName;
    props.descriptor = std::move(descriptor);
}

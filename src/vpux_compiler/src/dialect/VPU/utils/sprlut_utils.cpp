//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/config/utils/setup_pipeline_options_utils.hpp"

using namespace vpux;

bool VPU::hasSprLUTAttribute(PPEAttr ppeAttr) {
    if (ppeAttr == nullptr) {
        return false;
    }
    auto ppeFpAttr = ::mlir::dyn_cast<VPU::PPEFpAttr>(ppeAttr);
    if (ppeFpAttr == nullptr) {
        return false;
    }
    return ppeFpAttr.getSprlut() != nullptr;
}

Byte VPU::getSprLUTSize(PPEAttr ppeAttr) {
    if (!hasSprLUTAttribute(ppeAttr)) {
        return Byte{0};
    }
    auto ppeFpAttr = ::mlir::dyn_cast<VPU::PPEFpAttr>(ppeAttr);
    auto sprlut = ppeFpAttr.getSprlut();
    auto count = sprlut.getNumElements();
    auto elemType = sprlut.getElementType();
    auto elemSize = getElemTypeSize(elemType).to<Byte>();

    return Byte{count * elemSize.count()};
}

void VPU::addSprLutBufferIfPresent(PPEAttr ppeAttr, SmallVector<Byte>& buffers) {
    auto size = getSprLUTSize(ppeAttr);
    if (size != Byte{0}) {
        buffers.push_back(size);
    }
}

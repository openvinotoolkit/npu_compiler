//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/nce_sparsity_converters.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::NCESparsity::PPEConverterCb VPU::NCESparsity::getPPEConverterCb(config::ArchKind arch,
                                                                     [[maybe_unused]] bool isNewWeightTableFormat) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX: {
        return VPU::arch37xx::getScale;
    }
    case config::ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    }
}

VPU::NCESparsity::BiasConverterCb VPU::NCESparsity::getBiasConverterCb(config::ArchKind arch,
                                                                       [[maybe_unused]] bool isNewWeightTableFormat) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX:
        return VPU::arch37xx::getBias;
    case config::ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    }
}

VPU::NCESparsity::ScaleRetrieveCb VPU::NCESparsity::getScaleRetrieveCb(config::ArchKind arch,
                                                                       [[maybe_unused]] bool isNewWeightTableFormat) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX: {
        return VPU::arch37xx::retrieveScaleFromTable;
    }
    case config::ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    }
}

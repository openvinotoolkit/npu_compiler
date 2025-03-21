//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/factories/capture_workpoint_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/capture_workpoint_strategy.hpp"

using namespace vpux;

namespace vpux::VPUIP {

std::unique_ptr<ICaptureWorkpointStrategy> createCaptureWorkpointStrategy(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return std::make_unique<VPUIP::arch37xx::CaptureWorkpointStrategy>();
    default:
        return std::make_unique<VPUIP::ICaptureWorkpointStrategy>();
    }
}

}  // namespace vpux::VPUIP

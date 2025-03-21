//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_convert_to_dpu_checker.hpp"
#include "vpux/compiler/dialect/IE/interfaces/fuse_convert_to_dpu_checker.hpp"

namespace vpux {
namespace IE {

std::unique_ptr<FuseConvertToDPUCheckerBase> createFuseConvertToDPUChecker(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX: {
        return std::make_unique<IE::arch37xx::FuseConvertToDPUChecker>();
    }
    default: {
    }
    }
    VPUX_THROW("Unable to create FuseConvertToDPUChecker for arch {0}", arch);
}

}  // namespace IE
}  // namespace vpux

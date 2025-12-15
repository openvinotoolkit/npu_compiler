//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_convert_to_dpu_checker.hpp"
#include "vpux/compiler/dialect/IE/interfaces/fuse_convert_to_dpu_checker.hpp"

namespace vpux {
namespace IE {

std::unique_ptr<FuseConvertToDPUCheckerBase> createFuseConvertToDPUChecker(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX: {
        return std::make_unique<IE::arch37xx::FuseConvertToDPUChecker>();
    }
    default: {
        return std::make_unique<FuseConvertToDPUCheckerBase>();
    }
    }
    VPUX_THROW("Unable to create FuseConvertToDPUChecker for arch {0}", arch);
}

}  // namespace IE
}  // namespace vpux

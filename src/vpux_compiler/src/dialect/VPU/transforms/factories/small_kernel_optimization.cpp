//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/small_kernel_optimization.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/small_kernel_optimization.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/small_kernel_optimization.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/impl/small_kernel_optimization.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

bool VPU::isSmallKernelOptimizationSupported(mlir::Operation* op, const config::ArchKind arch, const int64_t KX,
                                             [[maybe_unused]] const int64_t KY, const int64_t SX,
                                             ArrayRef<VPU::DPUWorkloadOp> workloads) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return VPU::arch37xx::isSmallKernelOptimizationSupported();
    case config::ArchKind::NPU40XX:
        return VPU::arch40xx::isSmallKernelOptimizationSupported(op, KX, SX, workloads);
    case config::ArchKind::NPU50XX:
        return VPU::arch50xx::isSmallKernelOptimizationSupported(op, KX, KY, SX, workloads);
    case config::ArchKind::UNKNOWN:
    default:
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
}

bool VPU::doesWorkloadSupportSmallKernelOpt([[maybe_unused]] config::ArchKind arch, const int64_t KX, const int64_t SX,
                                            ArrayRef<int64_t> workloadOutSz, bool isFp16Input,
                                            [[maybe_unused]] const int64_t KY, [[maybe_unused]] const int64_t padLeft) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return VPU::arch37xx::doesWorkloadSupportSmallKernelOpt();
    case config::ArchKind::NPU40XX:
        return VPU::arch40xx::doesWorkloadSupportSmallKernelOpt(KX, SX, workloadOutSz, isFp16Input);
    case config::ArchKind::NPU50XX:
        return VPU::arch50xx::doesWorkloadSupportSmallKernelOpt(KX, SX, workloadOutSz, isFp16Input, KY, padLeft);
    case config::ArchKind::UNKNOWN:
    default:
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
}

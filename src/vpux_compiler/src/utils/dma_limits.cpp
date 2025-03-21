//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/utils/dma_limits.hpp>

namespace vpux::VPUIP::DMA {

const EngineLimits DEFAULT_ENGINE_LIMITS_DEFAULT = {TransferLimits(Byte(0), Byte(0xFFFFFF)),
                                                    DimCountLimits(1, 1),
                                                    StrideCountLimits(0, 0),
                                                    {
                                                            DimLimits(SizeLimits(0, 0xFFFFFF), {}, {}),
                                                    }};

// Default limits for NPU37XX (no acceleration mode)
const EngineLimits NPU37XX_ENGINE_LIMITS_DEFAULT = {
        // Actual maximum transferable value based on maximum values of individual dim and stride limits is 0xFFFFFF00
        TransferLimits(Byte(0), Byte(0xFFFFFF00)),
        // Minimum amount of dims is normally always 1.
        // More often than not, size 0 transfers are acceptable, but even in this case, the transfer makes use of 1
        DimCountLimits(1, 3), StrideCountLimits(0, 2),
        mlir::SmallVector({
                /* 0 */
                // First dim does not benefit from +1 to the actual value configured for the HW.
                // This is the "len" dim. It can potentially have a subsequent sub-dim, called "line" dim.
                DimLimits(SizeLimits(0, 0xFFFFFF), StrideLimits(-1 * static_cast<int64_t>(0x80000000), 0x7FFFFFFF),
                          DimLimits::SubLimits(SizeLimits(0, 0xFFFFFF),
                                               StrideLimits(-1 * static_cast<int64_t>(0x80000000), 0x7FFFFFFF))),

                /* 1 */
                // This is the "planes" dim
                DimLimits(SizeLimits(1, 0x100), StrideLimits(-1 * static_cast<int64_t>(0x80000000), 0x7FFFFFFF), {}),
                /**/
        })};

// Default limits for NPU40XX (no acceleration mode)
const EngineLimits NPU40XX_ENGINE_LIMITS_DEFAULT = {
        // Actual maximum transfer limit is 0xFFFFFFFF00000000
        // Transfer limit based on individual dim limits would exceed 64-bit representation, but since all strides are
        // 32-bit, the actual transfer limit is the one above.
        // Limit to a "reasonable" value of 128 GB
        TransferLimits(Byte(0), GB(128).to<Byte>() - Byte(1)),
        // Minimum amount of dims is normally always 1.
        // More often than not, size 0 transfers are acceptable, but even in this case, the transfer makes use of 1 dim.
        DimCountLimits(1, 6), StrideCountLimits(0, 5),
        mlir::SmallVector({
                /* 0 */
                // First dim does not benefit from +1 to the actual value configured for the HW
                DimLimits(SizeLimits(0, 0xFFFFFFFF), {}, {}),
                /* 1 */
                DimLimits(SizeLimits(1, 0x100000000), StrideLimits(0, 0xFFFFFFFF), {}),
                /* 2 */
                DimLimits(SizeLimits(1, 0x100000000), StrideLimits(0, 0xFFFFFFFF), {}),
                /* 3 */
                DimLimits(SizeLimits(1, 0x10000), StrideLimits(0, 0xFFFFFFFF), {}),
                /* 4 */
                DimLimits(SizeLimits(1, 0x10000), StrideLimits(0, 0xFFFFFFFF), {}),
                /* 5 */
                DimLimits(SizeLimits(1, 0x10000), StrideLimits(0, 0xFFFFFFFF), {}),
                /**/
        })};
const EngineLimits& getEngineLimits(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return NPU37XX_ENGINE_LIMITS_DEFAULT;
    case VPU::ArchKind::NPU40XX:
        return NPU40XX_ENGINE_LIMITS_DEFAULT;
    default:
        return DEFAULT_ENGINE_LIMITS_DEFAULT;
    }
}

}  // namespace vpux::VPUIP::DMA

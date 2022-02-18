//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "attributes.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#pragma once

namespace vpux {
namespace VPUIP {

struct WorkloadCostParams {
    bool isZTilingSupported;
    VPUIP::NCETaskType nceTaskType;
    mlir::Type dataType;
    VPU::ArchKind arch;
    VPU::MPEMode mpeMode;
    Shape inputShape;
    Shape outputShape;
    PadInfo padInfo;
    int64_t numDPU;
    SmallVector<int64_t> kernelSize;
    SmallVector<int64_t> kernelStride;
};

class DpuTiler final {
public:
    DpuTiler(ShapeRef outShape, VPU::MPEMode mpeMode): _outShape(outShape.raw()), _mpeMode(mpeMode) {
    }

    SmallVector<uint32_t> generateSplitNumberPool(int64_t numDPU, uint32_t maxSplits);
    void tileOverH(int64_t numDPU);
    void tileOverZ(uint32_t splitNumber);
    SmallVector<OutputTiling> getSplitPool();

    uint32_t cost(const OutputTiling& dpuTiles, const WorkloadCostParams& params);
    double simpleCost(const OutputTiling& dpuTiles, const WorkloadCostParams& params);

private:
    std::pair<uint8_t, uint8_t> getMode();

    Shape _outShape;
    VPU::MPEMode _mpeMode;
    SmallVector<OutputTiling> _splitPool;
};

}  // namespace VPUIP
}  // namespace vpux

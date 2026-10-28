//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/utils.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/descriptors.hpp"

namespace vpux {
namespace NPUReg50XX {

VPUMI40XX::TaskDescriptorConstraintsPassArgs getTaskDescriptorConstraintsArgs() {
    using namespace Descriptors;
    return {DpuInvariantRegister{}.size(), DpuVariantRegister{}.size(), VpuActKernelInvocation{}.size(),
            VpuActKernelRange{}.size(),    DMARegister{}.size(),        VpuMediaTask{}.size()};
}

}  // namespace NPUReg50XX
}  // namespace vpux

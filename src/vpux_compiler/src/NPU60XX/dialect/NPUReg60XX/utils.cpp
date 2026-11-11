//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/utils.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/descriptors.hpp"

namespace vpux {
namespace NPUReg60XX {

VPUMI40XX::TaskDescriptorConstraintsPassArgs getTaskDescriptorConstraintsArgs() {
    using namespace Descriptors;
    return {DpuInvariantRegister{}.size(), DpuVariantRegister{}.size(), VpuActKernelInvocation{}.size(),
            VpuActKernelRange{}.size(), DMARegister{}.size()};
}

}  // namespace NPUReg60XX
}  // namespace vpux

//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#pragma once

#include <openvino/itt.hpp>

namespace vpux {
namespace itt {
namespace domains {

OV_ITT_DOMAIN(VPUXPlugin);
OV_ITT_DOMAIN(VpualBackend);
OV_ITT_DOMAIN(LevelZeroBackend);
OV_ITT_DOMAIN(ImdBackend);

}  // namespace domains
}  // namespace itt
}  // namespace vpux

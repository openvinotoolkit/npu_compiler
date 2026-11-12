//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstdlib>

namespace npu60xx {

// clang-format off

#include <details/api/60xx/vpu_dma_hw_40xx.h>
#include <details/api/60xx/vpu_media_hw.h>
#include <details/api/60xx/vpu_nce_hw_60xx.h>
#include <details/api/60xx/vpu_nnrt_api.h>
#include <details/api/60xx/vpu_nnrt_api_40xx.h>
#include <details/api/60xx/vpu_nnrt_api_ver.h>
#include <details/api/60xx/vpu_nnrt_wlm.h>
#include <details/api/60xx/vpu_pwrmgr_api.h>
#include <details/api/60xx/vpu_nnrt_shavert.h>

#define VPU_CONCAT_NNRT_API_VER(MAJOR, MINOR) (((MAJOR) << 16) | (MINOR))

// clang-format on

}  // namespace npu60xx

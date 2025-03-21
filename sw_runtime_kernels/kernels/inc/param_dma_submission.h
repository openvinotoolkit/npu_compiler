//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <common_types.h>
#include <stddef.h>

#ifdef __cplusplus
namespace sw_params {
#endif

struct __attribute__((packed)) DmaSubmissionParams {
    struct MemRefData input;
    struct MemRefData dmaDescriptor;
    struct MemRefData output;

    int64_t waitEnd;
};

#ifdef __cplusplus
}  // namespace sw_params
#endif

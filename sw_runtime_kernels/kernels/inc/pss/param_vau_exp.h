//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <common_types.h>

#ifdef __cplusplus
namespace sw_params {
#endif

#pragma pack(push, 1)

struct VauExpParams {
    enum { NumInputs = 1, NumOutputs = 1, NumTensors = NumInputs + NumOutputs };
    struct MemRefData tensors[NumTensors];
};

#pragma pack(pop)

inline struct BaseKernelParams ToBaseKernelParams(struct VauExpParams * params) {
    struct BaseKernelParams result;
    result.numInputs = VauExpParams::NumInputs;
    result.numOutputs = VauExpParams::NumOutputs;
#ifdef  __cplusplus
    result.inputsOffset = reinterpret_cast<uint8_t*>(params->tensors + 0) - reinterpret_cast<uint8_t*>(params);
    result.outputsOffset = reinterpret_cast<uint8_t*>(params->tensors + VauExpParams::NumInputs) - reinterpret_cast<uint8_t*>(params);
#else
    result.inputsOffset = (uint8_t*)(params->tensors + 0) - (uint8_t*)(params);
    result.outputsOffset = (uint8_t*)(params->tensors + VauExpParams::NumInputs) - (uint8_t*)(params);
#endif
    return result;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif

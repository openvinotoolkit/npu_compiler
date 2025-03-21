//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#ifdef __MOVICOMPILE__
#include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

#include <common_types.h>
#include <cstddef>

namespace sw_params {

#pragma pack(push, 1)
constexpr size_t MAX_TENSOR_COUNT = 10;
/*
    MAX_TENSOR_COUNT = 10 due to:
    struct MemRefData inputs[4]; - inputROI, inputPyramid, inputPyramidScale1, inputPyramidScale2 (variable inputs)
    struct MemRefData reorderedRois; // auxiliary buffer
    struct MemRefData originalRoiMap; // auxiliary buffer
    struct MemRefData outputRoisFeaturesTemp; // auxiliary buffer
    struct MemRefData levels; //auxiliary buffer

    struct MemRefData outputs[2]; - outputFeatures, outputROIs

*/
struct ExperimentalDetectronROIFeatureExtractorData {
    struct MemRefData tensors[MAX_TENSOR_COUNT];

};

struct ExperimentalDetectronROIFeatureExtractorAttributes {
    int64_t outputSize;
    int64_t samplingRatio;
    int64_t aligned;
    int64_t pyramidScales[3]; // size 3 because the maximum feature inputs are for now only 3
};

#pragma pack(pop)

}  // namespace sw_params

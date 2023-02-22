//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <ie_blob.h>

namespace utils {

void argMax_channels(const InferenceEngine::MemoryBlob::Ptr blob, std::vector<uint8_t>& resultArgmax);

template <typename T>
std::vector<float> mean_IoU(std::vector<T> actOutput, std::vector<T> refOutput, uint32_t classes,
                            uint32_t ignoreLabel) {
    std::vector<uint8_t> output;

    for (size_t i = 0; i < refOutput.size(); i++) {
        auto mask = (refOutput[i] < classes) & (actOutput[i] < classes);

        if (mask == 1) {
            output.push_back(classes * refOutput[i] + actOutput[i]);
        }
    }

    std::vector<int> binC(classes * classes, 0);
    for (size_t i = 0; i < output.size(); i++) {
        binC[output[i]]++;
    }

    std::vector<std::vector<int>> hist(classes, std::vector<int>(classes));
    for (size_t i = 0; i < classes; i++) {
        for (size_t j = 0; j < classes; j++) {
            hist[i][j] = binC[i * classes + j];
        }
    }

    if (ignoreLabel != std::numeric_limits<uint32_t>::max()) {
        // Sanity check
        IE_ASSERT(ignoreLabel < classes);
        for (size_t i = 0; i < classes; i++) {
            hist[i][ignoreLabel] = 0;
            hist[ignoreLabel][i] = 0;
        }
    }

    std::vector<float> diagonal(classes, 0.0f);
    std::vector<float> sum0(classes, 0.0f);
    std::vector<float> sum1(classes, 0.0f);
    for (size_t i = 0; i < classes; i++) {
        for (size_t j = 0; j < classes; j++) {
            if (i == j) {
                diagonal[i] = static_cast<float>(hist[i][j]);
            }
            sum0[j] += static_cast<float>(hist[i][j]);
            sum1[i] += static_cast<float>(hist[i][j]);
        }
    }

    std::vector<float> unionVect(classes, 0.0f);
    for (size_t i = 0; i < sum0.size(); i++) {
        unionVect[i] = sum1[i] + sum0[i] - diagonal[i];
    }

    std::vector<float> iou(classes, 0.0f);
    for (size_t i = 0; i < diagonal.size() - 1; i++) {
        if (unionVect[i] != 0) {
            iou[i] = (diagonal[i] / unionVect[i]) * 100.0f;
        }
    }

    return iou;
}

}  // namespace utils

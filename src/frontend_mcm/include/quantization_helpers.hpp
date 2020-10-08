//
// Copyright 2019-2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once
#include <legacy/ie_layers_internal.hpp>
#include <low_precision_transformations/network_helper.hpp>
#include <low_precision_transformations/quantization_details.hpp>
#include <mcm/tensor/quantization_params.hpp>
#include <string>
#include <vector>

namespace vpu {

namespace QuantizationHelpers {

// for symmetric case only, using mcm logic
int64_t calculateZeroPoint(float high, float low, int levels, InferenceEngine::Precision precision);

bool isCNNNetworkQuantized(const InferenceEngine::CNNNetwork& network);

}  // namespace QuantizationHelpers
}  // namespace vpu

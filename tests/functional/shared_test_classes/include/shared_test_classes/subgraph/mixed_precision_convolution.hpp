//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <vpux/utils/core/error.hpp>
#include "common_test_utils/test_enums.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

enum class LowFpType { NF4, F4E2M1, Undefined };

inline std::string lowFpType2String(LowFpType lowFpType) {
    switch (lowFpType) {
    case LowFpType::NF4:
        return "NF4";
    case LowFpType::F4E2M1:
        return "F4E2M1";
    case LowFpType::Undefined:
        return "Undefined";
    default:
        VPUX_THROW("unknown low fp type!");
    }
}

typedef std::tuple<std::vector<size_t>,                      // kernel
                   std::vector<size_t>,                      // stride
                   std::vector<ptrdiff_t>,                   // pad begin
                   std::vector<ptrdiff_t>,                   // pad end
                   std::vector<size_t>,                      // dilation
                   size_t,                                   // output channels
                   LowFpType,                                // low fp type
                   size_t,                                   // quant levels (to lower to I8 or I4)
                   ov::test::utils::QuantizationGranularity  // quant granularity (per tensor/channel)
                   >
        mixedPrecisionConvSpecificParams;
typedef std::tuple<mixedPrecisionConvSpecificParams,  // specific params
                   ov::element::Type,                 // network precision
                   ov::Shape,                         // input shape
                   std::string                        // target device
                   >
        mixedPrecisionConvSubGraphTestParamsSet;

class MixedPrecisionConvSubGraphTest :
        public testing::WithParamInterface<mixedPrecisionConvSubGraphTestParamsSet>,
        virtual public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<mixedPrecisionConvSubGraphTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

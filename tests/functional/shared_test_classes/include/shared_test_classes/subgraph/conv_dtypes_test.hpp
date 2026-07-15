//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

// Quantized convolution test parameters
using convDtypesSpecificParams = std::tuple<ov::element::Type,  // filter storage type
                                            size_t,             // input quantization levels
                                            ov::element::Type,  // output storage type
                                            size_t,             // output quantization levels
                                            bool                // per-axis weights
                                            >;

using convDtypesTestParamsSet = std::tuple<convDtypesSpecificParams,  // specific params
                                           ov::element::Type,         // network precision
                                           ov::Shape                  // input shape
                                           >;

//
//   [Input]                 [Filter]
//      |                        |
//   (FakeQuantize)          (Convert)
//      |                        |
//   (Convolution)     ---   (Multiply(zScale))
//      |
//   (FakeQuantize, if outputStorageType != f16)
//      |
//   Result
//

class ConvDtypesTest : public testing::WithParamInterface<convDtypesTestParamsSet>, virtual public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convDtypesTestParamsSet>& obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset14.hpp"
#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
using namespace ov::test;

namespace ov::test {

//
// Dynamic shape attention tests
// The SDPA kernel supports dynamic e (embedding dim), eV (value embedding dim), and sSL (source sequence length).
// These tests use ov::Dimension ranges on those axes while keeping batch and heads static.
//

struct DynamicAttentionShapeConfig {
    std::vector<BoundedDim> boundedQ;  // Q shape [N, qH, tSL, e] (use _Dyn for dynamic dims)
    std::vector<BoundedDim> boundedK;  // K shape [N, kvH, sSL, e] (use _Dyn for dynamic dims)
    std::vector<BoundedDim> boundedV;  // V shape [N, kvH, sSL, eV] (use _Dyn for dynamic dims)
    ov::Shape maskShape;               // Mask shape (empty = no mask)
    ov::Shape scaleShape;              // Scale shape (empty = no scale)
};

class DynamicAttentionTest : public VpuOv2LayerTest, public testing::WithParamInterface<DynamicAttentionShapeConfig> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DynamicAttentionShapeConfig> obj) {
        const auto& p = obj.param;
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << "_";
        result << "TestIdx=" << obj.index << "_";

        auto qShape = generateTestShape(p.boundedQ, runtimeShapeCallback);
        auto kShape = generateTestShape(p.boundedK, runtimeShapeCallback);
        auto vShape = generateTestShape(p.boundedV, runtimeShapeCallback);

        result << "dynQ=" << qShape.first.to_string() << "_";
        result << "dynK=" << kShape.first.to_string() << "_";
        result << "dynV=" << vShape.first.to_string() << "_";
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 1;
            in_data.resolution = 32768;
            auto tensor = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                  targetInputStaticShapes[i], in_data);
            VpuOv2LayerTest::inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        const auto& params = GetParam();
        inType = outType = ov::element::f16;

        auto shapeQ = generateTestShape(params.boundedQ, runtimeShapeCallback);
        auto shapeK = generateTestShape(params.boundedK, runtimeShapeCallback);
        auto shapeV = generateTestShape(params.boundedV, runtimeShapeCallback);

        std::vector<ov::test::InputShape> inputShapes = {shapeQ, shapeK, shapeV};

        auto paramQ = std::make_shared<ov::op::v0::Parameter>(inType, shapeQ.first);
        auto paramK = std::make_shared<ov::op::v0::Parameter>(inType, shapeK.first);
        auto paramV = std::make_shared<ov::op::v0::Parameter>(inType, shapeV.first);
        ov::ParameterVector funcParams = {paramQ, paramK, paramV};
        ov::OutputVector sdpaInputs = {paramQ, paramK, paramV};

        if (!params.maskShape.empty()) {
            auto paramMask = std::make_shared<ov::op::v0::Parameter>(inType, params.maskShape);
            funcParams.push_back(paramMask);
            inputShapes.push_back({params.maskShape, {params.maskShape}});
            sdpaInputs.push_back(paramMask);
        } else if (!params.scaleShape.empty()) {
            auto emptyMask = std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{});
            funcParams.push_back(emptyMask);
            inputShapes.push_back({ov::PartialShape{}, {ov::Shape{}}});
            sdpaInputs.push_back(emptyMask);
        }

        if (!params.scaleShape.empty()) {
            auto paramScale = std::make_shared<ov::op::v0::Parameter>(inType, params.scaleShape);
            funcParams.push_back(paramScale);
            inputShapes.push_back({params.scaleShape, {params.scaleShape}});
            sdpaInputs.push_back(paramScale);
        }

        init_input_shapes(inputShapes);

        const auto sdpa = std::make_shared<ov::opset14::ScaledDotProductAttention>(sdpaInputs, false);
        const auto result = std::make_shared<ov::op::v0::Result>(sdpa);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, funcParams, "DynamicAttentionTest");
    }

private:
    // Generates a single runtime shape at bound/2.
    // Linked dimensions (sSL, e) across Q/K/V use the same bound, ensuring consistent values.
    static std::vector<int> runtimeShapeCallback(const BoundedDim& boundedDim) {
        return {boundedDim.bound / 2};
    }

    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "decompose-attention=false";
    }
};

TEST_P(DynamicAttentionTest, NPU5010_HW) {
    abs_threshold = 0.012;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicAttention, DynamicAttentionTest,
                         ::testing::ValuesIn(std::vector<DynamicAttentionShapeConfig>{
                                 // Dynamic sSL: K/V sequence length varies, Q fixed
                                 {
                                         /*Q=*/{1, 6, 64, 64},
                                         /*K=*/{1, 6, 256_Dyn, 64},
                                         /*V=*/{1, 6, 256_Dyn, 64},
                                         /*mask=*/{},
                                         /*scale=*/{1},
                                 },
                                 // Dynamic e (embedding dim): Q and K last dim varies
                                 {
                                         /*Q=*/{1, 6, 64, 128_Dyn},
                                         /*K=*/{1, 6, 64, 128_Dyn},
                                         /*V=*/{1, 6, 64, 64},
                                         /*mask=*/{},
                                         /*scale=*/{1},
                                 },
                                 // Dynamic sSL + broadcast mask
                                 {
                                         /*Q=*/{1, 12, 16, 16},
                                         /*K=*/{1, 12, 128_Dyn, 16},
                                         /*V=*/{1, 12, 128_Dyn, 16},
                                         /*mask=*/{1, 1, 1, 1},
                                         /*scale=*/{1},
                                 },
                                 // Dynamic sSL: cross-attention (V has different embedding dim)
                                 {
                                         /*Q=*/{1, 4, 32, 64},
                                         /*K=*/{1, 4, 128_Dyn, 64},
                                         /*V=*/{1, 4, 128_Dyn, 32},
                                         /*mask=*/{},
                                         /*scale=*/{1},
                                 },
                                 // Dynamic tSL: Q target sequence varies
                                 {
                                         /*Q=*/{1, 4, 64_Dyn, 32},
                                         /*K=*/{1, 4, 64, 32},
                                         /*V=*/{1, 4, 64, 32},
                                         /*mask=*/{},
                                         /*scale=*/{1},
                                 },
                         }),
                         DynamicAttentionTest::getTestCaseName);

}  // namespace ov::test

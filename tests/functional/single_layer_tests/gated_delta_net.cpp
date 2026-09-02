//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/gated_delta_net.hpp>

#include <array>

#include <common/print_test_case_name.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>

namespace ov::test {

namespace {

PRETTY_PARAM(Batch, int64_t);
PRETTY_PARAM(SeqLen, int64_t);
PRETTY_PARAM(QkHeads, int64_t);
PRETTY_PARAM(VHeads, int64_t);
PRETTY_PARAM(QkHeadSize, int64_t);
PRETTY_PARAM(VHeadSize, int64_t);
PRETTY_PARAM(Fuse, bool);

using GatedDeltaNetParams = std::tuple<Batch, SeqLen, QkHeads, VHeads, QkHeadSize, VHeadSize, Fuse>;

class GatedDeltaNetLayerTest : public VpuOv2LayerTest, public ::testing::WithParamInterface<GatedDeltaNetParams> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& modelInputs = function->inputs();
        for (size_t i = 0; i < modelInputs.size(); i++) {
            const auto genData = (i == 4) ? ov::test::utils::InputGenerateData(-4, 4, 32, 1)
                                          : ov::test::utils::InputGenerateData(0, 1, 32, 1);
            auto tensor = ov::test::utils::create_and_fill_tensor(modelInputs[i].get_element_type(),
                                                                  targetInputStaticShapes[i], genData);
            inputs.insert({modelInputs[i].get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        const auto& [bP, sP, qkHP, vHP, qkDP, vDP, fuseP] = GetParam();
        const auto b = static_cast<size_t>(bP.value());
        const auto s = static_cast<size_t>(sP.value());
        const auto qkH = static_cast<size_t>(qkHP.value());
        const auto vH = static_cast<size_t>(vHP.value());
        const auto qkD = static_cast<size_t>(qkDP.value());
        const auto vD = static_cast<size_t>(vDP.value());

        const std::vector<ov::Shape> shapes = {
                {b, s, qkH, qkD}, {b, s, qkH, qkD}, {b, s, vH, vD}, {b, vH, qkD, vD}, {b, s, vH}, {b, s, vH},
        };
        init_input_shapes(static_shapes_to_test_representation(shapes));

        const std::array<const char*, 6> names = {"q", "k", "v", "h0", "gate", "beta"};
        ov::ParameterVector params;
        for (size_t i = 0; i < shapes.size(); i++) {
            auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shapes[i]);
            p->set_friendly_name(names[i]);
            params.push_back(p);
        }

        ov::OutputVector args;
        for (auto& p : params) {
            args.emplace_back(p);
        }
        const bool fuse = fuseP.value();
        auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(args, fuse, fuse ? 1e-3F : 1e-6F,
                                                                     fuse ? 2e-3F : 1e-6F);
        gdn->set_friendly_name("gdn");

        ov::ResultVector results;
        for (size_t i = 0; i < gdn->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(gdn->output(i)));
        }
        function = std::make_shared<ov::Model>(results, params, "GatedDeltaNet");
    }
};

TEST_P(GatedDeltaNetLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    enableProfiling();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(smoke_GatedDeltaNet, GatedDeltaNetLayerTest,
                         ::testing::Values(GatedDeltaNetParams{Batch{1}, SeqLen{8}, QkHeads{2}, VHeads{2},
                                                               QkHeadSize{16}, VHeadSize{16}, Fuse{false}},
                                           GatedDeltaNetParams{Batch{1}, SeqLen{16}, QkHeads{2}, VHeads{4},
                                                               QkHeadSize{16}, VHeadSize{16}, Fuse{false}},
                                           GatedDeltaNetParams{Batch{1}, SeqLen{8}, QkHeads{2}, VHeads{2},
                                                               QkHeadSize{16}, VHeadSize{16}, Fuse{true}}),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(GatedDeltaNet_QwenScale, GatedDeltaNetLayerTest,
                         ::testing::Values(GatedDeltaNetParams{Batch{1}, SeqLen{512}, QkHeads{16}, VHeads{16},
                                                               QkHeadSize{128}, VHeadSize{128}, Fuse{true}}),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(GatedDeltaNet_GQAMultiCluster, GatedDeltaNetLayerTest,
                         ::testing::Values(GatedDeltaNetParams{Batch{1}, SeqLen{512}, QkHeads{6}, VHeads{12},
                                                               QkHeadSize{128}, VHeadSize{128}, Fuse{true}}),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(GatedDeltaNet_GQAUnevenMultiCluster, GatedDeltaNetLayerTest,
                         ::testing::Values(GatedDeltaNetParams{Batch{1}, SeqLen{512}, QkHeads{16}, VHeads{32},
                                                               QkHeadSize{128}, VHeadSize{128}, Fuse{true}},
                                           GatedDeltaNetParams{Batch{1}, SeqLen{64}, QkHeads{4}, VHeads{8},
                                                               QkHeadSize{32}, VHeadSize{32}, Fuse{true}}),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(GatedDeltaNet_MCFit, GatedDeltaNetLayerTest,
                         ::testing::Values(GatedDeltaNetParams{Batch{1}, SeqLen{64}, QkHeads{16}, VHeads{16},
                                                               QkHeadSize{32}, VHeadSize{32}, Fuse{true}}),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(GatedDeltaNet_SmallHead, GatedDeltaNetLayerTest,
                         ::testing::Values(GatedDeltaNetParams{Batch{1}, SeqLen{48}, QkHeads{2}, VHeads{2},
                                                               QkHeadSize{16}, VHeadSize{16}, Fuse{true}}),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(GatedDeltaNet_MCSeq, GatedDeltaNetLayerTest,
                         ::testing::Values(GatedDeltaNetParams{Batch{1}, SeqLen{16}, QkHeads{16}, VHeads{16},
                                                               QkHeadSize{128}, VHeadSize{128}, Fuse{true}},
                                           GatedDeltaNetParams{Batch{1}, SeqLen{32}, QkHeads{16}, VHeads{16},
                                                               QkHeadSize{128}, VHeadSize{128}, Fuse{true}},
                                           GatedDeltaNetParams{Batch{1}, SeqLen{64}, QkHeads{16}, VHeads{16},
                                                               QkHeadSize{128}, VHeadSize{128}, Fuse{true}},
                                           GatedDeltaNetParams{Batch{1}, SeqLen{128}, QkHeads{16}, VHeads{16},
                                                               QkHeadSize{128}, VHeadSize{128}, Fuse{true}},
                                           GatedDeltaNetParams{Batch{1}, SeqLen{256}, QkHeads{16}, VHeads{16},
                                                               QkHeadSize{128}, VHeadSize{128}, Fuse{true}}),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(GatedDeltaNet_Decode, GatedDeltaNetLayerTest,
                         ::testing::Values(GatedDeltaNetParams{Batch{1}, SeqLen{1}, QkHeads{16}, VHeads{16},
                                                               QkHeadSize{128}, VHeadSize{128}, Fuse{true}}),
                         PrintTestCaseName());

class GatedDeltaNetDynamicLayerTest :
        public VpuOv2LayerTest,
        public ::testing::WithParamInterface<GatedDeltaNetParams> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& modelInputs = function->inputs();
        for (size_t i = 0; i < modelInputs.size(); i++) {
            const auto genData = (i == 4) ? ov::test::utils::InputGenerateData(-4, 4, 32, 1)
                                          : ov::test::utils::InputGenerateData(0, 1, 32, 1);
            auto tensor = ov::test::utils::create_and_fill_tensor(modelInputs[i].get_element_type(),
                                                                  targetInputStaticShapes[i], genData);
            inputs.insert({modelInputs[i].get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        const auto& [bP, sP, qkHP, vHP, qkDP, vDP, fuseP] = GetParam();
        const int64_t b = bP.value(), sMax = sP.value(), qkH = qkHP.value(), vH = vHP.value(), qkD = qkDP.value(),
                      vD = vDP.value();
        const auto S = ov::Dimension(1, sMax);
        const auto sm = static_cast<size_t>(sMax);

        const std::vector<ov::test::InputShape> inShapes = {
                {{b, S, qkH, qkD}, {{(size_t)b, sm, (size_t)qkH, (size_t)qkD}}},
                {{b, S, qkH, qkD}, {{(size_t)b, sm, (size_t)qkH, (size_t)qkD}}},
                {{b, S, vH, vD}, {{(size_t)b, sm, (size_t)vH, (size_t)vD}}},
                {{b, vH, qkD, vD}, {{(size_t)b, (size_t)vH, (size_t)qkD, (size_t)vD}}},
                {{b, S, vH}, {{(size_t)b, sm, (size_t)vH}}},
                {{b, S, vH}, {{(size_t)b, sm, (size_t)vH}}},
        };
        init_input_shapes(inShapes);

        const std::array<const char*, 6> names = {"q", "k", "v", "h0", "gate", "beta"};
        ov::ParameterVector params;
        for (size_t i = 0; i < inputDynamicShapes.size(); i++) {
            auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[i]);
            p->set_friendly_name(names[i]);
            params.push_back(p);
        }
        ov::OutputVector args;
        for (auto& p : params) {
            args.emplace_back(p);
        }
        const bool fuse = fuseP.value();
        auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(args, fuse, fuse ? 1e-3F : 1e-6F,
                                                                     fuse ? 2e-3F : 1e-6F);
        gdn->set_friendly_name("gdn");
        ov::ResultVector results;
        for (size_t i = 0; i < gdn->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(gdn->output(i)));
        }
        function = std::make_shared<ov::Model>(results, params, "GatedDeltaNetDynamic");
    }
};

TEST_P(GatedDeltaNetDynamicLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(smoke_GatedDeltaNetDynamic, GatedDeltaNetDynamicLayerTest,
                         ::testing::Values(GatedDeltaNetParams{Batch{1}, SeqLen{8}, QkHeads{2}, VHeads{2},
                                                               QkHeadSize{16}, VHeadSize{16}, Fuse{false}}),
                         PrintTestCaseName());

}  // namespace

}  // namespace ov::test

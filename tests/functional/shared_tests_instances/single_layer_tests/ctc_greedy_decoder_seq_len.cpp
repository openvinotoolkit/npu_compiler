//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/ctc_greedy_decoder_seq_len.hpp"
#include <random>
#include "common/npu_test_env_cfg.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"

using namespace ov::test::utils;
namespace ov {
namespace test {

// OpenVino CTCGreedyDecoderSeqLenLayerTest from OpenVino test infrastructire doesn't allow to create CTCDecoderSeqLen
// layer without blankIndex input so we have to create our own test class which allows to do so
typedef std::tuple<ov::Shape,           // Input shape
                   int,                 // Sequence lengths
                   ov::element::Type,   // Probabilities precision
                   ov::element::Type,   // Indices precision
                   std::optional<int>,  // Blank index
                   bool,                // Merge repeated
                   std::string          // Device name
                   >
        NPUCTCGreedyDecoderSeqLenParams;

class NPUCTCGreedyDecoderSeqLenLayerTest :
        public testing::WithParamInterface<NPUCTCGreedyDecoderSeqLenParams>,
        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NPUCTCGreedyDecoderSeqLenParams>& obj) {
        ov::Shape inputShape;
        int sequenceLengths;
        ov::element::Type dataPrecision, indicesPrecision;
        std::optional<int> blankIndex;
        bool mergeRepeated;
        std::string targetDevice;
        std::tie(inputShape, sequenceLengths, dataPrecision, indicesPrecision, blankIndex, mergeRepeated,
                 targetDevice) = obj.param;

        std::ostringstream result;

        result << "IS=" << inputShape.to_string() << '_';
        result << "seqLen=" << sequenceLengths << '_';
        result << "dataPRC=" << dataPrecision.get_type_name() << '_';
        result << "idxPRC=" << indicesPrecision.get_type_name() << '_';
        result << "BlankIdx=" << blankIndex.value_or(-1) << '_';
        result << "mergeRepeated=" << std::boolalpha << mergeRepeated << '_';
        result << "trgDev=" << test_utils::TARGET_DEVICE;

        return result.str();
    }

protected:
    void SetUp() override {
        ov::Shape inputShape;
        int sequenceLengths;
        ov::element::Type dataPrecision, indicesPrecision;
        std::optional<int> blankIndex;
        bool mergeRepeated;
        std::tie(inputShape, sequenceLengths, dataPrecision, indicesPrecision, blankIndex, mergeRepeated,
                 targetDevice) = GetParam();

        init_input_shapes(static_shapes_to_test_representation({inputShape}));
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(dataPrecision, inputShape)};

        const auto sequenceLenNode = [&] {
            const size_t B = inputShape[0];
            const size_t T = inputShape[1];

            // Cap sequence length up to T
            const int seqLen = std::min<int>(T, sequenceLengths);

            std::mt19937 gen{42};
            std::uniform_int_distribution<int> dist(1, seqLen);

            std::vector<int> sequenceLenData(B);
            for (int b = 0; b < B; b++) {
                const int len = dist(gen);
                sequenceLenData[b] = len;
            }

            return std::make_shared<ov::op::v0::Constant>(indicesPrecision, ov::Shape{B}, sequenceLenData);
        }();

        // Cap blank index up to C - 1
        int C = inputShape.at(2);
        if (blankIndex.has_value()) {
            blankIndex.value() = std::min(blankIndex.value(), C - 1);
        }

        const auto blankIndexNode = [&] {
            if (indicesPrecision == ov::element::i32) {
                const auto blankIdxDataI32 = std::vector<int32_t>{blankIndex.value_or(C - 1)};
                return std::make_shared<ov::op::v0::Constant>(indicesPrecision, ov::Shape{1}, blankIdxDataI32);
            } else if (indicesPrecision == ov::element::i64) {
                const auto blankIdxDataI64 = std::vector<int64_t>{blankIndex.value_or(C - 1)};
                return std::make_shared<ov::op::v0::Constant>(indicesPrecision, ov::Shape{1}, blankIdxDataI64);
            }
            throw std::logic_error("Unsupported index precision");
        }();

        auto ctcGreedyDecoderSeqLen =
                blankIndex.has_value()
                        ? std::make_shared<ov::op::v6::CTCGreedyDecoderSeqLen>(params[0], sequenceLenNode,
                                                                               blankIndexNode, mergeRepeated,
                                                                               indicesPrecision, indicesPrecision)
                        : std::make_shared<ov::op::v6::CTCGreedyDecoderSeqLen>(
                                  params[0], sequenceLenNode, mergeRepeated, indicesPrecision, indicesPrecision);
        ov::OutputVector results;
        for (int i = 0; i < ctcGreedyDecoderSeqLen->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(ctcGreedyDecoderSeqLen->output(i)));
        }
        function = std::make_shared<ov::Model>(results, params, "CTCGreedyDecoderSeqLen");
    }
};

class CTCGreedyDecoderSeqLenLayerTestCommon :
        public NPUCTCGreedyDecoderSeqLenLayerTest,
        virtual public VpuOv2LayerTest {};

TEST_P(CTCGreedyDecoderSeqLenLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(CTCGreedyDecoderSeqLenLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(CTCGreedyDecoderSeqLenLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

const std::vector<ov::element::Type> probPrecisions = {ov::element::f16};
const std::vector<ov::element::Type> idxPrecisions = {ov::element::i32, ov::element::i64};

std::vector<bool> mergeRepeated{true, false};

const auto inputShape = std::vector<ov::Shape>{{1, 1, 1}, {4, 80, 80}, {80, 4, 80}, {80, 80, 4}, {8, 20, 128}};

const auto sequenceLengths = std::vector<int>{1, 50, 100};

const auto blankIndexes = std::vector<std::optional<int>>{0, 50, std::nullopt};

const auto params = testing::Combine(::testing::ValuesIn(inputShape), ::testing::ValuesIn(sequenceLengths),
                                     ::testing::ValuesIn(probPrecisions), ::testing::ValuesIn(idxPrecisions),
                                     ::testing::ValuesIn(blankIndexes), ::testing::ValuesIn(mergeRepeated),
                                     ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_CTCGreedyDecoderSeqLenTests, CTCGreedyDecoderSeqLenLayerTestCommon, params,
                         NPUCTCGreedyDecoderSeqLenLayerTest::getTestCaseName);

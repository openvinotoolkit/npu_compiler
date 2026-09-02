//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset14.hpp"
#include "openvino/opsets/opset3.hpp"
#include "vpu_ov2_layer_test.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

using namespace ov::test::utils;
using namespace ov::test;

namespace ov::test {

enum class AttentionType {
    SDPA,    // Use opset14::ScaledDotProductAttention operation
    PATTERN  // Use manual pattern: MatMul -> Scale -> Mask -> Bias -> Softmax -> MatMul
};

enum class AttentionOption {
    SCALE_CONST,    // Create scale as constant (computed: 1/sqrt(d)) instead of parameter
    SCALE_Q,        // Apply scale to Q input (PATTERN only): (scale * Q) @ K.T
    SCALE_K,        // Apply scale to K input (PATTERN only): Q @ (scale * K).T
    CAUSAL_MASK,    // Generate causal mask constant
    CAUSAL_FLAG,    // Set causal flag for SDPA operation (SDPA only)
    BOOLEAN_MASK,   // Use boolean mask type (true=keep, false=mask).
    FULL_LINE_MASK  // Generate mask with some rows fully masked (at FP16 lowest)
};

// Shape configuration separate from AttentionType (for Combine)
struct AttentionShapeConfig {
    ov::Shape inputQ;
    ov::Shape inputK;
    ov::Shape inputV;
    ov::Shape mask = {};
    ov::Shape scale = {};
    ov::Shape bias = {};
    std::vector<AttentionOption> options = {};
};

struct AttentionParams : AttentionShapeConfig {
    AttentionType attentionType = AttentionType::PATTERN;

    AttentionParams(const AttentionShapeConfig& config, AttentionType type)
            : AttentionShapeConfig(config), attentionType(type) {
    }

    AttentionParams() = default;
};

class AttentionTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::tuple<AttentionShapeConfig, AttentionType>> {
public:
    AttentionTestCommon(): log(vpux::Logger::global().nest("AttentionTest", 0)) {
    }

    struct OptionFlags {
        bool scaleConst = false;
        bool scaleQ = false;
        bool scaleK = false;
        bool causalMask = false;
        bool causalFlag = false;
        bool booleanMask = false;
        bool fullLineMask = false;

        explicit OptionFlags(const std::vector<AttentionOption>& options) {
            for (const auto& opt : options) {
                switch (opt) {
                case AttentionOption::SCALE_CONST:
                    scaleConst = true;
                    break;
                case AttentionOption::SCALE_Q:
                    scaleQ = true;
                    break;
                case AttentionOption::SCALE_K:
                    scaleK = true;
                    break;
                case AttentionOption::CAUSAL_MASK:
                    causalMask = true;
                    break;
                case AttentionOption::CAUSAL_FLAG:
                    causalFlag = true;
                    break;
                case AttentionOption::BOOLEAN_MASK:
                    booleanMask = true;
                    break;
                case AttentionOption::FULL_LINE_MASK:
                    fullLineMask = true;
                    break;
                }
            }
        }
    };

    // Helper function to check if an option is present (for validation)
    static bool hasOption(const std::vector<AttentionOption>& options, AttentionOption option) {
        return std::find(options.begin(), options.end(), option) != options.end();
    }

    // Helper function to check if a shape is broadcastable to a target shape.
    static bool isBroadcastableTo(const ov::Shape& shape, const ov::Shape& targetShape) {
        if (shape.empty()) {
            return true;  // Empty shape is considered valid
        }
        if (shape.size() > targetShape.size()) {
            return false;
        }
        const size_t rankDiff = targetShape.size() - shape.size();
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != 1 && shape[i] != targetShape[rankDiff + i]) {
                return false;
            }
        }
        return true;
    }

    // Validate attention parameters
    static void validateAttentionParams(const AttentionParams& params) {
        // Validate SCALE_Q and SCALE_K are mutually exclusive (both control scale position)
        const bool hasScaleQ = hasOption(params.options, AttentionOption::SCALE_Q);
        const bool hasScaleK = hasOption(params.options, AttentionOption::SCALE_K);
        VPUX_THROW_UNLESS(!(hasScaleQ && hasScaleK),
                          "SCALE_Q and SCALE_K are mutually exclusive (both control scale position)");

        // Ensure Q, K, V all have the same rank
        VPUX_THROW_UNLESS(params.inputQ.size() == params.inputK.size() && params.inputK.size() == params.inputV.size(),
                          "Q, K, V must have the same rank. Q: {0}, K: {1}, V: {2}",
                          ov::test::utils::vec2str(params.inputQ), ov::test::utils::vec2str(params.inputK),
                          ov::test::utils::vec2str(params.inputV));

        // Check Q, K, V are 3D or 4D tensors
        const bool is3D = params.inputQ.size() == 3;
        const bool is4D = params.inputQ.size() == 4;

        VPUX_THROW_UNLESS(is3D || is4D, "Q, K, V shapes must be 3D [qH, tSL, e] or 4D [N, qH, tSL, e], got Q: {0}",
                          ov::test::utils::vec2str(params.inputQ));

        size_t N = 1;  // For 3D case, batch size is implicitly 1
        const size_t offset = is4D ? 1 : 0;

        // Extract batch dimension if 4D
        if (is4D) {
            N = params.inputQ[0];
            const size_t K_N = params.inputK[0];
            const size_t V_N = params.inputV[0];
            VPUX_THROW_UNLESS(N == K_N, "Q and K must have matching batch size. Q: {0}, K: {1}", N, K_N);
            VPUX_THROW_UNLESS(N == V_N, "Q and V must have matching batch size. Q: {0}, V: {1}", N, V_N);
        }

        // Extract remaining dimensions using offset (works for both 3D and 4D)
        const size_t qH = params.inputQ[offset];
        const size_t tSL = params.inputQ[offset + 1];
        const size_t e = params.inputQ[offset + 2];

        const size_t kvH = params.inputK[offset];
        const size_t sSL = params.inputK[offset + 1];
        const size_t K_e = params.inputK[offset + 2];

        const size_t V_kvH = params.inputV[offset];
        const size_t V_sSL = params.inputV[offset + 1];

        VPUX_THROW_UNLESS(kvH == V_kvH, "K and V must have matching head count. K: {0}, V: {1}", kvH, V_kvH);
        VPUX_THROW_UNLESS(sSL == V_sSL, "K and V must have matching sequence length. K: {0}, V: {1}", sSL, V_sSL);
        VPUX_THROW_UNLESS(e == K_e, "Q and K must have matching embedding dimension. Q: {0}, K: {1}", e, K_e);

        // Validate mask shape (if provided)
        if (!params.mask.empty()) {
            ov::Shape expectedMaskShape = is4D ? ov::Shape{N, qH, tSL, sSL} : ov::Shape{qH, tSL, sSL};
            VPUX_THROW_UNLESS(isBroadcastableTo(params.mask, expectedMaskShape),
                              "Mask shape is not broadcastable to expected shape. "
                              "Mask: {0}, Expected: {1}",
                              ov::test::utils::vec2str(params.mask), ov::test::utils::vec2str(expectedMaskShape));
        }

        // Validate bias shape (if provided)
        if (!params.bias.empty()) {
            ov::Shape expectedBiasShape = is4D ? ov::Shape{N, qH, tSL, sSL} : ov::Shape{qH, tSL, sSL};
            VPUX_THROW_UNLESS(isBroadcastableTo(params.bias, expectedBiasShape),
                              "Bias shape is not broadcastable to expected shape. "
                              "Bias: {0}, Expected: {1}",
                              ov::test::utils::vec2str(params.bias), ov::test::utils::vec2str(expectedBiasShape));
        }

        // Validate scale shape (if provided) based on position
        if (!params.scale.empty()) {
            const bool hasScaleQ = hasOption(params.options, AttentionOption::SCALE_Q);
            const bool hasScaleK = hasOption(params.options, AttentionOption::SCALE_K);

            if (hasScaleQ) {
                // SCALE_Q: Scale applied to Q input
                VPUX_THROW_UNLESS(isBroadcastableTo(params.scale, params.inputQ) || params.scale == ov::Shape{1},
                                  "Scale shape is not broadcastable to Q shape when SCALE_Q is used. "
                                  "Scale: {0}, Q: {1}",
                                  ov::test::utils::vec2str(params.scale), ov::test::utils::vec2str(params.inputQ));
            } else if (hasScaleK) {
                // SCALE_K: Scale applied to broadcasted K
                ov::Shape expectedKBroadcastShape = is4D ? ov::Shape{N, qH, sSL, K_e} : ov::Shape{qH, sSL, K_e};
                VPUX_THROW_UNLESS(
                        isBroadcastableTo(params.scale, expectedKBroadcastShape) || params.scale == ov::Shape{1},
                        "Scale shape is not broadcastable to broadcasted K shape when SCALE_K is used. "
                        "Scale: {0}, Broadcasted K: {1}",
                        ov::test::utils::vec2str(params.scale), ov::test::utils::vec2str(expectedKBroadcastShape));
            } else {
                // Default: Scale applied after MatMul, must be broadcastable to attention scores shape
                ov::Shape expectedScaleShape = is4D ? ov::Shape{N, qH, tSL, sSL} : ov::Shape{qH, tSL, sSL};
                VPUX_THROW_UNLESS(isBroadcastableTo(params.scale, expectedScaleShape) || params.scale == ov::Shape{1},
                                  "Scale shape is not broadcastable to attention scores shape. "
                                  "Scale: {0}, Expected: {1}",
                                  ov::test::utils::vec2str(params.scale), ov::test::utils::vec2str(expectedScaleShape));
            }
        }
    }

    // Helper function to broadcast K/V tensors for GQA/MQA
    static std::shared_ptr<ov::Node> broadcastHeadsDimension(std::shared_ptr<ov::Node> input,
                                                             const ov::Shape& inputShape, int64_t targetHeads) {
        const bool is3D = inputShape.size() == 3;
        const int64_t inputHeads = is3D ? inputShape[0] : inputShape[1];

        if (inputHeads == targetHeads) {
            return input;
        }

        if (targetHeads % inputHeads != 0) {
            VPUX_THROW("Target heads must be divisible by input heads for GQA/MQA");
        }

        const int64_t repeatFactor = targetHeads / inputHeads;
        const int64_t SL = static_cast<int64_t>(inputShape[is3D ? 1 : 2]);
        const int64_t E = static_cast<int64_t>(inputShape[is3D ? 2 : 3]);

        // Helper to create constant node
        auto makeConst = [](const std::vector<int64_t>& vals) {
            return std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{vals.size()}, vals);
        };

        if (is3D) {
            // 3D: [heads, SL, E] -> [heads, 1, SL, E] -> [heads, repeatFactor, SL, E] -> [targetHeads, SL, E]
            auto reshaped1 = std::make_shared<ov::op::v1::Reshape>(input, makeConst({inputHeads, 1, SL, E}), false);
            auto broadcasted = std::make_shared<ov::op::v3::Broadcast>(
                    reshaped1, makeConst({inputHeads, repeatFactor, SL, E}), ov::op::BroadcastType::BIDIRECTIONAL);
            return std::make_shared<ov::op::v1::Reshape>(broadcasted, makeConst({targetHeads, SL, E}), false);
        } else {
            // 4D: [N, heads, SL, E] -> [N, heads, 1, SL, E] -> [N, heads, repeatFactor, SL, E] -> [N, targetHeads, SL,
            // E]
            const int64_t N = static_cast<int64_t>(inputShape[0]);
            auto reshaped1 = std::make_shared<ov::op::v1::Reshape>(input, makeConst({N, inputHeads, 1, SL, E}), false);
            auto broadcasted = std::make_shared<ov::op::v3::Broadcast>(
                    reshaped1, makeConst({N, inputHeads, repeatFactor, SL, E}), ov::op::BroadcastType::BIDIRECTIONAL);
            return std::make_shared<ov::op::v1::Reshape>(broadcasted, makeConst({N, targetHeads, SL, E}), false);
        }
    }

    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<AttentionShapeConfig, AttentionType>> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        const auto& [shapeConfig, attentionType] = obj.param;
        const AttentionParams params(shapeConfig, attentionType);

        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "Q=" << ov::test::utils::vec2str(params.inputQ) << sep;
        result << "K=" << ov::test::utils::vec2str(params.inputK) << sep;
        result << "V=" << ov::test::utils::vec2str(params.inputV) << sep;

        // Mask shape
        if (!params.mask.empty()) {
            result << "Mask=" << ov::test::utils::vec2str(params.mask) << sep;
        } else {
            result << "Mask=NONE" << sep;
        }

        // Scale shape
        if (!params.scale.empty()) {
            result << "Scale=" << ov::test::utils::vec2str(params.scale) << sep;
        } else {
            result << "Scale=NONE" << sep;
        }

        // Bias shape
        if (!params.bias.empty()) {
            result << "Bias=" << ov::test::utils::vec2str(params.bias) << sep;
        } else {
            result << "Bias=NONE" << sep;
        }

        // SDPA type
        switch (params.attentionType) {
        case AttentionType::SDPA:
            result << "Type=SDPA" << sep;
            break;
        case AttentionType::PATTERN:
            result << "Type=PATTERN" << sep;
            break;
        }

        // Options
        if (!params.options.empty()) {
            result << "Options=";
            for (size_t i = 0; i < params.options.size(); ++i) {
                switch (params.options[i]) {
                case AttentionOption::SCALE_CONST:
                    result << "SCALE_CONST";
                    break;
                case AttentionOption::SCALE_Q:
                    result << "SCALE_Q";
                    break;
                case AttentionOption::SCALE_K:
                    result << "SCALE_K";
                    break;
                case AttentionOption::CAUSAL_MASK:
                    result << "CAUSAL_MASK";
                    break;
                case AttentionOption::CAUSAL_FLAG:
                    result << "CAUSAL_FLAG";
                    break;
                case AttentionOption::BOOLEAN_MASK:
                    result << "BOOLEAN_MASK";
                    break;
                case AttentionOption::FULL_LINE_MASK:
                    result << "FULL_LINE_MASK";
                    break;
                }
                if (i < params.options.size() - 1) {
                    result << "+";
                }
            }
            result << sep;
        }

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        const auto& shapeConfig = std::get<0>(GetParam());
        const OptionFlags opts(shapeConfig.options);

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 1;
            in_data.resolution = 32768;

            ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                            targetInputStaticShapes[i], in_data);

            // For FULL_LINE_MASK, use a hash-based pseudo-random scheme to decide which rows are masked.
            // First and last rows are always masked to guarantee coverage. Remaining rows are masked
            // when a simple hash of the row index falls below ~1/3 of the range, providing non-periodic variety.
            if (opts.fullLineMask && i == 3) {
                const auto& shape = targetInputStaticShapes[i];
                const auto cols = shape.back();
                const auto totalSize = ov::shape_size(shape);
                const auto numRows = totalSize / cols;
                const auto shouldMask = [numRows](size_t row) {
                    if (row == 0 || row == numRows - 1) {
                        return true;
                    }
                    // Mixing function: multiply by a large odd constant and xor-shift.
                    uint32_t h = static_cast<uint32_t>(row) * 2654435761u;
                    h ^= h >> 16;
                    return (h % 3) == 0;
                };
                const auto elemType = funcInputs[i].get_element_type();
                if (elemType == ov::element::f32) {
                    auto* data = tensorData.data<float>();
                    const float maskedValue = std::numeric_limits<float>::lowest();
                    for (size_t row = 0; row < numRows; ++row) {
                        if (shouldMask(row)) {
                            std::fill(data + row * cols, data + (row + 1) * cols, maskedValue);
                        }
                    }
                } else if (elemType == ov::element::f16) {
                    auto* data = tensorData.data<ov::float16>();
                    const ov::float16 maskedValue = std::numeric_limits<ov::float16>::lowest();
                    for (size_t row = 0; row < numRows; ++row) {
                        if (shouldMask(row)) {
                            std::fill(data + row * cols, data + (row + 1) * cols, maskedValue);
                        }
                    }
                }
            }

            VpuOv2LayerTest::inputs.insert({funcInputs[i].get_node_shared_ptr(), tensorData});
        }
    }

    void SetUp() override {
        const auto& [shapeConfig, attentionType] = GetParam();
        const AttentionParams testParams(shapeConfig, attentionType);
        const OptionFlags setupOpts(testParams.options);

        // Run the test model in FP32 to avoid compounding FP16 error from both reference and NPU paths.
        inType = outType = ov::element::f32;
        // Validate parameters
        validateAttentionParams(testParams);

        const auto inputQShape = testParams.inputQ;
        auto inputKShape = testParams.inputK;
        auto inputVShape = testParams.inputV;

        const size_t rank = inputQShape.size();
        const size_t headDim = rank - 3;  // Index for number of heads dimension
        const int64_t qHeads = inputQShape[headDim];
        const int64_t kvHeads = inputKShape[headDim];

        // Evaluate all options in a single pass for efficiency
        const OptionFlags opts(testParams.options);

        // Validate SDPA constraints and force adjustments if needed
        if (testParams.attentionType == AttentionType::SDPA) {
            const bool isGQA = (qHeads > kvHeads) && (kvHeads > 1);
            if (isGQA) {
                log.warning("SDPA op does not support GQA configurations (qHeads={0} > kvHeads={1} > 1). "
                            "Forcing MHA configuration (qHeads == kvHeads).",
                            qHeads, kvHeads);
                // For GQA, adjust K and V to have the same number of heads as Q
                inputKShape[headDim] = qHeads;
                inputVShape[headDim] = qHeads;
            }
        }

        bool useBias = !testParams.bias.empty() && testParams.attentionType != AttentionType::SDPA;
        if (!testParams.bias.empty() && testParams.attentionType == AttentionType::SDPA) {
            log.warning("SDPA op does not support BIAS. Ignoring bias from configuration.");
        }

        std::vector<ov::Shape> inputShapes = {inputQShape, inputKShape, inputVShape};

        ov::ParameterVector params;
        const auto inputQ = std::make_shared<ov::op::v0::Parameter>(inType, inputQShape);
        const auto inputK = std::make_shared<ov::op::v0::Parameter>(inType, inputKShape);
        const auto inputV = std::make_shared<ov::op::v0::Parameter>(inType, inputVShape);
        params.push_back(inputQ);
        params.push_back(inputK);
        params.push_back(inputV);

        // Handle Mask
        std::shared_ptr<ov::Node> attentionMask = nullptr;
        bool isCausal = false;

        if (opts.causalFlag) {
            // CAUSAL_FLAG sets the causal parameter for SDPA op
            if (testParams.attentionType == AttentionType::PATTERN) {
                log.warning("CAUSAL_FLAG option is only supported with AttentionType::SDPA. "
                            "It will be ignored for PATTERN mode.");
            } else {
                isCausal = true;
            }
        }

        // Warn if SCALE_Q or SCALE_K used with SDPA mode (they only work with PATTERN)
        if (testParams.attentionType == AttentionType::SDPA && (opts.scaleQ || opts.scaleK)) {
            log.warning("SCALE_Q and SCALE_K options are only supported with AttentionType::PATTERN. "
                        "They will be ignored for SDPA mode.");
        }

        if (opts.causalMask && !testParams.mask.empty()) {
            if (opts.booleanMask) {
                log.warning("BOOLEAN_MASK with CAUSAL_MASK constant is redundant. "
                            "OpenVINO constant-folds Select(bool_const, 0.0, -inf) into a float constant. "
                            "Generating float causal mask directly.");
            }

            // Use N and C (heads) from mask shape, but tSL and sSL from Q and K sequence dimensions
            const auto& maskShape = testParams.mask;

            // Extract dimensions
            const size_t N = maskShape.size() == 4 ? maskShape[0] : 1;
            const size_t C = maskShape[headDim];
            const size_t tSL = inputQShape[rank - 2];
            const size_t sSL = inputKShape[rank - 2];
            const size_t numPlanes = N * C;
            const size_t planeSize = tSL * sSL;
            const size_t totalElements = numPlanes * planeSize;

            // Create mask with full shape [N, C, tSL, sSL]
            const ov::Shape causalMaskShape = (rank == 4) ? ov::Shape{N, C, tSL, sSL} : ov::Shape{C, tSL, sSL};

            // Float mask: 0.0 = keep, -inf = mask
            std::vector<float> maskData(totalElements, 0.0f);

            const float minusInf = -std::numeric_limits<float>::infinity();
            for (size_t plane = 0; plane < numPlanes; ++plane) {
                const size_t planeOffset = plane * planeSize;
                for (size_t i = 0; i < tSL; ++i) {
                    for (size_t j = i + 1; j < sSL; ++j) {
                        maskData[planeOffset + i * sSL + j] = minusInf;
                    }
                }
            }

            attentionMask = ov::op::v0::Constant::create(inType, causalMaskShape, maskData);
        } else if (!testParams.mask.empty() && !opts.causalMask) {
            std::shared_ptr<ov::op::v0::Parameter> inputMask;
            if (opts.booleanMask) {
                inputMask = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, testParams.mask);
            } else {
                inputMask = std::make_shared<ov::op::v0::Parameter>(inType, testParams.mask);
            }
            params.push_back(inputMask);
            inputShapes.push_back(testParams.mask);
            attentionMask = inputMask;
        }

        if (testParams.attentionType == AttentionType::SDPA && !attentionMask && !testParams.scale.empty()) {
            const auto emptyMask = std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{});
            params.push_back(emptyMask);
            inputShapes.push_back(ov::Shape{});
            attentionMask = emptyMask;
        }

        // Handle Scale
        std::shared_ptr<ov::Node> scale = nullptr;
        if (!testParams.scale.empty()) {
            ov::Shape scaleShape = testParams.scale;
            if (testParams.attentionType == AttentionType::SDPA && scaleShape != ov::Shape{1}) {
                log.warning("SDPA operation requires scale to be 1D with shape [1]. "
                            "Current scale shape: {0}. Using shape [1] instead.",
                            ov::test::utils::vec2str(scaleShape));
                scaleShape = ov::Shape{1};
            }

            if (opts.scaleConst) {
                const float scaleFactor = 1.0f / std::sqrt(static_cast<float>(inputQShape[rank - 1]));
                scale = ov::op::v0::Constant::create(inType, scaleShape, {scaleFactor});
            } else {
                const auto inputScale = std::make_shared<ov::op::v0::Parameter>(inType, scaleShape);
                params.push_back(inputScale);
                inputShapes.push_back(scaleShape);
                scale = inputScale;
            }
        }

        // Handle Bias
        std::shared_ptr<ov::Node> bias = nullptr;
        if (useBias) {
            const auto inputBias = std::make_shared<ov::op::v0::Parameter>(inType, testParams.bias);
            params.push_back(inputBias);
            inputShapes.push_back(testParams.bias);
            bias = inputBias;
        }

        init_input_shapes(ov::test::static_shapes_to_test_representation(inputShapes));

        if (testParams.attentionType == AttentionType::SDPA) {
            ov::OutputVector inputs;
            inputs.emplace_back(inputQ);
            inputs.emplace_back(inputK);
            inputs.emplace_back(inputV);

            if (attentionMask) {
                inputs.emplace_back(attentionMask);
            }

            if (scale) {
                inputs.emplace_back(scale);
            }

            const auto sdp = std::make_shared<ov::opset14::ScaledDotProductAttention>(inputs, isCausal);
            sdp->set_friendly_name("sdp");

            const auto result = std::make_shared<ov::op::v0::Result>(sdp);
            function = std::make_shared<ov::Model>(ov::ResultVector{result}, params, "AttentionTest");
        } else {
            // For GQA/MQA: Broadcast K and V to match Q's number of heads
            std::shared_ptr<ov::Node> finalK = broadcastHeadsDimension(inputK, inputKShape, qHeads);
            std::shared_ptr<ov::Node> finalV = broadcastHeadsDimension(inputV, inputVShape, qHeads);

            std::shared_ptr<ov::Node> scaledQ = inputQ;
            std::shared_ptr<ov::Node> scaledK = finalK;

            if (scale && opts.scaleQ) {
                // SCALE_Q: (scale * Q) @ K.T
                scaledQ = std::make_shared<ov::op::v1::Multiply>(inputQ, scale);
            } else if (scale && opts.scaleK) {
                // SCALE_K: Q @ (scale * K).T
                scaledK = std::make_shared<ov::op::v1::Multiply>(finalK, scale);
            }

            // MatMul(Q, K)
            auto matmulQK = std::make_shared<ov::op::v0::MatMul>(scaledQ, scaledK, false, true);
            std::shared_ptr<ov::Node> attentionScores = matmulQK;

            // Apply scale after MatMul (default position when SCALE_Q/SCALE_K not used)
            if (scale && !opts.scaleQ && !opts.scaleK) {
                attentionScores = std::make_shared<ov::op::v1::Multiply>(attentionScores, scale);
            }

            // Apply attention mask
            if (attentionMask) {
                if (opts.booleanMask && !opts.causalMask) {
                    // If mask is true (keep), use 0.0; if false (mask), use -inf
                    const auto keepValue = ov::op::v0::Constant::create(inType, ov::Shape{}, {0.0f});
                    const auto maskValueF32 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{},
                                                                           {-std::numeric_limits<float>::infinity()});
                    const auto maskValue = std::make_shared<ov::op::v1::ConvertLike>(maskValueF32, attentionScores);
                    const auto selectedMask = std::make_shared<ov::op::v1::Select>(attentionMask, keepValue, maskValue,
                                                                                   ov::op::AutoBroadcastType::NUMPY);
                    attentionScores = std::make_shared<ov::op::v1::Add>(attentionScores, selectedMask);
                } else {
                    attentionScores = std::make_shared<ov::op::v1::Add>(attentionScores, attentionMask);
                }
            }

            // Apply bias (add)
            if (bias) {
                attentionScores = std::make_shared<ov::op::v1::Add>(attentionScores, bias);
            }

            // Softmax over last dimension
            const int64_t softmaxAxis = static_cast<int64_t>(inputQShape.size()) - 1;
            auto softmax = std::make_shared<ov::op::v1::Softmax>(attentionScores, softmaxAxis);

            // MatMul(Softmax, V)
            auto output = std::make_shared<ov::op::v0::MatMul>(softmax, finalV, false, false);

            const auto result = std::make_shared<ov::op::v0::Result>(output);
            function = std::make_shared<ov::Model>(ov::ResultVector{result}, params, "AttentionTest");
        }
    }

private:
    vpux::Logger log;

    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "decompose-attention=false";
    }
};

class AttentionTestDecompose : public AttentionTestCommon {
    void configure_model() override {
        VpuOv2LayerTest::configure_model();
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "decompose-attention=true convert-to-attention=true";
    }
};

// Attention on NPU4 is always decomposed. NPU4 runs only the dedicated smoke_DecomposeNPU4 suite,
// so this subclass exclusively registers the NPU4000 test.
class AttentionTestDecomposeNPU4 : public AttentionTestDecompose {};

TEST_P(AttentionTestCommon, NPU5010_HW) {
    abs_threshold = 0.012;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(AttentionTestDecompose, NPU5010_HW) {
    abs_threshold = 0.012;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(AttentionTestDecomposeNPU4, NPU4000_HW) {
    abs_threshold = 0.012;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(
        precommit, AttentionTestCommon,
        ::testing::Combine(::testing::ValuesIn(std::vector<AttentionShapeConfig>{
                                   // SelfAttention Tests
                                   {{1, 1, 1, 8}, {1, 1, 16, 8}, {1, 1, 16, 8}, {1, 1, 1, 16}, {1}, {}, {}},
                                   {{1, 1, 12, 8}, {1, 1, 16, 8}, {1, 1, 16, 8}, {1, 1, 12, 16}, {1}, {}, {}},
                                   {{1, 32, 12, 8}, {1, 32, 16, 8}, {1, 32, 16, 8}, {1, 32, 12, 16}, {1}, {}, {}},

                                   // CrossAttention Tests
                                   {{1, 1, 1, 8}, {1, 1, 16, 8}, {1, 1, 16, 4}, {1, 1, 1, 16}, {1}, {}, {}},
                                   {{1, 1, 12, 8}, {1, 1, 16, 8}, {1, 1, 16, 4}, {1, 1, 12, 16}, {1}, {}, {}},
                                   {{1, 8, 12, 8}, {1, 8, 16, 8}, {1, 8, 16, 4}, {1, 8, 12, 16}, {1}, {}, {}},
                           }),
                           ::testing::Values(AttentionType::SDPA, AttentionType::PATTERN)),
        AttentionTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_fullSanityCheck, AttentionTestCommon,
        ::testing::Combine(
                ::testing::ValuesIn(std::vector<AttentionShapeConfig>{
                        // Simple Self and Cross attention
                        {{1, 2, 64, 32}, {1, 2, 64, 32}, {1, 2, 64, 32}, {}, {}, {}, {}},
                        {{1, 2, 64, 32}, {1, 2, 128, 32}, {1, 2, 128, 32}, {}, {}, {}, {}},

                        // Square attention
                        {{1, 8, 1, 64}, {1, 8, 64, 64}, {1, 8, 64, 64}, {}, {}, {}, {}},
                        {{1, 8, 64, 64}, {1, 8, 64, 64}, {1, 8, 64, 64}, {}, {}, {}, {}},

                        // Optional Mask
                        {{1, 2, 64, 32}, {1, 2, 16, 32}, {1, 2, 16, 32}, {1, 2, 64, 16}, {1}, {}, {}},
                        {{1, 2, 64, 32}, {1, 2, 16, 32}, {1, 2, 16, 32}, {1, 1, 64, 16}, {1}, {}, {}},
                        {{1, 2, 64, 32}, {1, 2, 16, 32}, {1, 2, 16, 32}, {1, 1, 1, 16}, {1}, {}, {}},
                        {{1, 2, 64, 32}, {1, 2, 16, 32}, {1, 2, 16, 32}, {1, 1, 64, 1}, {1}, {}, {}},
                        {{1, 2, 64, 32}, {1, 2, 16, 32}, {1, 2, 16, 32}, {1, 1, 1, 1}, {1}, {}, {}},

                        // Optional Scale
                        {{1, 2, 64, 32}, {1, 2, 16, 32}, {1, 2, 16, 32}, {1, 1, 1, 1}, {}, {}, {}},
                        {{1, 2, 64, 32}, {1, 2, 16, 32}, {1, 2, 16, 32}, {}, {1}, {}, {}},
                        {{1, 2, 64, 32}, {1, 2, 16, 32}, {1, 2, 16, 32}, {1, 1, 1, 1}, {1, 2, 1, 1}, {}, {}},

                        // Optional Bias
                        {{1, 4, 16, 32}, {1, 4, 16, 32}, {1, 4, 16, 32}, {1, 4, 16, 16}, {1}, {1, 4, 16, 16}, {}},
                        {{1, 4, 128, 32}, {1, 4, 256, 32}, {1, 4, 256, 32}, {1, 1, 128, 256}, {1}, {1, 1, 1, 256}, {}},

                        // MultiQueryAttention configurations
                        {{1, 8, 64, 32}, {1, 1, 16, 32}, {1, 1, 16, 32}, {1, 1, 64, 16}, {1}, {}, {}},

                        // Options
                        {{1, 2, 64, 32},
                         {1, 2, 16, 32},
                         {1, 2, 16, 32},
                         {1, 2, 64, 16},
                         {1},
                         {},
                         {AttentionOption::CAUSAL_MASK}},
                        {{1, 2, 64, 32},
                         {1, 2, 16, 32},
                         {1, 2, 16, 32},
                         {1, 2, 64, 16},
                         {1},
                         {},
                         {AttentionOption::CAUSAL_FLAG}},
                        {{1, 2, 64, 32},
                         {1, 2, 16, 32},
                         {1, 2, 16, 32},
                         {1, 2, 64, 16},
                         {1},
                         {},
                         {AttentionOption::SCALE_CONST}},
                        {{1, 2, 64, 32},
                         {1, 2, 16, 32},
                         {1, 2, 16, 32},
                         {1, 2, 64, 16},
                         {1},
                         {},
                         {AttentionOption::SCALE_Q}},
                        {{1, 2, 64, 32},
                         {1, 2, 16, 32},
                         {1, 2, 16, 32},
                         {1, 2, 64, 16},
                         {1},
                         {},
                         {AttentionOption::SCALE_K}},
                }),
                ::testing::Values(AttentionType::SDPA, AttentionType::PATTERN)),
        AttentionTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Decompose, AttentionTestDecompose,
        ::testing::Combine(
                ::testing::ValuesIn(std::vector<AttentionShapeConfig>{
                        // MHA Decomposition configurations
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {}, {}, {}, {}},
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {1, 4, 16, 24}, {}, {}, {}},
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {1, 1, 16, 24}, {}, {}, {}},
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {}, {1}, {}, {}},
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {1, 4, 16, 24}, {1}, {}, {}},
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {1, 4, 16, 24}, {1}, {1, 4, 16, 24}, {}},
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {1, 1, 16, 24}, {1}, {1, 1, 1, 24}, {}},
                        {{1, 4, 16, 32},
                         {1, 4, 24, 32},
                         {1, 4, 24, 64},
                         {1, 4, 16, 24},
                         {1},
                         {},
                         {AttentionOption::BOOLEAN_MASK}},
                }),
                ::testing::Values(AttentionType::SDPA, AttentionType::PATTERN)),
        AttentionTestDecompose::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_DecomposeNPU4, AttentionTestDecomposeNPU4,
        ::testing::Combine(
                ::testing::ValuesIn(std::vector<AttentionShapeConfig>{
                        // MHA Decomposition configurations
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {}, {}, {}, {}},
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {1, 4, 16, 24}, {}, {}, {}},
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {1, 1, 16, 24}, {}, {}, {}},
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {}, {1}, {}, {}},
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {1, 4, 16, 24}, {1}, {}, {}},
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {1, 4, 16, 24}, {1}, {1, 4, 16, 24}, {}},
                        {{1, 4, 16, 32}, {1, 4, 24, 32}, {1, 4, 24, 64}, {1, 1, 16, 24}, {1}, {1, 1, 1, 24}, {}},
                        {{1, 4, 16, 32},
                         {1, 4, 24, 32},
                         {1, 4, 24, 64},
                         {1, 4, 16, 24},
                         {1},
                         {},
                         {AttentionOption::BOOLEAN_MASK}},
                }),
                ::testing::Values(AttentionType::SDPA, AttentionType::PATTERN)),
        AttentionTestDecomposeNPU4::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_LegalConfigs, AttentionTestDecompose,
        ::testing::Combine(
                ::testing::ValuesIn(std::vector<AttentionShapeConfig>{
                        // Legal MHA configurations (qH == kH)
                        {{16, 12, 225, 16}, {16, 12, 225, 16}, {16, 12, 225, 16}, {}, {}, {}, {}},
                        {{1, 192, 225, 64}, {1, 192, 225, 64}, {1, 192, 225, 64}, {}, {}, {}, {}},
                        {{1, 12, 3600, 64}, {1, 12, 3600, 64}, {1, 12, 3600, 64}, {}, {}, {}, {}},
                        {{1, 8, 300, 64}, {1, 8, 300, 64}, {1, 8, 300, 64}, {}, {}, {}, {}},
                        {{1, 16, 577, 64}, {1, 16, 577, 64}, {1, 16, 577, 64}, {}, {}, {}, {}},
                        {{1, 10, 1024, 64}, {1, 10, 1024, 64}, {1, 10, 1024, 64}, {}, {}, {}, {}},
                        {{1, 10, 1024, 64}, {1, 10, 77, 64}, {1, 10, 77, 64}, {}, {}, {}, {}},
                        {{1, 20, 256, 64}, {1, 20, 256, 64}, {1, 20, 256, 64}, {}, {}, {}, {}},
                        {{1, 20, 256, 64}, {1, 20, 77, 64}, {1, 20, 77, 64}, {}, {}, {}, {}},
                        {{1, 6, 3072, 64}, {1, 6, 3072, 64}, {1, 6, 3072, 64}, {1, 1, 1, 3072}, {1}, {}, {}},
                        {{1, 1, 32, 64}, {1, 1, 96, 64}, {1, 1, 96, 64}, {1, 1, 1, 96}, {1}, {}, {}},
                        {{1, 6, 151, 64}, {1, 6, 151, 64}, {1, 6, 151, 64}, {}, {}, {}, {}},
                        {{1, 16, 256, 72}, {1, 16, 256, 72}, {1, 16, 256, 72}, {}, {1}, {}, {}},

                        // Legal MQA configuration (qH > kH && kH == 1)
                        {{1, 8, 570, 256}, {1, 1, 570, 256}, {1, 1, 570, 256}, {1, 1, 570, 570}, {1}, {}, {}},
                }),
                ::testing::Values(AttentionType::PATTERN)),
        AttentionTestDecompose::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FullLineMask, AttentionTestCommon,
                         ::testing::Combine(::testing::ValuesIn(std::vector<AttentionShapeConfig>{
                                                    // sSL <= 32: short inner softmax path
                                                    {{1, 2, 64, 32},
                                                     {1, 2, 16, 32},
                                                     {1, 2, 16, 32},
                                                     {1, 2, 64, 16},
                                                     {1},
                                                     {},
                                                     {AttentionOption::FULL_LINE_MASK}},
                                                    {{1, 2, 64, 32},
                                                     {1, 2, 16, 32},
                                                     {1, 2, 16, 32},
                                                     {1, 1, 64, 16},
                                                     {1},
                                                     {},
                                                     {AttentionOption::FULL_LINE_MASK}},
                                                    {{1, 1, 8, 16},
                                                     {1, 1, 32, 16},
                                                     {1, 1, 32, 16},
                                                     {1, 1, 8, 32},
                                                     {1},
                                                     {},
                                                     {AttentionOption::FULL_LINE_MASK}},
                                                    // sSL > 32: wider inner softmax path
                                                    {{1, 2, 16, 32},
                                                     {1, 2, 48, 32},
                                                     {1, 2, 48, 32},
                                                     {1, 2, 16, 48},
                                                     {1},
                                                     {},
                                                     {AttentionOption::FULL_LINE_MASK}},
                                                    {{1, 2, 16, 32},
                                                     {1, 2, 64, 32},
                                                     {1, 2, 64, 32},
                                                     {1, 1, 16, 64},
                                                     {1},
                                                     {},
                                                     {AttentionOption::FULL_LINE_MASK}},
                                                    {{1, 1, 1, 32},
                                                     {1, 1, 48, 32},
                                                     {1, 1, 48, 32},
                                                     {1, 1, 1, 48},
                                                     {1},
                                                     {},
                                                     {AttentionOption::FULL_LINE_MASK}},
                                                    {{1, 12, 512, 64},
                                                     {1, 12, 512, 64},
                                                     {1, 12, 512, 64},
                                                     {1, 1, 1, 512},
                                                     {1},
                                                     {},
                                                     {AttentionOption::FULL_LINE_MASK}},
                                            }),
                                            ::testing::Values(AttentionType::SDPA)),
                         AttentionTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_GQA, AttentionTestCommon,
        ::testing::Combine(
                ::testing::ValuesIn(std::vector<AttentionShapeConfig>{
                        {{1, 64, 1024, 128}, {1, 8, 1024, 128}, {1, 8, 1024, 128}, {1, 1, 1024, 1024}, {1}, {}, {}},
                        {{1, 32, 1024, 128}, {1, 8, 1024, 128}, {1, 8, 1024, 128}, {1, 1, 1024, 1024}, {1}, {}, {}},
                        {{1, 28, 1024, 128}, {1, 4, 1024, 128}, {1, 4, 1024, 128}, {1, 1, 1024, 1024}, {1}, {}, {}},
                        {{1, 32, 1, 128}, {1, 2, 1152, 128}, {1, 2, 1152, 128}, {1, 1, 1, 1152}, {1}, {}, {}},
                        {{1, 32, 1, 128}, {1, 2, 1152, 128}, {1, 2, 1152, 128}, {1, 32, 1, 1152}, {1}, {}, {}},
                }),
                ::testing::Values(AttentionType::PATTERN)),
        AttentionTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_AttentionDMA, AttentionTestDecompose,
        ::testing::Combine(
                ::testing::ValuesIn(std::vector<AttentionShapeConfig>{AttentionShapeConfig{
                        {1, 12, 3600, 16}, {1, 12, 3600, 16}, {1, 12, 3600, 16}, {1, 1, 1, 3600}, {1}, {}, {}}}),
                ::testing::Values(AttentionType::SDPA)),
        AttentionTestDecompose::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_AttentionDMA, AttentionTestDecomposeNPU4,
        ::testing::Combine(
                ::testing::ValuesIn(std::vector<AttentionShapeConfig>{AttentionShapeConfig{
                        {1, 12, 3600, 16}, {1, 12, 3600, 16}, {1, 12, 3600, 16}, {1, 1, 1, 3600}, {1}, {}, {}}}),
                ::testing::Values(AttentionType::SDPA)),
        AttentionTestDecomposeNPU4::getTestCaseName);

}  // namespace ov::test

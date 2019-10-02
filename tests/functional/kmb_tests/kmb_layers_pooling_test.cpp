//
// Copyright 2019 Intel Corporation.
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
#include <vpu/kmb_plugin_config.hpp>

#include "kmb_layers_tests.hpp"

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;

typedef std::tuple<tensor_test_params, std::string, std::string, std::string, param_size, param_size, param_size, param_size> pooling_test_params;
typedef kmbLayerTestBaseWithParam< pooling_test_params > kmbLayersTestsPoolingParams;

#ifdef ENABLE_MCM_COMPILER
TEST_F(kmbLayersTests_nightly, DISABLED_TestsPoolingAfterConvolution) {
    // TODO: mcmCompiler compilation fails (Convolution with bias): Segmentation fault. Jira: VPUNND-1474
    const std::string model = R"V0G0N(
    <net batch="1" name="POOLING_TEST" version="2">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="scale_shift" precision="FP16" type="ScaleShift">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </input>
                <output>
                    <port id="3">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="0" size="6"/>
                    <biases offset="6" size="6"/>
                </blobs>
            </layer>
            <layer id="2" name="conv_test" precision="FP16" type="Convolution">
                <data dilations="1,1" group="1" kernel="7,7" output="64" pads_begin="3,3" pads_end="3,3" strides="2,2"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </input>
                <output>
                    <port id="3">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>112</dim>
                        <dim>112</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="12" size="18816"/>
                    <biases offset="18828" size="128"/>
                </blobs>
            </layer>
            <layer id="3" name="pooling_test" precision="FP16" type="Pooling">
                <data auto_pad="same_upper" exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="1,1" pool-method="max" strides="2,2"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>112</dim>
                        <dim>112</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
            <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        </edges>
    </net>
        )V0G0N";

    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t >(18828 + 128));

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["pooling_test"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);

    // Parsing only is enabled because mcmCompiler can't compile layers.
    // TODO: turn off parsing only when mcmCompiler will be able to compile this layers.
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(_exeNetwork = ie.LoadNetwork(network, "kmb", config));
}

TEST_F(kmbLayersTests_nightly, DISABLED_TestsPoolingOnly) {
    const std::string model = R"V0G0N(
    <net batch="1" name="POOLING_TEST" version="2">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="pooling_test" precision="FP16" type="Pooling">
                <data auto_pad="same_upper" exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="1,1" pool-method="max" strides="2,2"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </output>
            </layer>
            </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        </edges>
    </net>
        )V0G0N";

    StatusCode st;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["pooling_test"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(_exeNetwork = ie.LoadNetwork(network, "kmb", config));
}

TEST_P(kmbLayersTestsPoolingParams, DISABLED_TestsPoolingNetInit) {
    auto param = GetParam();
    tensor_test_params tensor = std::get<0>(param);
    std::string sameUpper = std::get<1>(param);
    std::string excludePad = std::get<2>(param);
    std::string poolMethod = std::get<3>(param);
    param_size kernel = std::get<4>(param);
    param_size padsBegin = std::get<5>(param);
    param_size padsEnd = std::get<6>(param);
    param_size strides = std::get<7>(param);

    const ::testing::TestInfo* const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();

    std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name() << " test_info->name()=" <<
            test_info->name() << " test_info->test_case_name() " << test_info->test_case_name() << std::endl;

    std::map<std::string, std::string> params;

    params["same_upper"] = sameUpper;
    params["exclude-pad"] = excludePad;
    params["pool-method"] = poolMethod;
    params["kernel"] = std::to_string(kernel.x) + "," + std::to_string(kernel.y);
    params["pads_begin"] = std::to_string(padsBegin.x) + "," + std::to_string(padsBegin.y);
    params["pads_end"] = std::to_string(padsEnd.x) + "," + std::to_string(padsEnd.y);
    params["strides"] = std::to_string(strides.x) + "," + std::to_string(strides.y);

    SetInputTensor(tensor);
    SetOutputTensor(tensor);
    NetworkInit("Pooling",
                &params,
                0,
                0,
                nullptr,
                Precision::FP16 // output precision
    );
}

static const pooling_test_params paramsTable[] = {
    std::make_tuple<tensor_test_params, std::string, std::string, std::string, param_size, param_size, param_size, param_size>(
        {1, 3, 224, 224},  // input and output tensors
        "same_upper",       // auto_pad
        "true",             // exclude-pad
        "max",              // pool-method
        {3, 3},             // kernel
        {0, 0},             // pads_begin
        {1, 1},             // pads_end
        {2, 2}              // strides
    ),
};

INSTANTIATE_TEST_CASE_P(loadNetworkNoThrow, kmbLayersTestsPoolingParams,
    ::testing::ValuesIn(paramsTable)
);

struct PoolingTestParams {
    SizeVector input_size;
    pool_common_params pool_params;
};

class PoolingTest : public KmbPerLayerTest, public testing::WithParamInterface<PoolingTestParams> {};

const std::string uint8_pooling = R"V0G0N(
<net batch="1" name="POOLING_TEST" version="2">
    <layers>
        <layer id="0" name="input" precision="U8" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>_INPUT_HEIGHT_</dim>
                    <dim>_INPUT_WIDTH_</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="pooling_test" precision="U8" type="Pooling">
            <data kernel="_KERNEL_" strides="_STRIDE_" exclude-pad="_EXCLUDE_PAD_" pool-method="max"  auto_pad="same_upper" pads_begin="0,0" pads_end="1,1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>_INPUT_HEIGHT_</dim>
                    <dim>_INPUT_WIDTH_</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>_OUTPUT_HEIGHT_</dim>
                    <dim>_OUTPUT_WIDTH_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

// Disabled due to bug in mcmCompiler with Release build type.
// Corresponding Jira ticket VPUNND-1910
TEST_P(PoolingTest, DISABLED_pooling_only) {
    auto model = uint8_pooling;

    auto params = GetParam().pool_params;
    REPLACE_WITH_NUM_VECTOR(model, "_KERNEL_", params.kernel);
    REPLACE_WITH_NUM_VECTOR(model, "_STRIDE_", params.stride);
    REPLACE_WITH_STR(model, "_EXCLUDE_PAD_", params.exclude_pad ? "true" : "false");

    auto inputSize = GetParam().input_size;
    SizeVector outputSize;
    getPoolOutShape(inputSize, params, outputSize);

    REPLACE_WITH_NUM(model, "_INPUT_HEIGHT_", inputSize[2]);
    REPLACE_WITH_NUM(model, "_INPUT_WIDTH_", inputSize[3]);

    REPLACE_WITH_NUM(model, "_OUTPUT_HEIGHT_", outputSize[2]);
    REPLACE_WITH_NUM(model, "_OUTPUT_WIDTH_", outputSize[3]);


    CNNNetReader reader;
    reader.ReadNetwork(model.data(), model.length());
    ASSERT_TRUE(reader.isParseSuccess());

    Core ie;
    ExecutableNetwork exeNetwork;

    auto config = getCommonConfig();
    ASSERT_NO_THROW(exeNetwork = ie.LoadNetwork(reader.getNetwork(), "KMB", config));
    ASSERT_NO_THROW(exeNetwork.Export(getTestResultFilename() + ".blob"));
}

// Assuming input layout have NCHW order
std::vector<PoolingTestParams> int_pooling_params = {
        {{1, 3, 224, 224}, {{2, 2}, {2, 2}, {0, 0}, {1, 1}, "same_upper", false, "true"}},
        {{1, 3, 224, 224}, {{2, 2}, {4, 4}, {0, 0}, {1, 1}, "same_upper", false, "true"}},
        {{1, 3, 224, 224}, {{2, 2}, {2, 2}, {0, 0}, {1, 1}, "same_upper", false, "false"}},
};

INSTANTIATE_TEST_CASE_P(PerLayer, PoolingTest, ::testing::ValuesIn(int_pooling_params));

#endif

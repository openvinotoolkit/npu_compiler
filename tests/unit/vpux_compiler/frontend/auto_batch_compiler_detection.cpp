//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/op/convert.hpp>
#include <openvino/op/relu.hpp>
#include <string>
#include <vector>

#include "common/utils.hpp"
#include "frontend/utils.hpp"
#include "vpux/compiler/frontend/ov_batch_detection.hpp"
#include "vpux/compiler/utils/batch.hpp"
#include "vpux/utils/ov/config.hpp"
#include "vpux/utils/ov/options.hpp"

namespace {

using IODescription = std::tuple<ov::PartialShape, std::optional<ov::Layout>>;
using IODescriptions = std::vector<IODescription>;
inline std::shared_ptr<ov::Model> create_multi_input_model(const IODescriptions& decriptions) {
    static const ov::element::Type type = ov::element::f32;

    size_t index = 0;
    ov::ParameterVector dataArray;
    ov::ResultVector resultArray;
    for (const auto& [shape, layout] : decriptions) {
        auto data = std::make_shared<ov::op::v0::Parameter>(type, shape);
        std::string inputName{"input"};
        std::string indexStr = std::to_string(index);
        inputName += indexStr;

        data->set_friendly_name(inputName);
        data->output(0).get_tensor().set_names({inputName});
        if (layout.has_value()) {
            data->set_layout(layout.value());
        }
        auto op = std::make_shared<ov::op::v0::Relu>(data);
        auto convertOp = std::make_shared<ov::op::v0::Convert>(op, ov::element::boolean);
        std::shared_ptr<ov::op::v0::Result> res = std::make_shared<ov::op::v0::Result>(convertOp);
        std::string resultName{"Result"};
        resultName += indexStr;
        res->set_friendly_name(resultName);
        res->output(0).get_tensor().set_names({resultName});
        if (layout.has_value()) {
            res->set_layout(layout.value());
        }
        dataArray.push_back(std::move(data));
        resultArray.push_back(std::move(res));
        index++;
    }
    return std::make_shared<ov::Model>(std::move(resultArray), std::move(dataArray));
}

template <class... TestParams>
using CatenatedTestParams = decltype(std::tuple_cat(std::declval<TestParams>()...));

using CfgMap = std::map<std::string, std::string>;
using AutoBatchCompilerDetectionTestsBaseParams = std::tuple<IODescriptions, CfgMap>;

template <class... ExtraParams>
struct AutoBatchCompilerDetectionTestsBase :
        public testing::TestWithParam<
                CatenatedTestParams<AutoBatchCompilerDetectionTestsBaseParams, std::tuple<ExtraParams>...>> {
    virtual void SetUp() override {
        // Not all C++17 compilers are able to digest structural binding with a variable params amount like this
        // auto [a, b, ...c]
        // Therefore we use the conventional way extracting them using std::get
        static constexpr size_t netInfoParamIndex = 0;
        const IODescriptions& netIoDescriptions = std::get<netInfoParamIndex>(this->GetParam());
        static constexpr size_t configurationParamIndex = netInfoParamIndex + 1;
        CfgMap configuration = std::get<configurationParamIndex>(this->GetParam());
        static constexpr size_t nextParamParamIndex = configurationParamIndex + 1;
        static_assert(nextParamParamIndex == getBasicParamSize(),
                      "All params must be obtained from AutoBatchCompilerDetectionTestsBaseParams");

        ov_stub_model = create_multi_input_model(netIoDescriptions);
        if (!optionDescPtr) {
            optionDescPtr = std::make_shared<vpux::OV::OptionsDesc>();
        }
        optionDescPtr->add<vpux::OV::BATCH_COMPILER_MODE_SETTINGS>();
        optionDescPtr->add<vpux::OV::BATCH_MODE>();
        optionDescPtr->add<vpux::OV::LOG_LEVEL>();
        configurationPtr.reset(new vpux::OV::Config(optionDescPtr));
        configurationPtr->update(configuration);
    }

    static constexpr size_t getBasicParamSize() {
        return std::tuple_size_v<AutoBatchCompilerDetectionTestsBaseParams>;
    }
    static std::string getTestCaseName(
            const testing::TestParamInfo<
                    CatenatedTestParams<AutoBatchCompilerDetectionTestsBaseParams, std::tuple<ExtraParams>...>>& obj) {
        // Not all C++17 compilers are able to digest structural binding with a variable params amount like this
        // auto [a, b, ...c]
        // Therefore we use the conventional way extracting them using std::get
        static constexpr size_t netInfoParamIndex = 0;
        const IODescriptions& netInfo = std::get<netInfoParamIndex>(obj.param);
        static constexpr size_t configurationParamIndex = netInfoParamIndex + 1;
        CfgMap configuration = std::get<configurationParamIndex>(obj.param);
        static constexpr size_t nextParamParamIndex = configurationParamIndex + 1;
        static_assert(nextParamParamIndex == getBasicParamSize(),
                      "All params must be obtained from AutoBatchCompilerDetectionTestsBaseParams");

        std::ostringstream result;
        result << "netIOCount=" << netInfo.size();
        for (const auto& info : netInfo) {
            result << std::get<0>(info).to_string();
            if (const auto& layoutOpt = std::get<1>(info); layoutOpt.has_value()) {
                result << std::get<1>(info)->to_string();
            }
        }
        if (!configuration.empty()) {
            for (const auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second;
            }
        }
        return result.str();
    }

protected:
    std::shared_ptr<vpux::OV::OptionsDesc> optionDescPtr;
    std::unique_ptr<vpux::OV::Config> configurationPtr;
    std::shared_ptr<ov::Model> ov_stub_model;
    vpux::Logger logger = vpux::Logger::global();
};

using AutoBatchCompilerDetectionTestParams =
        CatenatedTestParams<AutoBatchCompilerDetectionTestsBaseParams, std::tuple<bool>>;
class AutoBatchCompilerDetectionTests : public AutoBatchCompilerDetectionTestsBase<bool> {
public:
    using Base = AutoBatchCompilerDetectionTestsBase<bool>;
    static constexpr size_t getFirstParamTupleIndex() {
        return Base::getBasicParamSize();
    }
    void SetUp() override {
        Base::SetUp();
        result = std::get<getFirstParamTupleIndex()>(this->GetParam());
    }

    static std::string getTestCaseName(const testing::TestParamInfo<AutoBatchCompilerDetectionTestParams>& obj) {
        std::ostringstream result;
        result << Base::getTestCaseName(obj)
               << "_batchDetectionExpected=" << std::get<getFirstParamTupleIndex()>(obj.param);
        return result.str();
    }

protected:
    bool result;
};

class AutoBatchCompilerForbiddenByLayoutTests : public AutoBatchCompilerDetectionTests {};

TEST_P(AutoBatchCompilerForbiddenByLayoutTests, doNotMakeBatchDetection) {
    const auto batchForbidden = isBatchForbiddenExplicitly(ov_stub_model, logger);
    EXPECT_EQ(batchForbidden, result);
}

CfgMap configs;
INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTest, AutoBatchCompilerForbiddenByLayoutTests,
        testing::Values(AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), {}),
                                                IODescription(ov::PartialShape({100, 200}), {}),
                                                IODescription(ov::PartialShape({300, 400}), {})}),
                                configs, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{})}),
                                configs, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"DCHW"}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{"..."}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{})}),
                                configs, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"NCHW"}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{"HW"}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{"HW"})}),
                                configs, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"NCHW"}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{"..."}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{"NC"})}),
                                configs, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"NCHW"}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{"HW"}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{"NH"})}),
                                configs, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({1, 3, 100, 200}), ov::Layout{"DCHW"}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{"HW"}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{"WH"})}),
                                configs, true)),
        AutoBatchCompilerForbiddenByLayoutTests::getTestCaseName);

class AutoBatchCompilerDetectionBasedOnLayoutTests : public AutoBatchCompilerDetectionTests {};

TEST_P(AutoBatchCompilerDetectionBasedOnLayoutTests, makeBatchDetection) {
    const auto& [batchDetected, strInfo] = isBatchDetectedByUserLayouts(ov_stub_model, logger);
    EXPECT_EQ(batchDetected, result);
}

INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTest, AutoBatchCompilerDetectionBasedOnLayoutTests,
        testing::Values(AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), {}),
                                                IODescription(ov::PartialShape({100, 200}), {}),
                                                IODescription(ov::PartialShape({300, 400}), {})}),
                                configs, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{})}),
                                configs, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"NCHW"}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{"HW"}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{"HW"})}),
                                configs, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"NCHW"}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{"..."}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{"NC"})}),
                                configs, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({1, 3, 100, 200}), ov::Layout{"NCHW"}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{"NC"}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{"NC"})}),
                                configs, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"NCHW"}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{"NC"}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{"NC"})}),
                                configs, true),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({1, 3, 100, 200}), ov::Layout{"NCHW"}),
                                                IODescription(ov::PartialShape({}), ov::Layout{"..."}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{"NC"})}),
                                configs, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"NCHW"}),
                                                IODescription(ov::PartialShape({}), ov::Layout{"..."}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{"NC"})}),
                                configs, true)),
        AutoBatchCompilerDetectionBasedOnLayoutTests::getTestCaseName);

class AutoBatchCompilerDetectionBasedOnOVTests : public AutoBatchCompilerDetectionTests {};
TEST_P(AutoBatchCompilerDetectionBasedOnOVTests, makeBatchDetection) {
    const auto& [batchDetected, strInfo] = isBatchDetectedByOVHeuristic(ov_stub_model, logger);
    EXPECT_EQ(batchDetected, result);
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, AutoBatchCompilerDetectionBasedOnOVTests,
                         testing::Values(AutoBatchCompilerDetectionTestParams(
                                                 IODescriptions({IODescription(ov::PartialShape({1, 3, 100, 200}), {}),
                                                                 IODescription(ov::PartialShape({1, 200}), {}),
                                                                 IODescription(ov::PartialShape({1, 400}), {})}),
                                                 configs, false),
                                         AutoBatchCompilerDetectionTestParams(
                                                 IODescriptions({IODescription(ov::PartialShape({1, 3, 100, 200}), {}),
                                                                 IODescription(ov::PartialShape({}), {}),
                                                                 IODescription(ov::PartialShape({300, 400}), {})}),
                                                 configs, false),
                                         AutoBatchCompilerDetectionTestParams(
                                                 IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), {}),
                                                                 IODescription(ov::PartialShape({100, 200}), {}),
                                                                 IODescription(ov::PartialShape({300, 400}), {})}),
                                                 configs, true),
                                         AutoBatchCompilerDetectionTestParams(
                                                 IODescriptions({IODescription(ov::PartialShape({2}), {}),
                                                                 IODescription(ov::PartialShape({200}), {}),
                                                                 IODescription(ov::PartialShape({400}), {})}),
                                                 configs, true)),
                         AutoBatchCompilerDetectionBasedOnOVTests::getTestCaseName);

class AutoBatchCompilerDetectionModelSuitableForDebatchTests : public AutoBatchCompilerDetectionTests {};
TEST_P(AutoBatchCompilerDetectionModelSuitableForDebatchTests, testModel) {
    bool isSuitable = isModelSuitableForDebatching(ov_stub_model, *configurationPtr, logger);
    EXPECT_EQ(isSuitable, result);
}

CfgMap modelDebatchSuitableDefaultConfig;
CfgMap modelNoLimitsDebatchSuitableConfig{
        {std::make_pair(vpux::OV::BATCH_COMPILER_MODE_SETTINGS::key().data(),
                        "batch-compile-method=debatch debatcher-settings={model-ops-number-enable-threshold=0 "
                        "max-batch-number-disable-limit=-1}")}};
CfgMap modelOpThresholdNotReachedDebatchSuitableConfig{
        {std::make_pair(vpux::OV::BATCH_COMPILER_MODE_SETTINGS::key().data(),
                        "batch-compile-method=debatch debatcher-settings={model-ops-number-enable-threshold=0 "
                        "max-batch-number-disable-limit=10}")}};
CfgMap modelBatchLimitExceedDebatchSuitableConfig{
        {std::make_pair(vpux::OV::BATCH_COMPILER_MODE_SETTINGS::key().data(),
                        "batch-compile-method=debatch debatcher-settings={model-ops-number-enable-threshold=1000 "
                        "max-batch-number-disable-limit=-1}")}};
INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTest, AutoBatchCompilerDetectionModelSuitableForDebatchTests,
        testing::Values(AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), {}),
                                                IODescription(ov::PartialShape({100, 200}), {}),
                                                IODescription(ov::PartialShape({300, 400}), {})}),
                                modelDebatchSuitableDefaultConfig, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{})}),
                                modelNoLimitsDebatchSuitableConfig, true),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"NCHW"}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{"HW"}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{"HW"})}),
                                modelOpThresholdNotReachedDebatchSuitableConfig, false),
                        AutoBatchCompilerDetectionTestParams(
                                IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"NCHW"}),
                                                IODescription(ov::PartialShape({100, 200}), ov::Layout{"..."}),
                                                IODescription(ov::PartialShape({300, 400}), ov::Layout{"NC"})}),
                                modelBatchLimitExceedDebatchSuitableConfig, false)),
        AutoBatchCompilerDetectionModelSuitableForDebatchTests::getTestCaseName);

using AutoBatchCompilerDebatchCoefficientsDeterminingTestsParams =
        CatenatedTestParams<AutoBatchCompilerDetectionTestsBaseParams, std::tuple<std::string>>;
class AutoBatchCompilerDebatchCoefficientsDeterminingTests : public AutoBatchCompilerDetectionTestsBase<std::string> {
public:
    using Base = AutoBatchCompilerDetectionTestsBase<std::string>;
    static constexpr size_t getFirstParamTupleIndex() {
        return Base::getBasicParamSize();
    }
    void SetUp() override {
        optionDescPtr = std::make_shared<vpux::OV::OptionsDesc>();
        // allow any other additional config params to be passed as an external option into the plugin config
        // required for checking that auto-batch detection is non-intrusive methof of options processing
        optionDescPtr->add<vpux::OV::PERFORMANCE_HINT>();
        optionDescPtr->add<vpux::OV::COMPILATION_MODE>();

        Base::SetUp();
        refinedCompileParams = std::get<getFirstParamTupleIndex()>(this->GetParam());
    }

    static std::string getTestCaseName(
            const testing::TestParamInfo<AutoBatchCompilerDebatchCoefficientsDeterminingTestsParams>& obj) {
        std::ostringstream result;
        result << Base::getTestCaseName(obj)
               << "_refinedCompileParams=" << std::get<getFirstParamTupleIndex()>(obj.param);
        return result.str();
    }

protected:
    std::string refinedCompileParams;
};

TEST_P(AutoBatchCompilerDebatchCoefficientsDeterminingTests, makeDebatchCoeffExtraction) {
    vpux::OV::Config refinedConfig =
            std::get<0>(vpux::autoDetectBatchedModelIfPossible(ov_stub_model, *configurationPtr));
    // check equality of modified string and expected string.
    // As it's expected that modified string will contain some extra spaces,
    // direct comparison using EXPECT_EQ is not applicable here because it's not space-symbol agnostic
    // so that `isStrEqualSpaceAgnostic` is employed here.
    // At first, we check this function result of space-symbol agnostic comparison.
    // We don't use EXPECT_TRUE here, because the macro won't print compared values, which are beneficial for test
    // results analysis rather than just true/false provided by EXPECT_TRUE. Once `isStrEqualSpaceAgnostic` has failed,
    // a macro EXPECT_EQ will print failed values which are not met space-symbol agnostic comparison condition above
    if (!test_utils::isStrEqualSpaceAgnostic(refinedConfig.get<vpux::OV::BATCH_COMPILER_MODE_SETTINGS>(),
                                             refinedCompileParams)) {
        EXPECT_EQ(refinedConfig.get<vpux::OV::BATCH_COMPILER_MODE_SETTINGS>(), refinedCompileParams);
    }
}

const CfgMap coeffDeterminingConfigBatchEmpty;
const CfgMap coeffDeterminingConfigBatchInAuto{{std::make_pair(vpux::OV::BATCH_MODE::key().data(), "AUTO")}};
const std::string turnOffDebatchDisableConditions{
        "batch-compile-method=debatch debatcher-settings={model-ops-number-enable-threshold=0 "
        "max-batch-number-disable-limit=-1}"};
const CfgMap coeffDeterminingConfigBatchInCompiler{
        {std::make_pair(vpux::OV::BATCH_MODE::key().data(), "COMPILER"),
         std::make_pair(vpux::OV::BATCH_COMPILER_MODE_SETTINGS::key().data(), turnOffDebatchDisableConditions)}};
const CfgMap coeffDeterminingConfigBatchInCompilerForHostCompile{
        {std::make_pair(vpux::OV::BATCH_MODE::key().data(), "COMPILER"),
         std::make_pair(vpux::OV::COMPILATION_MODE::key().data(), "HostCompile"),
         std::make_pair(vpux::OV::BATCH_COMPILER_MODE_SETTINGS::key().data(), turnOffDebatchDisableConditions)}};
const CfgMap coeffDeterminingConfigBatchInPlugin{{std::make_pair(vpux::OV::BATCH_MODE::key().data(), "PLUGIN")}};
const std::string predefinedDebatchCoefficients{
        "batch-compile-method=debatch batch-unroll-settings=" + vpux::BatchUnrollOptions::getDefaultOptions() +
        " debatcher-settings={debatcher-input-coefficients-partitions=[10-10], "
        "debatching-inlining-method=naive}"};
const CfgMap coeffDeterminingConfigBatchInCompilerWithOverridedCoefficients{
        {std::make_pair(vpux::OV::BATCH_MODE::key().data(), "COMPILER"),
         std::make_pair(vpux::OV::BATCH_COMPILER_MODE_SETTINGS::key().data(), predefinedDebatchCoefficients)}};

const vpux::DebatchCoefficients sixCoefficientsWithBatchByOne =
        vpux::DebatchCoefficients::create("[0-1],[0-1],[0-1],[0-1],[0-1],[0-1]").value();
INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTest, AutoBatchCompilerDebatchCoefficientsDeterminingTests,
        testing::Values(
                AutoBatchCompilerDebatchCoefficientsDeterminingTestsParams(
                        IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), {}),
                                        IODescription(ov::PartialShape({2, 200}), {}),
                                        IODescription(ov::PartialShape({2, 400}), {})}),
                        coeffDeterminingConfigBatchEmpty, ""),
                AutoBatchCompilerDebatchCoefficientsDeterminingTestsParams(
                        IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), {}),
                                        IODescription(ov::PartialShape({2, 200}), {}),
                                        IODescription(ov::PartialShape({2, 400}), {})}),
                        coeffDeterminingConfigBatchInPlugin, ""),
                AutoBatchCompilerDebatchCoefficientsDeterminingTestsParams(
                        IODescriptions({IODescription(ov::PartialShape({1, 3, 100, 200}), {}),
                                        IODescription(ov::PartialShape({2, 200}), {}),
                                        IODescription(ov::PartialShape({300, 400}), {})}),
                        coeffDeterminingConfigBatchInCompiler,
                        "batch-compile-method=debatch batch-unroll-settings=" +
                                vpux::BatchUnrollOptions::getDefaultOptions() +
                                " debatcher-settings={debatcher-input-coefficients-partitions=" +
                                sixCoefficientsWithBatchByOne.to_string() + ", debatching-inlining-method=naive}"),
                AutoBatchCompilerDebatchCoefficientsDeterminingTestsParams(
                        IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), {}),
                                        IODescription(ov::PartialShape({100, 200}), {}),
                                        IODescription(ov::PartialShape({300, 400}), {})}),
                        coeffDeterminingConfigBatchInCompiler,
                        "batch-compile-method=debatch batch-unroll-settings=" +
                                vpux::BatchUnrollOptions::getDefaultOptions() +
                                " debatcher-settings={debatcher-input-coefficients-partitions=" +
                                sixCoefficientsWithBatchByOne.to_string() + ", debatching-inlining-method=naive}"),
                AutoBatchCompilerDebatchCoefficientsDeterminingTestsParams(
                        IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"NCHW"}),
                                        IODescription(ov::PartialShape({100, 200}), ov::Layout{"NC"}),
                                        IODescription(ov::PartialShape({300, 400}), ov::Layout{"NC"})}),
                        coeffDeterminingConfigBatchInCompiler,
                        "batch-compile-method=debatch batch-unroll-settings=" +
                                vpux::BatchUnrollOptions::getDefaultOptions() +
                                " debatcher-settings={debatcher-input-coefficients-partitions=" +
                                sixCoefficientsWithBatchByOne.to_string() + ", debatching-inlining-method=naive}"),
                AutoBatchCompilerDebatchCoefficientsDeterminingTestsParams(
                        IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"NCHW"}),
                                        IODescription(ov::PartialShape({100, 200}), ov::Layout{"NC"}),
                                        IODescription(ov::PartialShape({300, 400}), ov::Layout{"NC"})}),
                        coeffDeterminingConfigBatchInCompilerForHostCompile,
                        "batch-compile-method=debatch batch-unroll-settings=" +
                                vpux::BatchUnrollOptions::getDefaultOptions() +
                                " debatcher-settings={debatcher-input-coefficients-partitions=" +
                                sixCoefficientsWithBatchByOne.to_string() +
                                ", debatching-inlining-method=host_pipeline}"),
                AutoBatchCompilerDebatchCoefficientsDeterminingTestsParams(
                        IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), {}),
                                        IODescription(ov::PartialShape({100, 200}), {}),
                                        IODescription(ov::PartialShape({300, 400}), {})}),
                        coeffDeterminingConfigBatchInCompilerWithOverridedCoefficients, predefinedDebatchCoefficients),
                AutoBatchCompilerDebatchCoefficientsDeterminingTestsParams(
                        IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), ov::Layout{"DCHW"}),
                                        IODescription(ov::PartialShape({100, 200}), ov::Layout{"HW"}),
                                        IODescription(ov::PartialShape({300, 400}), ov::Layout{"HW"})}),
                        coeffDeterminingConfigBatchInCompiler, turnOffDebatchDisableConditions)),
        AutoBatchCompilerDebatchCoefficientsDeterminingTests::getTestCaseName);

using AutoBatchCompilerDetectionCompatTestsParams =
        CatenatedTestParams<AutoBatchCompilerDetectionTestsBaseParams, std::tuple<bool>>;
class AutoBatchCompilerDetectionCompatTests : public AutoBatchCompilerDetectionTestsBase<bool> {
public:
    using Base = AutoBatchCompilerDetectionTestsBase<bool>;
    static constexpr size_t getFirstParamTupleIndex() {
        return Base::getBasicParamSize();
    }

    void SetUp() override {
        Base::SetUp();
        mustBeValid = std::get<getFirstParamTupleIndex()>(this->GetParam());
    }

    static std::string getTestCaseName(const testing::TestParamInfo<AutoBatchCompilerDetectionCompatTestsParams>& obj) {
        std::ostringstream result;
        result << Base::getTestCaseName(obj) << "mustBeValid=" << std::get<getFirstParamTupleIndex()>(obj.param);
        return result.str();
    }

protected:
    bool mustBeValid;
};

TEST_P(AutoBatchCompilerDetectionCompatTests, OptionsCompatibility) {
    std::stringstream sstream;
    EXPECT_EQ(vpux::checkCfgOnBatchOptionConsistency(*configurationPtr, sstream), mustBeValid);
    (void)sstream;
}

const CfgMap cfgDiscrepancy{
        {std::make_pair(vpux::OV::BATCH_MODE::key().data(), "PLUGIN"),
         std::make_pair(vpux::OV::BATCH_COMPILER_MODE_SETTINGS::key().data(), predefinedDebatchCoefficients)}};
const CfgMap cfgCompilerBatchSuccess{
        {std::make_pair(vpux::OV::BATCH_MODE::key().data(), "COMPILER"),
         std::make_pair(vpux::OV::BATCH_COMPILER_MODE_SETTINGS::key().data(), predefinedDebatchCoefficients)}};
const CfgMap cfgAutoBatchSuccess{
        {std::make_pair(vpux::OV::BATCH_MODE::key().data(), "AUTO"),
         std::make_pair(vpux::OV::BATCH_COMPILER_MODE_SETTINGS::key().data(), predefinedDebatchCoefficients)}};
const CfgMap cfgDefaultBatchSuccess{
        {std::make_pair(vpux::OV::BATCH_COMPILER_MODE_SETTINGS::key().data(), predefinedDebatchCoefficients)}};
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, AutoBatchCompilerDetectionCompatTests,
                         testing::Values(AutoBatchCompilerDetectionCompatTestsParams(
                                                 IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), {}),
                                                                 IODescription(ov::PartialShape({2, 200}), {}),
                                                                 IODescription(ov::PartialShape({2, 400}), {})}),
                                                 cfgDiscrepancy, false),
                                         AutoBatchCompilerDetectionCompatTestsParams(
                                                 IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), {}),
                                                                 IODescription(ov::PartialShape({2, 200}), {}),
                                                                 IODescription(ov::PartialShape({2, 400}), {})}),
                                                 cfgCompilerBatchSuccess, true),
                                         AutoBatchCompilerDetectionCompatTestsParams(
                                                 IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), {}),
                                                                 IODescription(ov::PartialShape({2, 200}), {}),
                                                                 IODescription(ov::PartialShape({2, 400}), {})}),
                                                 cfgAutoBatchSuccess, true),
                                         AutoBatchCompilerDetectionCompatTestsParams(
                                                 IODescriptions({IODescription(ov::PartialShape({2, 3, 100, 200}), {}),
                                                                 IODescription(ov::PartialShape({2, 200}), {}),
                                                                 IODescription(ov::PartialShape({2, 400}), {})}),
                                                 cfgDefaultBatchSuccess, true)),
                         AutoBatchCompilerDetectionCompatTests::getTestCaseName);

}  // namespace

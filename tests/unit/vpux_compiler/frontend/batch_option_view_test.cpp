//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/op/relu.hpp>
#include <string>
#include <vector>

#include "common/utils.hpp"
#include "frontend/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "vpux/compiler/frontend/ov_batch_detection.hpp"
#include "vpux/compiler/utils/batch.hpp"

namespace {

using BatchCompilerOptionsAdapterViewTestsParams = std::tuple<std::string, std::string>;
class AutoBatchCompilerOptionsAdapterViewTests :
        public testing::TestWithParam<BatchCompilerOptionsAdapterViewTestsParams> {
public:
    void SetUp() override {
        std::tie(inOptions, outOptions) = GetParam();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<BatchCompilerOptionsAdapterViewTestsParams>& obj) {
        std::string inOptions, outOptions;
        std::tie(inOptions, outOptions) = obj.param;
        std::ostringstream result;
        result << "inOptions=" << inOptions << "_out_option=" << outOptions;
        return result.str();
    }

protected:
    std::string inOptions;
    std::string outOptions;
};

TEST_P(AutoBatchCompilerOptionsAdapterViewTests, ParseAndValidate) {
    auto view = vpux::BatchCompilerOptionsAdapterView::tryExtractFromString(inOptions);
    EXPECT_EQ(view.has_value(), !outOptions.empty());
    if (!outOptions.empty()) {
        EXPECT_EQ(view->print(), outOptions);
    }
}

const vpux::DebatchCoefficients mixedCoefficients = vpux::DebatchCoefficients::create("[0-2],[0-2],[0-1]").value();
INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTest, AutoBatchCompilerOptionsAdapterViewTests,
        testing::Values(BatchCompilerOptionsAdapterViewTestsParams{"", ""},
                        BatchCompilerOptionsAdapterViewTestsParams{std::string("Bbatch-compileD-MmethodIC="), ""},
                        BatchCompilerOptionsAdapterViewTestsParams{
                                "batch-compile-method=unroll",
                                "batch-compile-method=unroll batch-unroll-settings=" +
                                        vpux::BatchUnrollOptions::getDefaultOptions() +
                                        " debatcher-settings={debatcher-input-coefficients-partitions= "
                                        "debatching-inlining-method=naive max-batch-number-disable-limit=-1 "
                                        "model-ops-number-enable-threshold=0}"},
                        BatchCompilerOptionsAdapterViewTestsParams{
                                "batch-compile-method=",
                                "batch-compile-method= batch-unroll-settings=" +
                                        vpux::BatchUnrollOptions::getDefaultOptions() +
                                        " debatcher-settings={debatcher-input-coefficients-partitions= "
                                        "debatching-inlining-method=naive max-batch-number-disable-limit=-1 "
                                        "model-ops-number-enable-threshold=0}"},
                        BatchCompilerOptionsAdapterViewTestsParams{
                                "batch-compile-method=debatch "
                                "debatcher-settings={debatcher-input-coefficients-partitions=" +
                                        mixedCoefficients.to_string() + ", debatching-inlining-method=naive}",
                                "batch-compile-method=debatch "
                                "batch-unroll-settings=" +
                                        vpux::BatchUnrollOptions::getDefaultOptions() +
                                        " debatcher-settings={debatcher-input-coefficients-partitions=" +
                                        mixedCoefficients.to_string() + ", debatching-inlining-method=naive}"}),
        AutoBatchCompilerOptionsAdapterViewTests::getTestCaseName);

using BatchCompilerOptionsAdapterViewInjectionTestsParams = std::tuple<std::string, std::string, std::string>;
class AutoBatchCompilerOptionsAdapterViewInjectionTests :
        public testing::TestWithParam<BatchCompilerOptionsAdapterViewInjectionTestsParams> {
public:
    void SetUp() override {
        std::tie(inOptions, origStringToModify, stringToExpect) = GetParam();
    }

    static std::string getTestCaseName(
            const testing::TestParamInfo<BatchCompilerOptionsAdapterViewInjectionTestsParams>& obj) {
        std::string inOptions, origStringToModify, stringToExpect;
        std::tie(inOptions, origStringToModify, stringToExpect) = obj.param;
        std::ostringstream result;
        result << "_inOptions=" << inOptions << "_stringToInject=" << origStringToModify
               << "_stringToExpect=" << stringToExpect;
        return result.str();
    }

protected:
    std::string inOptions;
    std::string origStringToModify;
    std::string stringToExpect;
};

TEST_P(AutoBatchCompilerOptionsAdapterViewInjectionTests, ParseAndValidate) {
    auto view = vpux::BatchCompilerOptionsAdapterView::tryExtractFromString(inOptions);
    EXPECT_TRUE(view.has_value());
    auto modifiedInjectedStr = view->injectInto(origStringToModify);
    // check equality of modified string and expected string.
    // As it's expected that modified string will contain some extra spaces,
    // direct comparison using EXPECT_EQ is not applicable here because it's not space-symbol agnostic
    // so that `isStrEqualSpaceAgnostic` is employed here.
    // At first, we check this function result of space-symbol agnostic comparison.
    // We don't use EXPECT_TRUE here, because the macro won't print compared values, which are beneficial for test
    // results analysis rather than just true/false provided by EXPECT_TRUE. Once `isStrEqualSpaceAgnostic` has
    // failed, a macro EXPECT_EQ will print failed values which are not met space-symbol agnostic comparison
    // condition above
    if (!test_utils::isStrEqualSpaceAgnostic(modifiedInjectedStr, stringToExpect)) {
        EXPECT_EQ(modifiedInjectedStr, stringToExpect);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, AutoBatchCompilerOptionsAdapterViewInjectionTests,
                         testing::Values(
                                 BatchCompilerOptionsAdapterViewInjectionTestsParams{
                                         "batch-compile-method=unroll", "",
                                         "batch-compile-method=unroll batch-unroll-settings=" +
                                                 vpux::BatchUnrollOptions::getDefaultOptions() +
                                                 " debatcher-settings={debatcher-input-coefficients-partitions= "
                                                 "debatching-inlining-method=naive max-batch-number-disable-limit=-1 "
                                                 "model-ops-number-enable-threshold=0}"},
                                 BatchCompilerOptionsAdapterViewInjectionTestsParams{
                                         "batch-compile-method=debatch", "batch-compile-method= ",
                                         "batch-compile-method=debatch batch-unroll-settings=" +
                                                 vpux::BatchUnrollOptions::getDefaultOptions() +
                                                 " debatcher-settings={debatcher-input-coefficients-partitions= "
                                                 "debatching-inlining-method=naive max-batch-number-disable-limit=-1 "
                                                 "model-ops-number-enable-threshold=0}"},
                                 BatchCompilerOptionsAdapterViewInjectionTestsParams{
                                         "batch-compile-method=debatch "
                                         "debatcher-settings={debatcher-input-coefficients-partitions=" +
                                                 mixedCoefficients.to_string() + ", debatching-inlining-method=naive}",
                                         "batch-compile-method= debatcher-settings={}",
                                         "batch-compile-method=debatch "
                                         "batch-unroll-settings=" +
                                                 vpux::BatchUnrollOptions::getDefaultOptions() +
                                                 "debatcher-settings={debatcher-input-coefficients-partitions=" +
                                                 mixedCoefficients.to_string() + ", debatching-inlining-method=naive}"},
                                 BatchCompilerOptionsAdapterViewInjectionTestsParams{
                                         "batch-compile-method=debatch "
                                         "debatcher-settings={debatcher-input-coefficients-partitions=" +
                                                 mixedCoefficients.to_string() + ", debatching-inlining-method=naive} ",
                                         "batch-compile-method=unroll "
                                         "batch-unroll-settings={skip-unroll-batch=true} "
                                         "debatcher-settings={debatcher-input-coefficients-partitions=[888-888]"
                                         " debatching-inlining-method=somthing-other} ",
                                         "batch-compile-method=debatch "
                                         "batch-unroll-settings=skip-unroll-batch=true "
                                         "debatcher-settings={debatcher-input-coefficients-partitions=" +
                                                 mixedCoefficients.to_string() + ", debatching-inlining-method=naive} "}

                                 ),
                         AutoBatchCompilerOptionsAdapterViewInjectionTests::getTestCaseName);
}  // namespace

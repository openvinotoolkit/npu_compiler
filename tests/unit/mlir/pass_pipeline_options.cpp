//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/options.hpp"

#include <gtest/gtest.h>

using namespace vpux;

struct DevicePipelineOptions final : mlir::PassPipelineOptions<DevicePipelineOptions> {
    StrOption strOption{*this, "str-option-name", llvm::cl::desc("This option has some default value"),
                        llvm::cl::init("default-value")};
    StrOption strOptionEmpty{*this, "str-empty-option-name",
                             llvm::cl::desc("This option does not have a default value")};
};

// This is not a real test, but rather an example of how to use mlir::PassPipelineOptions
TEST(MLIR_PipelineOptionsTest, BehaviourTests) {
    // Empty string
    std::string noExplicitSettings = "";

    auto optionHasNoValue = DevicePipelineOptions::createFromString(noExplicitSettings);
    ASSERT_FALSE(optionHasNoValue->strOption.hasValue());
    ASSERT_TRUE(optionHasNoValue->strOption.getValue() == "default-value");

    ASSERT_FALSE(optionHasNoValue->strOptionEmpty.hasValue());
    // strOptionEmpty does not have a default value, but you can still safely call the getter
    ASSERT_TRUE(optionHasNoValue->strOptionEmpty.getValue() == "");

    // Explicit default value
    std::string defaultSettings = "str-option-name=default-value str-empty-option-name=";

    auto optionHasDefaultValue = DevicePipelineOptions::createFromString(defaultSettings);
    ASSERT_TRUE(optionHasDefaultValue->strOption.hasValue());
    ASSERT_TRUE(optionHasDefaultValue->strOption.getValue() == "default-value");

    ASSERT_TRUE(optionHasDefaultValue->strOptionEmpty.hasValue());
    ASSERT_TRUE(optionHasDefaultValue->strOptionEmpty.getValue() == "");

    // Explicit non-default value
    std::string nonDefaultSettings = "str-option-name=non-default-value str-empty-option-name=another-value";

    auto optionHasNonDefaultValue = DevicePipelineOptions::createFromString(nonDefaultSettings);
    ASSERT_TRUE(optionHasNonDefaultValue->strOption.hasValue());
    ASSERT_TRUE(optionHasNonDefaultValue->strOption.getValue() == "non-default-value");

    ASSERT_TRUE(optionHasNonDefaultValue->strOptionEmpty.hasValue());
    ASSERT_TRUE(optionHasNonDefaultValue->strOptionEmpty.getValue() == "another-value");

    // Explicit empty value
    std::string nonDefaultEmptySettings = "str-option-name=";

    auto optionHasEmptyValue = DevicePipelineOptions::createFromString(nonDefaultEmptySettings);
    ASSERT_TRUE(optionHasEmptyValue->strOption.hasValue());
    ASSERT_TRUE(optionHasEmptyValue->strOption.getValue() == "");
}

//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>

#include <vpux/utils/logger/common_logger.hpp>
#include <vpux/utils/logger/logger.hpp>
#include <vpux/utils/ov/config.hpp>
#include "vpux/utils/ov/options.hpp"

using namespace vpux;

namespace {

struct SimpleOption : vpux::OV::OptionBase<SimpleOption, bool> {
    static std::string_view key() {
        return "PUBLIC_OPT";
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {"DEPRECATED_OPT"};
    }

    static bool isPublic() {
        return true;
    }
};

struct PrivateOption : vpux::OV::OptionBase<PrivateOption, int64_t> {
    static std::string_view key() {
        return "PRIVATE_OPT";
    }

    static int64_t defaultValue() {
        return 42;
    }

    static void validateValue(int64_t val) {
        OPENVINO_ASSERT(val >= 0, "Got negative value");
    }

    static vpux::OV::OptionMode mode() {
        return vpux::OV::OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

class MLIRConfigTests : public testing::Test {
public:
    std::shared_ptr<vpux::OV::OptionsDesc> options;
    vpux::OV::Config conf;

    MLIRConfigTests(): options(std::make_shared<vpux::OV::OptionsDesc>()), conf(options) {
    }

    void SetUp() override {
        testing::Test::SetUp();

        vpux::Logger::global().setLevel(vpux::LogLevel::Warning);

        options->add<SimpleOption>();
        options->add<PrivateOption>();
    }
};

class MLIRConfigSerializationTests : public testing::Test {
public:
    std::shared_ptr<vpux::OV::OptionsDesc> options;
    vpux::OV::Config conf;

    MLIRConfigSerializationTests(): options(std::make_shared<vpux::OV::OptionsDesc>()), conf(options) {
    }
};

}  // namespace

TEST_F(MLIRConfigTests, GetSupported) {
    const auto publicOpts = options->getSupported(false);
    EXPECT_EQ(publicOpts, std::vector<std::string>({"PUBLIC_OPT"}));

    auto allOpts = options->getSupported(true);
    std::sort(allOpts.begin(), allOpts.end());
    EXPECT_EQ(allOpts, std::vector<std::string>({"PRIVATE_OPT", "PUBLIC_OPT"}));
}

TEST_F(MLIRConfigTests, UpdateAndValidate) {
    EXPECT_NO_THROW(conf.update({{"PUBLIC_OPT", "YES"}}));
    EXPECT_ANY_THROW(conf.update({{"PUBLIC_OPT", "2"}}));

    EXPECT_NO_THROW(conf.update({{"PRIVATE_OPT", "15"}}));
    EXPECT_ANY_THROW(conf.update({{"PRIVATE_OPT", "aaa"}}));
    EXPECT_ANY_THROW(conf.update({{"PRIVATE_OPT", "-1"}}));
}

TEST_F(MLIRConfigTests, UpdateAndValidateNullptr) {
    EXPECT_NO_THROW(conf.update({{"PUBLIC_OPT", "YES"}}));
    EXPECT_ANY_THROW(SimpleOption::parse(""));
    EXPECT_NO_THROW(conf.update({{"PRIVATE_OPT", "15"}}));
    EXPECT_ANY_THROW(PrivateOption::parse(""));
}

TEST_F(MLIRConfigTests, UpdateAndHas) {
    EXPECT_FALSE(conf.has<SimpleOption>());
    EXPECT_FALSE(conf.has<PrivateOption>());

    conf.update({{"PUBLIC_OPT", "YES"}});
    EXPECT_TRUE(conf.has<SimpleOption>());

    conf.update({{"PRIVATE_OPT", "5"}});
    EXPECT_TRUE(conf.has<PrivateOption>());
}

TEST_F(MLIRConfigTests, Get) {
    conf.update({{"PUBLIC_OPT", "YES"}, {"PRIVATE_OPT", "5"}});

    EXPECT_TRUE(conf.get<SimpleOption>());
    EXPECT_EQ(conf.get<PrivateOption>(), 5);
}

TEST_F(MLIRConfigTests, GetNonExist) {
    EXPECT_ANY_THROW(conf.get<SimpleOption>());
}

TEST_F(MLIRConfigTests, GetDefault) {
    EXPECT_EQ(conf.get<PrivateOption>(), 42);
}

TEST_F(MLIRConfigTests, Deprecated) {
    EXPECT_FALSE(conf.has<SimpleOption>());

    EXPECT_NO_THROW(conf.update({{"DEPRECATED_OPT", "NO"}}));
    EXPECT_TRUE(conf.has<SimpleOption>());
    EXPECT_FALSE(conf.get<SimpleOption>());

    EXPECT_ANY_THROW(conf.update({{"DEPRECATED_OPT", "2"}}));
}

TEST_F(MLIRConfigTests, Unsupported) {
    EXPECT_ANY_THROW(conf.update({{"UNSUPPORTED_OPT", "1"}}));
}

TEST(MLIR_ParseConfigStringTests, ParsesKeyValuePairs) {
    const auto parsed = vpux::OV::parseConfigString(R"(PUBLIC_OPT="YES" PRIVATE_OPT="5")");

    EXPECT_EQ(parsed, vpux::OV::Config::ConfigMap({{"PUBLIC_OPT", "YES"}, {"PRIVATE_OPT", "5"}}));
}

TEST(MLIR_ParseConfigStringTests, ParsesEmptyString) {
    EXPECT_TRUE(vpux::OV::parseConfigString("").empty());
}

TEST(MLIR_ParseConfigStringTests, DoesNotValidateAgainstAnyOptionsDesc) {
    // What the driver compiler relies on to tell an unknown option from an unsupported one: parsing
    // reports the keys as written, leaving it to the caller to decide which ones are acceptable.
    const auto parsed = vpux::OV::parseConfigString(R"(NOT_AN_OPTION="whatever")");

    EXPECT_EQ(parsed, vpux::OV::Config::ConfigMap({{"NOT_AN_OPTION", "whatever"}}));
}

TEST(MLIR_ParseConfigStringTests, ThrowsOnMalformedToken) {
    EXPECT_ANY_THROW(vpux::OV::parseConfigString("PUBLIC_OPT"));
    EXPECT_ANY_THROW(vpux::OV::parseConfigString("PUBLIC_OPT="));
    EXPECT_ANY_THROW(vpux::OV::parseConfigString(R"(PUBLIC_OPT="YES" BROKEN)"));
}

TEST(MLIR_ParseConfigStringTests, ParsesValuesWithSpaces) {
    const auto parsed = vpux::OV::parseConfigString(
            R"(NPU_COMPILATION_MODE_PARAMS="dynamic-dim-alignment=false auto-unrolling-mode=disabled" )"
            R"(NPU_PLATFORM="4000")");

    EXPECT_EQ(parsed, vpux::OV::Config::ConfigMap({{"NPU_COMPILATION_MODE_PARAMS",
                                                    "dynamic-dim-alignment=false auto-unrolling-mode=disabled"},
                                                   {"NPU_PLATFORM", "4000"}}));
}

TEST(MLIR_ParseConfigStringTests, ParsesValuesWithQuotes) {
    // A pair ends at the first word ending with a quote, so a quote inside a value is kept as long as it is not
    // followed by whitespace
    const auto parsed = vpux::OV::parseConfigString(R"(NPU_COMPILATION_MODE_PARAMS="opt="x"" STRING_OPT="a"b")");

    EXPECT_EQ(parsed,
              vpux::OV::Config::ConfigMap({{"NPU_COMPILATION_MODE_PARAMS", R"(opt="x")"}, {"STRING_OPT", R"(a"b)"}}));
}

TEST(MLIR_ParseConfigStringTests, CollapsesWhitespace) {
    const auto parsed = vpux::OV::parseConfigString("  PUBLIC_OPT=\"YES\"\t\nSTRING_OPT=\"MY   VALUE\"  ");

    EXPECT_EQ(parsed, vpux::OV::Config::ConfigMap({{"PUBLIC_OPT", "YES"}, {"STRING_OPT", "MY VALUE"}}));
}

TEST_F(MLIRConfigSerializationTests, CanDumpConfigToString) {
    struct StringOption final : vpux::OV::OptionBase<StringOption, std::string> {
        static std::string_view key() {
            return "STRING_OPT";
        }

        static std::string defaultValue() {
            return "VALUE";
        }
    };

    options->add<SimpleOption>();
    options->add<PrivateOption>();
    options->add<StringOption>();

    conf.update({{"PRIVATE_OPT", "5"}, {"PUBLIC_OPT", "NO"}, {"STRING_OPT", "AAA"}});
    std::string expected1 = "STRING_OPT=\"AAA\"";
    std::string expected2 = "PUBLIC_OPT=\"NO\"";
    std::string expected3 = "PRIVATE_OPT=\"5\"";

    EXPECT_TRUE(conf.toString().find(expected1) != std::string::npos);
    EXPECT_TRUE(conf.toString().find(expected2) != std::string::npos);
    EXPECT_TRUE(conf.toString().find(expected3) != std::string::npos);
}

TEST_F(MLIRConfigSerializationTests, CanDumpConfigWithDoubleToString) {
    struct DoubleOption final : vpux::OV::OptionBase<DoubleOption, double> {
        static std::string_view key() {
            return "DOUBLE_OPT";
        }

        static double defaultValue() {
            return 0.0;
        }
    };
    options->add<DoubleOption>();

    conf.update({{"DOUBLE_OPT", "1.0"}});
    std::string expected = "DOUBLE_OPT=\"1.00\"";

    EXPECT_EQ(expected, conf.toString());
}

TEST_F(MLIRConfigSerializationTests, CanDumpConfigWithSpacesToString) {
    struct StringWithSpacesOption final : vpux::OV::OptionBase<StringWithSpacesOption, std::string> {
        static std::string_view key() {
            return "STRING_WITH_SPACES_OPT";
        }

        static std::string defaultValue() {
            return "MY DEFAULT VALUE WITH SPACES";
        }
    };
    options->add<StringWithSpacesOption>();

    conf.update({{"STRING_WITH_SPACES_OPT", "MY VALUE WITH SPACES"}});
    std::string expected = "STRING_WITH_SPACES_OPT=\"MY VALUE WITH SPACES\"";

    EXPECT_EQ(expected, conf.toString());
}

TEST_F(MLIRConfigSerializationTests, CanRoundTripConfigThroughString) {
    options->add<SimpleOption>();
    options->add<PrivateOption>();
    options->add<vpux::OV::COMPILATION_MODE_PARAMS>();

    const vpux::OV::Config::ConfigMap values = {
            {"PUBLIC_OPT", "NO"},
            {"PRIVATE_OPT", "5"},
            {"NPU_COMPILATION_MODE_PARAMS", "dynamic-dim-alignment=false auto-unrolling-mode=disabled"}};
    conf.update(values);

    EXPECT_EQ(vpux::OV::parseConfigString(conf.toString()), values);

    vpux::OV::Config restored(options);
    restored.fromString(conf.toString());

    EXPECT_FALSE(restored.get<SimpleOption>());
    EXPECT_EQ(restored.get<PrivateOption>(), 5);
    EXPECT_EQ(restored.get<vpux::OV::COMPILATION_MODE_PARAMS>(),
              "dynamic-dim-alignment=false auto-unrolling-mode=disabled");
}

TEST_F(MLIRConfigSerializationTests, CanDumpLogLevel) {
    options->add<vpux::OV::LOG_LEVEL>();

    conf.update({{"LOG_LEVEL", "Trace"}});
    std::string expected = "LOG_LEVEL=\"Trace\"";

    EXPECT_EQ(expected, conf.toString());
}

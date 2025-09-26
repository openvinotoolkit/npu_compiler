//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/utils/IE/config.hpp"

#include "intel_npu/config/options.hpp"

#include <gtest/gtest.h>

using namespace vpux;
using namespace intel_npu;

struct SimpleOption : OptionBase<SimpleOption, bool> {
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

struct PrivateOption : OptionBase<PrivateOption, int64_t> {
    static std::string_view key() {
        return "PRIVATE_OPT";
    }

    static int64_t defaultValue() {
        return 42;
    }

    static void validateValue(int64_t val) {
        OPENVINO_ASSERT(val >= 0, "Got negative value");
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

class MLIR_ConfigTests : public testing::Test {
public:
    std::shared_ptr<OptionsDesc> options;
    Config conf;

    MLIR_ConfigTests(): options(std::make_shared<OptionsDesc>()), conf(options) {
    }

    void SetUp() override {
        testing::Test::SetUp();

        intel_npu::Logger::global().setLevel(ov::log::Level::WARNING);

        options->add<SimpleOption>();
        options->add<PrivateOption>();
    }
};

TEST_F(MLIR_ConfigTests, GetSupported) {
    const auto publicOpts = options->getSupported(false);
    EXPECT_EQ(publicOpts, std::vector<std::string>({"PUBLIC_OPT"}));

    auto allOpts = options->getSupported(true);
    std::sort(allOpts.begin(), allOpts.end());
    EXPECT_EQ(allOpts, std::vector<std::string>({"PRIVATE_OPT", "PUBLIC_OPT"}));
}

TEST_F(MLIR_ConfigTests, UpdateAndValidate) {
    EXPECT_NO_THROW(conf.update({{"PUBLIC_OPT", "YES"}}));
    EXPECT_ANY_THROW(conf.update({{"PUBLIC_OPT", "2"}}));

    EXPECT_NO_THROW(conf.update({{"PRIVATE_OPT", "15"}}));
    EXPECT_ANY_THROW(conf.update({{"PRIVATE_OPT", "aaa"}}));
    EXPECT_ANY_THROW(conf.update({{"PRIVATE_OPT", "-1"}}));
}

TEST_F(MLIR_ConfigTests, UpdateAndValidateNullptr) {
    EXPECT_NO_THROW(conf.update({{"PUBLIC_OPT", "YES"}}));
    EXPECT_ANY_THROW(SimpleOption::parse(""));
    EXPECT_NO_THROW(conf.update({{"PRIVATE_OPT", "15"}}));
    EXPECT_ANY_THROW(PrivateOption::parse(""));
}

TEST_F(MLIR_ConfigTests, UpdateAndHas) {
    EXPECT_FALSE(conf.has<SimpleOption>());
    EXPECT_FALSE(conf.has<PrivateOption>());

    conf.update({{"PUBLIC_OPT", "YES"}});
    EXPECT_TRUE(conf.has<SimpleOption>());

    conf.update({{"PRIVATE_OPT", "5"}}, OptionMode::RunTime);
    EXPECT_TRUE(conf.has<PrivateOption>());
}

TEST_F(MLIR_ConfigTests, Get) {
    conf.update({{"PUBLIC_OPT", "YES"}, {"PRIVATE_OPT", "5"}});

    EXPECT_TRUE(conf.get<SimpleOption>());
    EXPECT_EQ(conf.get<PrivateOption>(), 5);
}

TEST_F(MLIR_ConfigTests, GetNonExist) {
    EXPECT_ANY_THROW(conf.get<SimpleOption>());
}

TEST_F(MLIR_ConfigTests, GetDefault) {
    EXPECT_EQ(conf.get<PrivateOption>(), 42);
}

TEST_F(MLIR_ConfigTests, Deprecated) {
    EXPECT_FALSE(conf.has<SimpleOption>());

    EXPECT_NO_THROW(conf.update({{"DEPRECATED_OPT", "NO"}}));
    EXPECT_TRUE(conf.has<SimpleOption>());
    EXPECT_FALSE(conf.get<SimpleOption>());

    EXPECT_ANY_THROW(conf.update({{"DEPRECATED_OPT", "2"}}));
}

TEST_F(MLIR_ConfigTests, Unsupported) {
    EXPECT_ANY_THROW(conf.update({{"UNSUPPORTED_OPT", "1"}}));
}

class MLIR_ConfigSerializationTests : public testing::Test {
public:
    std::shared_ptr<OptionsDesc> options;
    Config conf;

    MLIR_ConfigSerializationTests(): options(std::make_shared<OptionsDesc>()), conf(options) {
    }
};

TEST_F(MLIR_ConfigSerializationTests, CanDumpConfigToString) {
    struct StringOption final : OptionBase<StringOption, std::string> {
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

TEST_F(MLIR_ConfigSerializationTests, CanDumpConfigWithDoubleToString) {
    struct DoubleOption final : OptionBase<DoubleOption, double> {
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

TEST_F(MLIR_ConfigSerializationTests, CanDumpConfigWithSpacesToString) {
    struct StringWithSpacesOption final : OptionBase<StringWithSpacesOption, std::string> {
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

TEST_F(MLIR_ConfigSerializationTests, CanDumpLogLevel) {
    options->add<intel_npu::LOG_LEVEL>();

    conf.update({{"LOG_LEVEL", "LOG_TRACE"}});
    std::string expected = "LOG_LEVEL=\"LOG_TRACE\"";

    EXPECT_EQ(expected, conf.toString());
}

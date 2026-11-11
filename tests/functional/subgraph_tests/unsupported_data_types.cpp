//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gtest/gtest_pred_impl.h>
#include "common/npu_test_env_cfg.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

using UnsupportedTypeParams = ov::element::Type_t;

class UnsupportedDataTypeTest : public VpuOv2LayerTest, public testing::WithParamInterface<UnsupportedTypeParams> {
public:
    void SetUp() override {
        const ov::element::Type elemType(GetParam());
        const auto param = std::make_shared<ov::op::v0::Parameter>(elemType, ov::Shape{1, 8});
        const auto result = std::make_shared<ov::op::v0::Result>(param->output(0));
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param},
                                               "UnsupportedTypeModel");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<UnsupportedTypeParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "elemType=" << ov::element::Type(obj.param).get_type_name();
        return result.str();
    }

    void runNegativeTest(const std::string_view platform) {
        configuration[ov::intel_npu::platform.name()] = std::string(platform);
        targetDevice = test_utils::TARGET_DEVICE;
        const std::string expectedMsg =
                std::string("Unsupported data type '") + ov::element::Type(GetParam()).get_type_name() + "'";
        OV_EXPECT_THROW_HAS_SUBSTRING(compile_model(), std::runtime_error, expectedMsg);
    }
};

// Per-architecture subclasses bind each TEST_P to its own INSTANTIATE_TEST_SUITE_P
// so that only unsupported types for that architecture are instantiated.
class UnsupportedDataTypeTestNPU37xx : public UnsupportedDataTypeTest {};
class UnsupportedDataTypeTestNPU40xx : public UnsupportedDataTypeTest {};
class UnsupportedDataTypeTestNPU50xx : public UnsupportedDataTypeTest {};
class UnsupportedDataTypeTestNPU60xx : public UnsupportedDataTypeTest {};

static const std::set<ov::element::Type_t> allSupportedTypes = {
        ov::element::Type_t::dynamic, ov::element::Type_t::boolean, ov::element::Type_t::bf16,
        ov::element::Type_t::f16,     ov::element::Type_t::f32,     ov::element::Type_t::f64,
        ov::element::Type_t::i4,      ov::element::Type_t::i8,      ov::element::Type_t::i16,
        ov::element::Type_t::i32,     ov::element::Type_t::i64,     ov::element::Type_t::u1,
        ov::element::Type_t::u2,      ov::element::Type_t::u4,      ov::element::Type_t::u8,
        ov::element::Type_t::u16,     ov::element::Type_t::u32,     ov::element::Type_t::u64,
        ov::element::Type_t::nf4,     ov::element::Type_t::f8e4m3,  ov::element::Type_t::f8e5m2,
        ov::element::Type_t::f8e8m0,
};

// Supported types per architecture from compiler.cpp isTypeSupported functions
static const std::set<ov::element::Type_t> supportedNPU37xx = {
        ov::element::Type_t::dynamic, ov::element::Type_t::boolean, ov::element::Type_t::bf16, ov::element::Type_t::f16,
        ov::element::Type_t::f32,     ov::element::Type_t::f64,     ov::element::Type_t::i4,   ov::element::Type_t::i8,
        ov::element::Type_t::i16,     ov::element::Type_t::i32,     ov::element::Type_t::i64,  ov::element::Type_t::u4,
        ov::element::Type_t::u8,      ov::element::Type_t::u16,     ov::element::Type_t::u32,  ov::element::Type_t::u64,
};

static const std::set<ov::element::Type_t> supportedNPU40xx = [] {
    auto s = supportedNPU37xx;
    s.insert({ov::element::Type_t::u2, ov::element::Type_t::nf4});
    return s;
}();

static const std::set<ov::element::Type_t> supportedNPU50xx = [] {
    auto s = supportedNPU40xx;
    s.insert({ov::element::Type_t::f8e4m3, ov::element::Type_t::f8e5m2});
    return s;
}();

static const std::set<ov::element::Type_t> supportedNPU60xx = [] {
    auto s = supportedNPU50xx;
    s.insert({ov::element::Type_t::u1, ov::element::Type_t::f8e8m0});
    return s;
}();

static std::set<ov::element::Type_t> computeUnsupportedTypes(const std::set<ov::element::Type_t>& supportedTypes) {
    std::set<ov::element::Type_t> result;
    for (const auto t : allSupportedTypes) {
        if (!supportedTypes.count(t)) {
            result.insert(t);
        }
    }
    return result;
}

//
// Platform test definitions
//

TEST_P(UnsupportedDataTypeTestNPU37xx, NPU3720) {
    runNegativeTest(Platform::NPU3720);
}
TEST_P(UnsupportedDataTypeTestNPU40xx, NPU4000) {
    runNegativeTest(Platform::NPU4000);
}
TEST_P(UnsupportedDataTypeTestNPU50xx, NPU5010) {
    runNegativeTest(Platform::NPU5010);
}
TEST_P(UnsupportedDataTypeTestNPU50xx, NPU5020) {
    runNegativeTest(Platform::NPU5020);
}
TEST_P(UnsupportedDataTypeTestNPU60xx, NPU6010) {
    runNegativeTest(Platform::NPU6010);
}

INSTANTIATE_TEST_SUITE_P(unsupportedDataType, UnsupportedDataTypeTestNPU37xx,
                         ::testing::ValuesIn(computeUnsupportedTypes(supportedNPU37xx)),
                         UnsupportedDataTypeTestNPU37xx::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(unsupportedDataType, UnsupportedDataTypeTestNPU40xx,
                         ::testing::ValuesIn(computeUnsupportedTypes(supportedNPU40xx)),
                         UnsupportedDataTypeTestNPU40xx::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(unsupportedDataType, UnsupportedDataTypeTestNPU50xx,
                         ::testing::ValuesIn(computeUnsupportedTypes(supportedNPU50xx)),
                         UnsupportedDataTypeTestNPU50xx::getTestCaseName);
// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(UnsupportedDataTypeTestNPU60xx);
INSTANTIATE_TEST_SUITE_P(unsupportedDataType, UnsupportedDataTypeTestNPU60xx,
                         ::testing::ValuesIn(computeUnsupportedTypes(supportedNPU60xx)),
                         UnsupportedDataTypeTestNPU60xx::getTestCaseName);

}  // namespace ov::test::subgraph

//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/instructions.hpp"
#include "bytecode_builder.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "npu_interpreter_runtime/virtual_machine.h"
#include "vm_value_helpers.hpp"

#include <llvm/Support/FormatVariadic.h>

#include <common_test_utils/test_common.hpp>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <gtest/internal/gtest-param-util.h>

using namespace intel_npu::vm;
using namespace utils;

namespace {

intel_npu::vm::Type getInt64Type() {
    return intel_npu::vm::Type{intel_npu::vm::IntegerType{sizeof(int64_t) * CHAR_BIT, true}};
}

intel_npu::vm::Type getF64Type() {
    return intel_npu::vm::Type{intel_npu::vm::FloatType{sizeof(double) * CHAR_BIT, FloatTypeFormat::IEEE754}};
}

// Base fixture for VM instruction execution tests.
// Each test case receives a freshly default-constructed VirtualMachine.
class VMInstructionTest : public testing::Test {
public:
    npu_vm_module* module{nullptr};
    npu_vm_engine* engine{nullptr};
    npu_vm_function_info* funcInfo{nullptr};

    void TearDown() override {
        if (funcInfo != nullptr) {
            npu_vm_destroy_function_info(funcInfo);
        }
        if (engine != nullptr) {
            npu_vm_destroy_engine(engine);
        }
        if (module != nullptr) {
            npu_vm_destroy_module(module);
        }
    }
};

struct ThreeOperandInstructionParams {
    OpCode opcode;
    int64_t lhs;
    int64_t rhs;
    std::variant<int64_t, double> expected;
};

std::ostream& operator<<(std::ostream& os, const ThreeOperandInstructionParams& params) {
    if (std::holds_alternative<double>(params.expected)) {
        return os << llvm::formatv("{ opcode={0}, lhs={1}, rhs={2}, expected={3} }",
                                   static_cast<uint16_t>(params.opcode), params.lhs, params.rhs,
                                   std::get<double>(params.expected))
                             .str();
    }
    return os << llvm::formatv("{ opcode={0}, lhs={1}, rhs={2}, expected={3} }", static_cast<uint16_t>(params.opcode),
                               params.lhs, params.rhs, std::get<int64_t>(params.expected))
                         .str();
}

class VirtualMachineThreeOperandInstructionTest :
        public VMInstructionTest,
        public testing::WithParamInterface<ThreeOperandInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ThreeOperandInstructionParams>& obj) {
        std::ostringstream result;
        result << "opcode=" << static_cast<uint16_t>(obj.param.opcode) << "_";
        result << "lhs=" << obj.param.lhs << "_";
        result << "rhs=" << obj.param.rhs << "_";
        result << "expected=";
        if (std::holds_alternative<double>(obj.param.expected)) {
            result << std::get<double>(obj.param.expected);
        } else {
            result << std::get<int64_t>(obj.param.expected);
        }
        return result.str();
    }
};

class VirtualMachineCallInstructionTest :
        public VMInstructionTest,
        public testing::WithParamInterface<ThreeOperandInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ThreeOperandInstructionParams>& obj) {
        std::ostringstream result;
        result << "opcode=" << static_cast<uint16_t>(obj.param.opcode) << "_";
        result << "lhs=" << obj.param.lhs << "_";
        result << "rhs=" << obj.param.rhs << "_";
        result << "expected=";
        if (std::holds_alternative<double>(obj.param.expected)) {
            result << std::get<double>(obj.param.expected);
        } else {
            result << std::get<int64_t>(obj.param.expected);
        }
        return result.str();
    }
};

struct JmpSelectTestParams {
    int64_t input;
    int64_t expected;
};

std::ostream& operator<<(std::ostream& os, const JmpSelectTestParams& params) {
    return os << llvm::formatv("{ input={0}, expected={1} }", params.input, params.expected).str();
}

class VirtualMachineJmpTest : public VMInstructionTest, public testing::WithParamInterface<JmpSelectTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<JmpSelectTestParams>& obj) {
        std::ostringstream result;
        result << "input=" << obj.param.input << "_expected=" << obj.param.expected;
        return result.str();
    }
};

class VirtualMachineJmpBackwardTest :
        public VMInstructionTest,
        public testing::WithParamInterface<JmpSelectTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<JmpSelectTestParams>& obj) {
        std::ostringstream result;
        result << "input=" << obj.param.input << "_expected=" << obj.param.expected;
        return result.str();
    }
};

struct AssertInstructionParams {
    int64_t condition;
    int16_t messageIndex;
    npu_vm_result expectedStatus;
    std::optional<int64_t> expectedResult;
};

std::ostream& operator<<(std::ostream& os, const AssertInstructionParams& params) {
    return os << llvm::formatv("{ condition={0}, messageIndex={1}, expectedStatus={2}, expectedResult={3} }",
                               params.condition, params.messageIndex, static_cast<uint32_t>(params.expectedStatus),
                               params.expectedResult.has_value() ? std::to_string(*params.expectedResult) : "none")
                         .str();
}

class VirtualMachineAssertInstructionTest :
        public VMInstructionTest,
        public testing::WithParamInterface<AssertInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<AssertInstructionParams>& obj) {
        std::ostringstream result;
        result << "condition=" << obj.param.condition << "_";
        result << "messageIndex=" << obj.param.messageIndex << "_";
        result << "status=" << static_cast<uint32_t>(obj.param.expectedStatus) << "_";
        result << "result=";
        if (obj.param.expectedResult.has_value()) {
            result << *obj.param.expectedResult;
        } else {
            result << "none";
        }
        return result.str();
    }
};

inline int64_t f64Bits(double v) {
    int64_t bits;
    std::memcpy(&bits, &v, sizeof(v));
    return bits;
}

constexpr double kInfF64 = std::numeric_limits<double>::infinity();
constexpr double kNaNF64 = std::numeric_limits<double>::quiet_NaN();

struct TwoOperandInstructionParams {
    OpCode opcode;
    int64_t src;
    std::variant<int64_t, double> expected;
};

std::ostream& operator<<(std::ostream& os, const TwoOperandInstructionParams& params) {
    if (std::holds_alternative<double>(params.expected)) {
        return os << llvm::formatv("{ opcode={0}, src={1}, expected={2} }", static_cast<uint16_t>(params.opcode),
                                   params.src, std::get<double>(params.expected))
                             .str();
    }
    return os << llvm::formatv("{ opcode={0}, src={1}, expected={2} }", static_cast<uint16_t>(params.opcode),
                               params.src, std::get<int64_t>(params.expected))
                         .str();
}

class VirtualMachineTwoOperandInstructionTest :
        public VMInstructionTest,
        public testing::WithParamInterface<TwoOperandInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TwoOperandInstructionParams>& obj) {
        std::ostringstream result;
        result << "opcode=" << static_cast<uint16_t>(obj.param.opcode) << "_";
        result << "src=" << obj.param.src << "_";
        result << "expected=";
        if (std::holds_alternative<double>(obj.param.expected)) {
            result << std::get<double>(obj.param.expected);
        } else {
            result << std::get<int64_t>(obj.param.expected);
        }
        return result.str();
    }
};

struct RoundF64Params {
    int64_t src;
    RoundingMode flag;
    double expected;
};

std::ostream& operator<<(std::ostream& os, const RoundF64Params& p) {
    return os << llvm::formatv("{ src={0}, flag={1}, expected={2} }", p.src, static_cast<int16_t>(p.flag), p.expected)
                         .str();
}

class VirtualMachineRoundF64Test : public VMInstructionTest, public testing::WithParamInterface<RoundF64Params> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RoundF64Params>& obj) {
        std::ostringstream result;
        result << "flag=" << static_cast<int16_t>(obj.param.flag) << "_src=" << obj.param.src
               << "_expected=" << obj.param.expected;
        return result.str();
    }
};

constexpr int64_t kMinI64 = std::numeric_limits<int64_t>::min();
constexpr int64_t kMaxI64 = std::numeric_limits<int64_t>::max();

struct CmpInstructionParams {
    OpCode opcode;
    int64_t lhs;
    int64_t rhs;
    CmpPredicate predicate;
    std::optional<bool> isSigned;
    bool expected;

    CmpInstructionParams(OpCode opcode, int64_t lhs, int64_t rhs, CmpPredicate predicate, bool isSigned, bool expected)
            : opcode(opcode), lhs(lhs), rhs(rhs), predicate(predicate), isSigned(isSigned), expected(expected) {
    }
    CmpInstructionParams(OpCode opcode, int64_t lhs, int64_t rhs, CmpPredicate predicate, bool expected)
            : opcode(opcode), lhs(lhs), rhs(rhs), predicate(predicate), isSigned(std::nullopt), expected(expected) {
    }
};

std::ostream& operator<<(std::ostream& os, const CmpInstructionParams& params) {
    return os << llvm::formatv("{ opcode={0}, lhs={1}, rhs={2}, predicate={3}, isSigned={4}, expected={5} }",
                               static_cast<uint16_t>(params.opcode), params.lhs, params.rhs,
                               static_cast<uint16_t>(params.predicate),
                               params.isSigned.has_value() ? std::to_string(*params.isSigned) : "none", params.expected)
                         .str();
}

class VirtualMachineComparisonInstructionTest :
        public VMInstructionTest,
        public testing::WithParamInterface<CmpInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CmpInstructionParams>& obj) {
        const auto predicateToStr = [](CmpPredicate predicate) {
            switch (predicate) {
            case CmpPredicate::EQ:
                return "EQ";
            case CmpPredicate::NE:
                return "NE";
            case CmpPredicate::GT:
                return "GT";
            case CmpPredicate::GTE:
                return "GTE";
            case CmpPredicate::LT:
                return "LT";
            case CmpPredicate::LTE:
                return "LTE";
            default:
                return "unknown";
            }
        };

        std::ostringstream result;
        result << "opcode=" << static_cast<uint16_t>(obj.param.opcode) << "_";
        result << "lhs=" << obj.param.lhs << "_";
        result << "rhs=" << obj.param.rhs << "_";
        result << "predicate=" << predicateToStr(obj.param.predicate) << "_";
        result << "isSigned=" << (obj.param.isSigned.has_value() ? std::to_string(*obj.param.isSigned) : "none") << "_";
        result << "expected=" << obj.param.expected;
        return result.str();
    }
};

struct SelectInstructionParams {
    int64_t condition;
    int64_t trueValue;
    int64_t falseValue;
    int64_t expected;

    SelectInstructionParams(int64_t condition, int64_t trueValue, int64_t falseValue, int64_t expected)
            : condition(condition), trueValue(trueValue), falseValue(falseValue), expected(expected) {
    }
};

std::ostream& operator<<(std::ostream& os, const SelectInstructionParams& params) {
    return os << llvm::formatv("{ condition={0}, trueValue={1}, falseValue={2}, expected={3} }", params.condition,
                               params.trueValue, params.falseValue, params.expected)
                         .str();
}

class VirtualMachineSelectInstructionTest :
        public VMInstructionTest,
        public testing::WithParamInterface<SelectInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SelectInstructionParams>& obj) {
        std::ostringstream result;
        result << "condition=" << obj.param.condition << "_";
        result << "trueValue=" << obj.param.trueValue << "_";
        result << "falseValue=" << obj.param.falseValue << "_";
        result << "expected=" << obj.param.expected;
        return result.str();
    }
};

intel_npu::vm::Type getF32Type() {
    return intel_npu::vm::Type{
            intel_npu::vm::FloatType{sizeof(float) * CHAR_BIT, intel_npu::vm::FloatTypeFormat::IEEE754}};
}

struct ConvertInstructionParams {
    OpCode opcode;
    intel_npu::vm::Type srcType;
    intel_npu::vm::Type dstType;
    npu_vm_value src;
    npu_vm_value expected;
};

std::ostream& operator<<(std::ostream& os, const ConvertInstructionParams& p) {
    return os << "opcode=" << static_cast<uint16_t>(p.opcode);
}

class VirtualMachineConvertInstructionTest :
        public VMInstructionTest,
        public testing::WithParamInterface<ConvertInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertInstructionParams>& obj) {
        std::ostringstream result;
        result << "opcode_" << static_cast<uint16_t>(obj.param.opcode) << "_idx_" << obj.index;
        return result.str();
    }
};

ConvertInstructionParams intToInt(OpCode op, int64_t src, int64_t expected) {
    return {op, getInt64Type(), getInt64Type(), intel_npu::vm::makeI64(src), intel_npu::vm::makeI64(expected)};
}
ConvertInstructionParams intToF32(OpCode op, int64_t src, float expected) {
    return {op, getInt64Type(), getF32Type(), intel_npu::vm::makeI64(src), intel_npu::vm::makeF32(expected)};
}
ConvertInstructionParams intToF64(OpCode op, int64_t src, double expected) {
    return {op, getInt64Type(), getF64Type(), intel_npu::vm::makeI64(src), intel_npu::vm::makeF64(expected)};
}
ConvertInstructionParams f32ToInt(OpCode op, float src, int64_t expected) {
    return {op, getF32Type(), getInt64Type(), intel_npu::vm::makeF32(src), intel_npu::vm::makeI64(expected)};
}
ConvertInstructionParams f32ToF64(OpCode op, float src, double expected) {
    return {op, getF32Type(), getF64Type(), intel_npu::vm::makeF32(src), intel_npu::vm::makeF64(expected)};
}
ConvertInstructionParams f64ToInt(OpCode op, double src, int64_t expected) {
    return {op, getF64Type(), getInt64Type(), intel_npu::vm::makeF64(src), intel_npu::vm::makeI64(expected)};
}
ConvertInstructionParams f64ToF32(OpCode op, double src, float expected) {
    return {op, getF64Type(), getF32Type(), intel_npu::vm::makeF64(src), intel_npu::vm::makeF32(expected)};
}

}  // namespace

TEST_P(VirtualMachineTwoOperandInstructionTest, Execute) {
    const auto& params = GetParam();
    const bool isFp = std::holds_alternative<double>(params.expected);

    const auto paramType = getInt64Type();
    const auto resultType = isFp ? getF64Type() : getInt64Type();

    const auto funcName = "test";
    auto bytecode = BytecodeBuilder{}
                            .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/2,
                                                                          {paramType}, {resultType},
                                                                          /*isEntrypoint=*/true)
                                                 .instruction(params.opcode, {/*dst=*/0, /*src=*/1})
                                                 .retv(/*resultRegs=*/{0}))
                            .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, funcName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {intel_npu::vm::makeI64(params.src)};
    std::vector<npu_vm_value> results = {isFp ? intel_npu::vm::makeF64(0.0) : intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, funcName, args.size(), args.data(), results.size(), results.data()) ==
                NPU_VM_SUCCESS);

    if (std::holds_alternative<int64_t>(params.expected)) {
        EXPECT_EQ(results[0].i64, std::get<int64_t>(params.expected));
    } else if (std::holds_alternative<double>(params.expected)) {
        const auto expected = std::get<double>(params.expected);
        const auto actual = results[0].f64;
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual));
        } else if (std::isinf(expected)) {
            EXPECT_TRUE(std::isinf(actual));
            EXPECT_EQ(std::signbit(actual), std::signbit(expected));
        } else {
            EXPECT_DOUBLE_EQ(actual, expected);
        }
    } else {
        FAIL() << "Invalid expected value type";
    }
}

// Program layout (3 registers):
//   opcode   %0, %1, %2
//   retv     1, %0
TEST_P(VirtualMachineThreeOperandInstructionTest, Execute) {
    const auto& params = GetParam();
    const bool isFp = std::holds_alternative<double>(params.expected);

    const auto paramType = getInt64Type();
    const auto resultType = isFp ? getF64Type() : getInt64Type();

    const auto funcName = "test";
    auto bytecode = BytecodeBuilder{}
                            .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/3,
                                                                          {paramType, paramType}, {resultType},
                                                                          /*isEntrypoint=*/true)
                                                 .instruction(params.opcode, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                                                 .retv(/*resultRegs=*/{0}))
                            .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, funcName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {intel_npu::vm::makeI64(params.lhs), intel_npu::vm::makeI64(params.rhs)};
    std::vector<npu_vm_value> results = {isFp ? intel_npu::vm::makeF64(0.0) : intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, funcName, args.size(), args.data(), results.size(), results.data()) ==
                NPU_VM_SUCCESS);

    if (std::holds_alternative<int64_t>(params.expected)) {
        EXPECT_EQ(results[0].i64, std::get<int64_t>(params.expected));
    } else if (std::holds_alternative<double>(params.expected)) {
        const auto expected = std::get<double>(params.expected);
        const auto actual = results[0].f64;
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual));
        } else if (std::isinf(expected)) {
            EXPECT_TRUE(std::isinf(actual));
            EXPECT_EQ(std::signbit(actual), std::signbit(expected));
        } else {
            EXPECT_DOUBLE_EQ(actual, expected);
        }
    } else {
        FAIL() << "Invalid expected value type";
    }
}

// Registers (3 total: 2 value + 1 param):
//   firstParamRegIndex = numRegisters - numParams = 3 - 1 = 2
//   0=rZero  1=rResult  2=rA (input param, last reg)
// Byte layout:
//   [ 0] set_imm rZero, 0      (12B)
//   [12] jne 36, rA, rZero     (14B) if rA != 0: jump to [48]
//   [26] set_imm rResult, 1    (12B) not-taken (rA == 0)
//   [38] jmp 22                (10B) -> [60] (retv)
//   [48] set_imm rResult, 2    (12B) taken (rA != 0)
//   [60] retv rResult           (6B)
TEST_P(VirtualMachineJmpTest, SelectJmp) {
    constexpr int16_t rZero = 0;
    constexpr int16_t rResult = 1;
    constexpr int16_t rA = 2;  // input param: firstParamRegIndex = numRegs(3) - numParams(1) = 2

    const auto funcName = "jmp_select_test";
    const auto bytecode = BytecodeBuilder{}
                                  .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/3,
                                                                                {getInt64Type()}, {getInt64Type()},
                                                                                /*isEntrypoint=*/true)
                                                       .setImm(rZero, 0)
                                                       .jne(36, rA, rZero)
                                                       .setImm(rResult, 1)
                                                       .jmp(22)
                                                       .setImm(rResult, 2)
                                                       .retv({rResult}))
                                  .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, funcName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(GetParam().input)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, funcName, parameters.size(), parameters.data(), results.size(),
                                         results.data()) == NPU_VM_SUCCESS);
    EXPECT_EQ(results[0].i64, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(AddI64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, 0, 0, int64_t{0}},
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, 1, 1, int64_t{2}},
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, 10, 32, int64_t{42}},
                                 // Mixed signs
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, -5, 5, int64_t{0}},
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, 100, -1, int64_t{99}},
                                 // Both negative
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, -10, -20, int64_t{-30}},
                                 // Identity: adding zero
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, std::numeric_limits<int64_t>::max(), 0,
                                                               std::numeric_limits<int64_t>::max()},
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, std::numeric_limits<int64_t>::min(), 0,
                                                               std::numeric_limits<int64_t>::min()},
                                 // Signed overflow wraps to INT64_MIN (two's complement)
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, std::numeric_limits<int64_t>::max(), 1,
                                                               std::numeric_limits<int64_t>::min()}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SubI64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, 0, 0, int64_t{0}},
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, 1, 1, int64_t{0}},
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, 32, 10, int64_t{22}},
                                 // Mixed signs
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, -5, 5, int64_t{-10}},
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, 100, -1, int64_t{101}},
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, -10, -20, int64_t{10}},
                                 // Identity: subtracting zero
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, std::numeric_limits<int64_t>::max(), 0,
                                                               std::numeric_limits<int64_t>::max()},
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, std::numeric_limits<int64_t>::min(), 0,
                                                               std::numeric_limits<int64_t>::min()},
                                 // Signed overflow wraps to INT64_MAX (two's complement)
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, std::numeric_limits<int64_t>::min(), 1,
                                                               std::numeric_limits<int64_t>::max()}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        MulI64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::MUL_I64, 0, 0, int64_t{0}},
                ThreeOperandInstructionParams{OpCode::MUL_I64, 12345, 0, int64_t{0}},
                ThreeOperandInstructionParams{OpCode::MUL_I64, -67890, 0, int64_t{0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::MUL_I64, 2, 3, int64_t{6}},
                ThreeOperandInstructionParams{OpCode::MUL_I64, 10, 32, int64_t{320}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::MUL_I64, -5, 5, int64_t{-25}},
                ThreeOperandInstructionParams{OpCode::MUL_I64, 100, -1, int64_t{-100}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::MUL_I64, -10, -20, int64_t{200}},
                // Identity: multiplying by one
                ThreeOperandInstructionParams{OpCode::MUL_I64, std::numeric_limits<int64_t>::max(), 1,
                                              std::numeric_limits<int64_t>::max()},
                ThreeOperandInstructionParams{OpCode::MUL_I64, std::numeric_limits<int64_t>::min(), 1,
                                              std::numeric_limits<int64_t>::min()},
                // Signed overflow wraps (two's complement)
                ThreeOperandInstructionParams{OpCode::MUL_I64, std::numeric_limits<int64_t>::max(), 2,
                                              static_cast<int64_t>(-2)},  // INT64_MAX * 2 -> -2 in two's complement
                ThreeOperandInstructionParams{OpCode::MUL_I64, std::numeric_limits<int64_t>::min(), 2,
                                              static_cast<int64_t>(0)}  // INT64_MIN * 2 -> 0 in two's complement
                ),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DivI64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Basic positive
                ThreeOperandInstructionParams{OpCode::DIV_I64, 6, 3, int64_t{2}},
                ThreeOperandInstructionParams{OpCode::DIV_I64, 10, 32, int64_t{0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::DIV_I64, -25, 5, int64_t{-5}},
                ThreeOperandInstructionParams{OpCode::DIV_I64, 100, -1, int64_t{-100}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::DIV_I64, -200, -10, int64_t{20}},
                // Identity: dividing by one
                ThreeOperandInstructionParams{OpCode::DIV_I64, std::numeric_limits<int64_t>::max(), 1,
                                              std::numeric_limits<int64_t>::max()},
                ThreeOperandInstructionParams{OpCode::DIV_I64, std::numeric_limits<int64_t>::min(), 1,
                                              std::numeric_limits<int64_t>::min()},
                // Division overflow: INT64_MIN / -1 -> INT64_MIN (defined behavior to avoid undefined behavior)
                ThreeOperandInstructionParams{OpCode::DIV_I64, std::numeric_limits<int64_t>::min(), -1,
                                              std::numeric_limits<int64_t>::min()}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        RemI64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Basic positive
                ThreeOperandInstructionParams{OpCode::REM_I64, 5, 3, int64_t{2}},
                ThreeOperandInstructionParams{OpCode::REM_I64, 10, 32, int64_t{10}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::REM_I64, -25, 5, int64_t{0}},
                ThreeOperandInstructionParams{OpCode::REM_I64, 100, -1, int64_t{0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::REM_I64, -200, -10, int64_t{0}},
                // Identity: remainder by one is always zero
                ThreeOperandInstructionParams{OpCode::REM_I64, std::numeric_limits<int64_t>::max(), 1, int64_t{0}},
                ThreeOperandInstructionParams{OpCode::REM_I64, std::numeric_limits<int64_t>::min(), 1, int64_t{0}},
                // Division overflow: INT64_MIN / -1 -> INT64_MIN -> remainder is 0 (defined behavior to avoid undefined
                // behavior)
                ThreeOperandInstructionParams{OpCode::REM_I64, std::numeric_limits<int64_t>::min(), -1, int64_t{0}}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DivU64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Basic positive
                ThreeOperandInstructionParams{OpCode::DIV_U64, 6, 3, int64_t{2}},
                ThreeOperandInstructionParams{OpCode::DIV_U64, 10, 32, int64_t{0}},
                // Large unsigned values (high bit set, interpreted as large positive when unsigned)
                ThreeOperandInstructionParams{OpCode::DIV_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 2,
                                              static_cast<int64_t>(0x7FFFFFFFFFFFFFFFULL)},
                // UINT64_MAX / 1 = UINT64_MAX
                ThreeOperandInstructionParams{OpCode::DIV_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 1,
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)},
                // UINT64_MAX / UINT64_MAX = 1
                ThreeOperandInstructionParams{OpCode::DIV_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL),
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), int64_t{1}},
                // Value with high bit set divided by large divisor
                ThreeOperandInstructionParams{OpCode::DIV_U64, static_cast<int64_t>(0x8000000000000000ULL), 2,
                                              static_cast<int64_t>(0x4000000000000000ULL)},
                // Identity: dividing by one
                ThreeOperandInstructionParams{OpCode::DIV_U64, 100, 1, int64_t{100}}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        MinU64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::MIN_U64, 0, 0, int64_t{0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::MIN_U64, 1, 2, int64_t{1}},
                ThreeOperandInstructionParams{OpCode::MIN_U64, 100, 50, int64_t{50}},
                // Large unsigned values (high bit set, interpreted as large positive when unsigned)
                ThreeOperandInstructionParams{OpCode::MIN_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 1,
                                              int64_t{1}},
                // Both large unsigned values
                ThreeOperandInstructionParams{OpCode::MIN_U64, static_cast<int64_t>(0x8000000000000000ULL),
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL),
                                              static_cast<int64_t>(0x8000000000000000ULL)},
                // UINT64_MAX vs UINT64_MAX
                ThreeOperandInstructionParams{OpCode::MIN_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL),
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL),
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)},
                // Signed negative is large unsigned
                ThreeOperandInstructionParams{OpCode::MIN_U64, -1, 5, int64_t{5}},
                // Identity
                ThreeOperandInstructionParams{OpCode::MIN_U64, 42, 42, int64_t{42}}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(AddU64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::ADD_U64, 3, 4, int64_t{7}},
                                 ThreeOperandInstructionParams{OpCode::ADD_U64, 0, 0, int64_t{0}},
                                 // Large unsigned values
                                 ThreeOperandInstructionParams{
                                         OpCode::ADD_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 1, int64_t{0}},
                                 // Wrapping overflow
                                 ThreeOperandInstructionParams{OpCode::ADD_U64,
                                                               static_cast<int64_t>(0x8000000000000000ULL), 1,
                                                               static_cast<int64_t>(0x8000000000000001ULL)},
                                 // Identity: adding zero
                                 ThreeOperandInstructionParams{OpCode::ADD_U64, 100, 0, int64_t{100}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        MaxU64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::MAX_U64, 0, 0, int64_t{0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::MAX_U64, 1, 2, int64_t{2}},
                ThreeOperandInstructionParams{OpCode::MAX_U64, 100, 50, int64_t{100}},
                // Large unsigned values (high bit set, interpreted as large positive when unsigned)
                ThreeOperandInstructionParams{OpCode::MAX_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 1,
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)},
                // Both large unsigned values
                ThreeOperandInstructionParams{OpCode::MAX_U64, static_cast<int64_t>(0x8000000000000000ULL),
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL),
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)},
                // Signed negative is large unsigned
                ThreeOperandInstructionParams{OpCode::MAX_U64, -1, 5, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)},
                // Identity
                ThreeOperandInstructionParams{OpCode::MAX_U64, 42, 42, int64_t{42}}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MulU64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::MUL_U64, 3, 4, int64_t{12}},
                                 ThreeOperandInstructionParams{OpCode::MUL_U64, 0, 100, int64_t{0}},
                                 // Identity: multiply by one
                                 ThreeOperandInstructionParams{OpCode::MUL_U64, 100, 1, int64_t{100}},
                                 // Large unsigned values with wrapping
                                 ThreeOperandInstructionParams{
                                         OpCode::MUL_U64, static_cast<int64_t>(0x8000000000000000ULL), 2, int64_t{0}},
                                 // UINT64_MAX * 1 = UINT64_MAX
                                 ThreeOperandInstructionParams{OpCode::MUL_U64,
                                                               static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 1,
                                                               static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(RemU64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::REM_U64, 7, 3, int64_t{1}},
                                 ThreeOperandInstructionParams{OpCode::REM_U64, 10, 5, int64_t{0}},
                                 // Large unsigned values
                                 ThreeOperandInstructionParams{
                                         OpCode::REM_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 2, int64_t{1}},
                                 // Modulo by 1 = 0
                                 ThreeOperandInstructionParams{OpCode::REM_U64, 100, 1, int64_t{0}},
                                 // Value with high bit set
                                 ThreeOperandInstructionParams{
                                         OpCode::REM_U64, static_cast<int64_t>(0x8000000000000000ULL), 3, int64_t{2}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        SubU64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Basic positive
                ThreeOperandInstructionParams{OpCode::SUB_U64, 10, 3, int64_t{7}},
                ThreeOperandInstructionParams{OpCode::SUB_U64, 0, 0, int64_t{0}},
                // Wrapping underflow (0 - 1 = UINT64_MAX)
                ThreeOperandInstructionParams{OpCode::SUB_U64, 0, 1, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)},
                // Large unsigned values
                ThreeOperandInstructionParams{OpCode::SUB_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 1,
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFEULL)},
                // Identity: subtracting zero
                ThreeOperandInstructionParams{OpCode::SUB_U64, 100, 0, int64_t{100}}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MinI64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, 0, 0, int64_t{0}},
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, 1, 2, int64_t{1}},
                                 // Basic negative
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, -10, -20, int64_t{-20}},
                                 // Mixed signs
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, -5, 5, int64_t{-5}},
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, 100, -1, int64_t{-1}},
                                 // Infinity
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, std::numeric_limits<int64_t>::min(), 1,
                                                               std::numeric_limits<int64_t>::min()},
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, std::numeric_limits<int64_t>::max(), 1,
                                                               int64_t{1}},
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, std::numeric_limits<int64_t>::max(),
                                                               std::numeric_limits<int64_t>::min(),
                                                               std::numeric_limits<int64_t>::min()}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MaxI64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, 0, 0, int64_t{0}},
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, 1, 2, int64_t{2}},
                                 // Basic negative
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, -10, -20, int64_t{-10}},
                                 // Mixed signs
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, -5, 5, int64_t{5}},
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, 100, -1, int64_t{100}},
                                 // Infinity
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, std::numeric_limits<int64_t>::min(), 1,
                                                               int64_t{1}},
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, std::numeric_limits<int64_t>::max(), 1,
                                                               std::numeric_limits<int64_t>::max()},
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, std::numeric_limits<int64_t>::max(),
                                                               std::numeric_limits<int64_t>::min(),
                                                               std::numeric_limits<int64_t>::max()}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        AbsI64, VirtualMachineTwoOperandInstructionTest,
        testing::Values(TwoOperandInstructionParams{OpCode::ABS_I64, 0, int64_t{0}},
                        TwoOperandInstructionParams{OpCode::ABS_I64, 12345, int64_t{12345}},
                        TwoOperandInstructionParams{OpCode::ABS_I64, -67890, int64_t{67890}},
                        // Identity: abs of INT64_MIN is INT64_MIN (defined behavior to avoid undefined behavior)
                        TwoOperandInstructionParams{OpCode::ABS_I64, std::numeric_limits<int64_t>::min(),
                                                    std::numeric_limits<int64_t>::min()}),
        VirtualMachineTwoOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(And64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(ThreeOperandInstructionParams{OpCode::AND_64, 0b1100, 0b1010, int64_t{0b1000}},
                                         ThreeOperandInstructionParams{OpCode::AND_64, 0b1111, 0b0000, int64_t{0}},
                                         ThreeOperandInstructionParams{OpCode::AND_64, 0b1010, 0b0101, int64_t{0}},
                                         ThreeOperandInstructionParams{OpCode::AND_64, 0b1111'1111, 0b1010'1010,
                                                                       int64_t{0b1010'1010}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Not64, VirtualMachineTwoOperandInstructionTest,
                         testing::Values(TwoOperandInstructionParams{OpCode::NOT_64, 0b1100, int64_t{~0b1100}},
                                         TwoOperandInstructionParams{OpCode::NOT_64, 0b1010, int64_t{~0b1010}},
                                         TwoOperandInstructionParams{OpCode::NOT_64, 0b1111'0000,
                                                                     int64_t{~0b1111'0000}}),
                         VirtualMachineTwoOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Or64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(ThreeOperandInstructionParams{OpCode::OR_64, 0b1100, 0b1010, int64_t{0b1110}},
                                         ThreeOperandInstructionParams{OpCode::OR_64, 0b1111, 0b0000, int64_t{0b1111}},
                                         ThreeOperandInstructionParams{OpCode::OR_64, 0b1010, 0b0101, int64_t{0b1111}},
                                         ThreeOperandInstructionParams{OpCode::OR_64, 0b1111'1111, 0b1010'1010,
                                                                       int64_t{0b1111'1111}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Xor64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(ThreeOperandInstructionParams{OpCode::XOR_64, 0b1100, 0b1010, int64_t{0b0110}},
                                         ThreeOperandInstructionParams{OpCode::XOR_64, 0b1111, 0b0000, int64_t{0b1111}},
                                         ThreeOperandInstructionParams{OpCode::XOR_64, 0b1010, 0b0101, int64_t{0b1111}},
                                         ThreeOperandInstructionParams{OpCode::XOR_64, 0b1111'1111, 0b1010'1010,
                                                                       int64_t{0b0101'0101}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        Sll64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(ThreeOperandInstructionParams{OpCode::SLL_64, 1, 1, int64_t{2}},
                        ThreeOperandInstructionParams{OpCode::SLL_64, 1, 63, int64_t{int64_t{1} << 63}},
                        ThreeOperandInstructionParams{OpCode::SLL_64, 1, 64, int64_t{1}},  // shift count masked by 63
                        ThreeOperandInstructionParams{OpCode::SLL_64, -1, 1, int64_t{-2}}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        Srl64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(ThreeOperandInstructionParams{OpCode::SRL_64, 2, 1, int64_t{1}},
                        ThreeOperandInstructionParams{OpCode::SRL_64, int64_t{1} << 63, 63, int64_t{1}},
                        ThreeOperandInstructionParams{OpCode::SRL_64, int64_t{1} << 63, 64,
                                                      int64_t{1} << 63},  // shift count masked by 63
                        ThreeOperandInstructionParams{OpCode::SRL_64, -2, 1, static_cast<int64_t>(UINT64_MAX >> 1)}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Sra64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(ThreeOperandInstructionParams{OpCode::SRA_64, 2, 1, int64_t{1}},
                                         ThreeOperandInstructionParams{OpCode::SRA_64, int64_t{1} << 63, 63,
                                                                       int64_t{-1}},
                                         ThreeOperandInstructionParams{OpCode::SRA_64, int64_t{1} << 63, 64,
                                                                       int64_t{1} << 63},  // shift count masked by 63
                                         ThreeOperandInstructionParams{OpCode::SRA_64, -2, 1, int64_t{-1}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

// Program layout (3 registers):
// func(%0 = opcode(%1, %2)):
//   opcode   %0, %1, %2
// Entrypoint that calls func:
//   call     0, 1, {dst_regs}, 2, {arg_regs} (function that contains an arith op [like Add])
//   retv     1, %0
TEST_P(VirtualMachineCallInstructionTest, Execute) {
    const auto& params = GetParam();

    auto builder = BytecodeBuilder{}.addFunction(BytecodeBuilder::FunctionBuilder("callee", /*numGeneralRegisters=*/3,
                                                                                  {getInt64Type(), getInt64Type()},
                                                                                  {getInt64Type()})
                                                         .instruction(params.opcode, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                                                         .retv(/*resultRegs=*/{0}));
    const auto calleeIndex = builder.getFunctionIndex("callee");
    ASSERT_TRUE(calleeIndex.has_value());

    const auto entryFuncName = "entry";
    const auto functionIndexReg = 0;
    const auto bytecode =
            builder.addFunction(
                           BytecodeBuilder::FunctionBuilder(entryFuncName, /*numGeneralRegisters=*/3,
                                                            {getInt64Type(), getInt64Type()}, {getInt64Type()},
                                                            /*isEntrypoint=*/true)
                                   .setImm(/*dst=*/functionIndexReg, /*imm=*/static_cast<int64_t>(calleeIndex.value()))
                                   .call(functionIndexReg, /*dstRegs=*/{0}, /*argRegs=*/{1, 2})
                                   .retv(/*resultRegs=*/{0}))
                    .build();

    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, entryFuncName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(params.lhs), intel_npu::vm::makeI64(params.rhs)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, entryFuncName, parameters.size(), parameters.data(), results.size(),
                                         results.data()) == NPU_VM_SUCCESS);
    EXPECT_EQ(results[0].i64, std::get<int64_t>(params.expected));
}

// Program layout (4 registers):
// func:
//   opcode   %0, %1, %2
// Entrypoint that calls func:
//   set.imm  1, callee_idx
//   call     1, 1, {dst_regs}, 2, {arg_regs} (function that contains an arith op [like Add])
//   call     1, 1, {dst_regs}, 2, {arg_regs} (function that contains an arith op [like Add] - result of the previous
//   add and second attr: add(dst, %1) to test caller state across multiple calls) retv 1, %0
TEST_P(VirtualMachineCallInstructionTest, ExecuteTwice) {
    const auto& params = GetParam();

    auto builder = BytecodeBuilder{}.addFunction(BytecodeBuilder::FunctionBuilder("callee", /*numGeneralRegisters=*/3,
                                                                                  {getInt64Type(), getInt64Type()},
                                                                                  {getInt64Type()})
                                                         .instruction(params.opcode, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                                                         .retv(/*resultRegs=*/{0}));
    const auto calleeIndex = builder.getFunctionIndex("callee");
    ASSERT_TRUE(calleeIndex.has_value());

    const auto entryFuncName = "entry_twice";
    const auto functionIndexReg = 1;
    const auto bytecode =
            builder.addFunction(
                           BytecodeBuilder::FunctionBuilder(entryFuncName, /*numGeneralRegisters=*/4,
                                                            {getInt64Type(), getInt64Type()}, {getInt64Type()},
                                                            /*isEntrypoint=*/true)
                                   .setImm(/*dst=*/functionIndexReg, /*imm=*/static_cast<int64_t>(calleeIndex.value()))
                                   .call(functionIndexReg, /*dstRegs=*/{0}, /*argRegs=*/{2, 3})
                                   .call(functionIndexReg, /*dstRegs=*/{0}, /*argRegs=*/{0, 2})
                                   .retv(/*resultRegs=*/{0}))
                    .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, entryFuncName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(params.lhs), intel_npu::vm::makeI64(params.rhs)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, entryFuncName, parameters.size(), parameters.data(), results.size(),
                                         results.data()) == NPU_VM_SUCCESS);
    EXPECT_EQ(results[0].i64, std::get<int64_t>(params.expected) + params.lhs);
}

// Program layout (4 registers):
// func_add:
//   opcode   %0, %1, %2
// func_mul:
//   opcode   %0, %1, %2
// Entrypoint that calls func:
//   set.imm  1, add_callee_idx
//   call     1, 1, {dst_regs}, 2, {arg_regs} (function that contains an arith op [like Add])
//   set.imm  1, mul_callee_idx
//   call     1, 1, {dst_regs}, 2, {arg_regs} (function that contains an arith op [like Mul] - result of the previous
//   add and second attr: mul(dst, %1) to test caller state across multiple calls) retv 1, %0
TEST_P(VirtualMachineCallInstructionTest, ExecuteTwiceDifferentFunctions) {
    const auto& params = GetParam();

    auto builder = BytecodeBuilder{}
                           .addFunction(BytecodeBuilder::FunctionBuilder("callee", /*numGeneralRegisters=*/3,
                                                                         {getInt64Type(), getInt64Type()},
                                                                         {getInt64Type()}, /*isEntrypoint=*/false)
                                                .instruction(params.opcode, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                                                .retv(/*resultRegs=*/{0}))
                           .addFunction(BytecodeBuilder::FunctionBuilder("callee2", /*numGeneralRegisters=*/3,
                                                                         {getInt64Type(), getInt64Type()},
                                                                         {getInt64Type()}, /*isEntrypoint=*/false)
                                                .instruction(OpCode::MUL_I64, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                                                .retv(/*resultRegs=*/{0}));
    const auto firstCalleeIndex = builder.getFunctionIndex("callee");
    ASSERT_TRUE(firstCalleeIndex.has_value());
    const auto secondCalleeIndex = builder.getFunctionIndex("callee2");
    ASSERT_TRUE(secondCalleeIndex.has_value());

    const auto entryFuncName = "entry_twice_different_callees";
    const auto functionIndexReg = 1;
    const auto bytecode = builder.addFunction(BytecodeBuilder::FunctionBuilder(entryFuncName, /*numGeneralRegisters=*/4,
                                                                               {getInt64Type(), getInt64Type()},
                                                                               {getInt64Type()}, /*isEntrypoint=*/true)
                                                      .setImm(/*dst=*/functionIndexReg,
                                                              /*imm=*/static_cast<int64_t>(firstCalleeIndex.value()))
                                                      .call(functionIndexReg, /*dstRegs=*/{0}, /*argRegs=*/{2, 3})
                                                      .setImm(/*dst=*/functionIndexReg,
                                                              /*imm=*/static_cast<int64_t>(secondCalleeIndex.value()))
                                                      .call(functionIndexReg, /*dstRegs=*/{0}, /*argRegs=*/{0, 2})
                                                      .retv(/*resultRegs=*/{0}))
                                  .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, entryFuncName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(params.lhs), intel_npu::vm::makeI64(params.rhs)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, entryFuncName, parameters.size(), parameters.data(), results.size(),
                                         results.data()) == NPU_VM_SUCCESS);
    EXPECT_EQ(results[0].i64, std::get<int64_t>(params.expected) * params.lhs);
}

INSTANTIATE_TEST_SUITE_P(CallOpTests, VirtualMachineCallInstructionTest,
                         testing::Values(
                                 // Zero
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, 0, 0, int64_t{0}},
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, 1, 1, int64_t{2}},
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, 10, 32, int64_t{42}},
                                 // Mixed signs
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, -5, 5, int64_t{0}},
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, 100, -1, int64_t{99}},
                                 // Both negative
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, -10, -20, int64_t{-30}}),
                         VirtualMachineCallInstructionTest::getTestCaseName);

TEST_F(VirtualMachineCallInstructionTest, CallChainThreeLevelsDeepPropagatesResult) {
    auto builder = BytecodeBuilder{}.addFunction(
            BytecodeBuilder::FunctionBuilder("inner", /*numGeneralRegisters=*/3, {getInt64Type(), getInt64Type()},
                                             {getInt64Type()})
                    .instruction(OpCode::ADD_I64, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                    .retv(/*resultRegs=*/{0}));
    const auto innerIndex = builder.getFunctionIndex("inner");
    ASSERT_TRUE(innerIndex.has_value());

    // middle has 4 registers: params occupy regs [2, 3], general regs are [0, 1]. Reg 1 holds the callee index
    builder.addFunction(BytecodeBuilder::FunctionBuilder("middle", /*numGeneralRegisters=*/4,
                                                         {getInt64Type(), getInt64Type()}, {getInt64Type()})
                                .setImm(/*dst=*/1, /*imm=*/static_cast<int64_t>(innerIndex.value()))
                                .call(/*functionIndexReg=*/1, /*dstRegs=*/{0}, /*argRegs=*/{2, 3})
                                .retv(/*resultRegs=*/{0}));
    const auto middleIndex = builder.getFunctionIndex("middle");
    ASSERT_TRUE(middleIndex.has_value());

    const auto entryFuncName = "entry";
    const auto bytecode =
            builder.addFunction(BytecodeBuilder::FunctionBuilder(entryFuncName, /*numGeneralRegisters=*/4,
                                                                 {getInt64Type(), getInt64Type()}, {getInt64Type()},
                                                                 /*isEntrypoint=*/true)
                                        .setImm(/*dst=*/1, /*imm=*/static_cast<int64_t>(middleIndex.value()))
                                        .call(/*functionIndexReg=*/1, /*dstRegs=*/{0}, /*argRegs=*/{2, 3})
                                        .retv(/*resultRegs=*/{0}))
                    .build();

    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(20), intel_npu::vm::makeI64(22)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, entryFuncName, parameters.size(), parameters.data(), results.size(),
                                         results.data()) == NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, 42);
}

TEST_F(VirtualMachineCallInstructionTest, RetvRoutesMultipleResultsToCallerDestinationRegisters) {
    // swap has 2 registers, both parameters [0, 1]; retv returns them reversed
    auto builder = BytecodeBuilder{}.addFunction(BytecodeBuilder::FunctionBuilder("swap", /*numGeneralRegisters=*/2,
                                                                                  {getInt64Type(), getInt64Type()},
                                                                                  {getInt64Type(), getInt64Type()})
                                                         .retv(/*resultRegs=*/{1, 0}));
    const auto swapIndex = builder.getFunctionIndex("swap");
    ASSERT_TRUE(swapIndex.has_value());

    // entry has 5 registers: params occupy regs [3, 4], general regs are [0, 1, 2]. Reg 2 holds the callee index
    const auto entryFuncName = "entry";
    const auto bytecode =
            builder.addFunction(BytecodeBuilder::FunctionBuilder(entryFuncName, /*numGeneralRegisters=*/5,
                                                                 {getInt64Type(), getInt64Type()},
                                                                 {getInt64Type(), getInt64Type()},
                                                                 /*isEntrypoint=*/true)
                                        .setImm(/*dst=*/2, /*imm=*/static_cast<int64_t>(swapIndex.value()))
                                        .call(/*functionIndexReg=*/2, /*dstRegs=*/{0, 1}, /*argRegs=*/{3, 4})
                                        .retv(/*resultRegs=*/{0, 1}))
                    .build();

    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(7), intel_npu::vm::makeI64(9)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0), intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, entryFuncName, parameters.size(), parameters.data(), results.size(),
                                         results.data()) == NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, 9);
    EXPECT_EQ(results.at(1).i64, 7);
}

TEST_F(VirtualMachineCallInstructionTest, SelfRecursiveFunctionComputesTriangularNumber) {
    // The `sum` function computes sum(n) = n + sum(n - 1) with base case sum(0) = 0, pushing one frame per recursion
    // level.
    // Byte layout (offsets are PC-relative from the function body start):
    //   [ 0] set_imm rZero, 0                       (12B)
    //   [12] je 72, rN, rZero        -> base case   (14B)
    //   [26] set_imm rOne, 1                        (12B)
    //   [38] sub.i64 rRec, rN, rOne                  (8B)
    //   [46] set_imm rFunc, sumFnIndex              (12B)
    //   [58] call rFunc, {rRec}, {rRec}             (12B)
    //   [70] add.i64 rResult, rN, rRec               (8B)
    //   [78] retv {rResult}                          (6B)
    //   [84] set_imm rResult, 0  (base case)        (12B)
    //   [96] retv {rResult}                          (6B)

    constexpr int64_t sumFnIndex = 0;       // The index of the sum function within the function section
    constexpr int64_t jumpToBaseCase = 72;  // The relative offset to the base case
    constexpr int64_t numRegs = 6;  // Total number of registers used by the function (general: [0-4], params: [5])

    constexpr int16_t rResult = 0;  // The result register
    constexpr int16_t rFunc = 1;    // The register to hold the function index for the recursive call
    constexpr int16_t rZero = 2;    // The register to hold the constant zero for the base case comparison
    constexpr int16_t rOne = 3;     // The register to hold the constant one for decrementing n in the recursive case
    constexpr int16_t rRec = 4;     // The register to hold the result of the recursive call sum(n - 1)
    constexpr int16_t rN = 5;       // The register to hold the input parameter n

    const auto funcName = "sum";
    auto builder = BytecodeBuilder{};
    builder.addFunction(BytecodeBuilder::FunctionBuilder(funcName, numRegs, {getInt64Type()}, {getInt64Type()},
                                                         /*isEntrypoint=*/true)
                                .setImm(rZero, 0)
                                .je(jumpToBaseCase, rN, rZero)
                                .setImm(rOne, 1)
                                .instruction(OpCode::SUB_I64, {rRec, rN, rOne})
                                .setImm(rFunc, sumFnIndex)
                                .call(/*functionIndexReg=*/rFunc, /*dstRegs=*/{rRec}, /*argRegs=*/{rRec})
                                .instruction(OpCode::ADD_I64, {rResult, rN, rRec})
                                .retv(/*resultRegs=*/{rResult})
                                .setImm(rResult, 0)
                                .retv(/*resultRegs=*/{rResult}));
    const auto resolvedIndex = builder.getFunctionIndex(funcName);
    ASSERT_TRUE(resolvedIndex.has_value());
    ASSERT_EQ(resolvedIndex.value(), static_cast<uint64_t>(sumFnIndex));

    const auto bytecode = builder.build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    struct RecursionCase {
        int64_t n;
        int64_t expected;
    };
    for (const auto& testCase : {RecursionCase{0, 0}, RecursionCase{1, 1}, RecursionCase{5, 15}, RecursionCase{10, 55},
                                 RecursionCase{1000, 500500}}) {
        std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(testCase.n)};
        std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(-1)};
        ASSERT_EQ(npu_vm_call_with_results(engine, funcName, parameters.size(), parameters.data(), results.size(),
                                           results.data()),
                  NPU_VM_SUCCESS)
                << "n=" << testCase.n;
        EXPECT_EQ(results.at(0).i64, testCase.expected) << "n=" << testCase.n;
        ASSERT_EQ(npu_vm_reset_state(engine, /*resetExecutionContext=*/false), NPU_VM_SUCCESS);
    }

    const auto overMaxDepthTestCase = 1001;
    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(overMaxDepthTestCase)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(-1)};
    ASSERT_NE(npu_vm_call_with_results(engine, funcName, parameters.size(), parameters.data(), results.size(),
                                       results.data()),
              NPU_VM_SUCCESS)
            << "n=" << overMaxDepthTestCase;
}

INSTANTIATE_TEST_SUITE_P(JmpSelect, VirtualMachineJmpTest,
                         testing::Values(JmpSelectTestParams{0, 1}, JmpSelectTestParams{1, 2}),
                         VirtualMachineJmpTest::getTestCaseName);

// Registers (3 total: 2 value + 1 param):
//   firstParamRegIndex = numRegisters - numParams = 3 - 1 = 2
//   0=rZero  1=rResult  2=rA (input param, last reg)
// Byte layout:
//   [ 0] set_imm rZero, 0      (12B)
//   [12] je 36, rA, rZero      (14B) if rA == 0: jump to [48]
//   [26] set_imm rResult, 2    (12B) not-taken (rA != 0)
//   [38] jmp 22                (10B) -> [60] (retv)
//   [48] set_imm rResult, 1    (12B) taken (rA == 0)
//   [60] retv rResult           (6B)
class VirtualMachineJETest : public VMInstructionTest, public testing::WithParamInterface<JmpSelectTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<JmpSelectTestParams>& obj) {
        std::ostringstream result;
        result << "input=" << obj.param.input << "_expected=" << obj.param.expected;
        return result.str();
    }
};

class VirtualMachineJNETest : public VMInstructionTest, public testing::WithParamInterface<JmpSelectTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<JmpSelectTestParams>& obj) {
        std::ostringstream result;
        result << "input=" << obj.param.input << "_expected=" << obj.param.expected;
        return result.str();
    }
};

TEST_P(VirtualMachineJETest, ConditionalJump) {
    constexpr int16_t rZero = 0;
    constexpr int16_t rResult = 1;
    constexpr int16_t rA = 2;  // input param: firstParamRegIndex = numRegs(3) - numParams(1) = 2

    const auto funcName = "je_test";
    const auto bytecode = BytecodeBuilder{}
                                  .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/3,
                                                                                {getInt64Type()}, {getInt64Type()},
                                                                                /*isEntrypoint=*/true)
                                                       .setImm(rZero, 0)
                                                       .je(36, rA, rZero)
                                                       .setImm(rResult, 2)
                                                       .jmp(22)
                                                       .setImm(rResult, 1)
                                                       .retv({rResult}))
                                  .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, funcName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(GetParam().input)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, funcName, parameters.size(), parameters.data(), results.size(),
                                         results.data()) == NPU_VM_SUCCESS);
    EXPECT_EQ(results[0].i64, GetParam().expected);
}

// Same layout as JE test, but jne instead of je.
// When rA != 0: jne taken → result = 1
// When rA == 0: jne not taken → result = 2
TEST_P(VirtualMachineJNETest, ConditionalJump) {
    constexpr int16_t rZero = 0;
    constexpr int16_t rResult = 1;
    constexpr int16_t rA = 2;  // input param: firstParamRegIndex = numRegs(3) - numParams(1) = 2

    const auto funcName = "jne_test";
    const auto bytecode = BytecodeBuilder{}
                                  .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/3,
                                                                                {getInt64Type()}, {getInt64Type()},
                                                                                /*isEntrypoint=*/true)
                                                       .setImm(rZero, 0)
                                                       .jne(36, rA, rZero)
                                                       .setImm(rResult, 2)
                                                       .jmp(22)
                                                       .setImm(rResult, 1)
                                                       .retv({rResult}))
                                  .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, funcName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(GetParam().input)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, funcName, parameters.size(), parameters.data(), results.size(),
                                         results.data()) == NPU_VM_SUCCESS);
    EXPECT_EQ(results[0].i64, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(JEConditionalJump, VirtualMachineJETest,
                         testing::Values(JmpSelectTestParams{0, 1}, JmpSelectTestParams{5, 2},
                                         JmpSelectTestParams{-1, 2}),
                         VirtualMachineJETest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(JNEConditionalJump, VirtualMachineJNETest,
                         testing::Values(JmpSelectTestParams{5, 1}, JmpSelectTestParams{-1, 1},
                                         JmpSelectTestParams{0, 2}),
                         VirtualMachineJNETest::getTestCaseName);

// Registers (4 total: 3 value + 1 param):
//   firstParamRegIndex = numRegisters - numParams = 4 - 1 = 3
//   0=rCounter  1=rOne  2=rZero  3=rA (input param, last reg)
// Byte layout:
//   [ 0] set     rCounter, rA              copy param into working register  ( 6B)
//   [ 6] set_imm rOne,  1                                                    (12B)
//   [18] set_imm rZero, 0                                                    (12B)
//   [30] sub_i64 rCounter, rCounter, rOne                                    ( 8B)
//   [38] jne -8, rCounter, rZero           if != 0: _pc += -8 → [30]        (14B)
//   [52] retv rCounter                     returns 0 when loop exits         ( 6B)
TEST_P(VirtualMachineJmpBackwardTest, BackwardJump) {
    constexpr int16_t rCounter = 0;
    constexpr int16_t rOne = 1;
    constexpr int16_t rZero = 2;
    constexpr int16_t rA = 3;  // input param: firstParamRegIndex = numRegs(4) - numParams(1) = 3

    const auto funcName = "jmp_backward_test";
    const auto bytecode = BytecodeBuilder{}
                                  .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/4,
                                                                                {getInt64Type()}, {getInt64Type()},
                                                                                /*isEntrypoint=*/true)
                                                       .instruction(OpCode::SET, {rCounter, rA})
                                                       .setImm(rOne, 1)
                                                       .setImm(rZero, 0)
                                                       .instruction(OpCode::SUB_I64, {rCounter, rCounter, rOne})
                                                       .jne(-8, rCounter, rZero)
                                                       .retv({rCounter}))
                                  .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, funcName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(GetParam().input)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, funcName, parameters.size(), parameters.data(), results.size(),
                                         results.data()) == NPU_VM_SUCCESS);
    EXPECT_EQ(results[0].i64, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(JmpBackward, VirtualMachineJmpBackwardTest,
                         testing::Values(JmpSelectTestParams{1, 0}, JmpSelectTestParams{3, 0},
                                         JmpSelectTestParams{5, 0}),
                         VirtualMachineJmpBackwardTest::getTestCaseName);

// Regression test: before the 64-bit offset fix, getRelativeOffset() used checked_cast<int16_t>
// which would abort on any jump offset > INT16_MAX (32767).
// Layout:
//   [     0] jmp 32788            (10B) — forward jump past all padding
//   [    10] set rResult, rResult × 5463 (32778B) — unreachable padding
//   [32788]  set_imm rResult, 1   (12B) — jump lands here
//   [32800]  retv {rResult}        (6B)
TEST_F(VMInstructionTest, JmpForwardOffsetExceedsInt16Range) {
    constexpr int16_t rResult = 0;
    const auto funcName = "large_jmp_test";

    constexpr int64_t padCount = 5463;  // 5463 * 6 = 32778 bytes > INT16_MAX
    constexpr int64_t setInstrSize = 6;
    constexpr int64_t jmpInstrSize = 10;
    const int64_t jmpOffset = jmpInstrSize + padCount * setInstrSize;  // 32788

    auto fb = BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/1, {}, {getInt64Type()},
                                               /*isEntrypoint=*/true);
    fb.jmp(jmpOffset);
    for (int64_t i = 0; i < padCount; ++i) {
        fb.instruction(OpCode::SET, {rResult, rResult});
    }
    fb.setImm(rResult, 1);
    fb.retv({rResult});

    const auto bytecode = BytecodeBuilder{}.addFunction(fb).build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, funcName, 0, nullptr, results.size(), results.data()) ==
                NPU_VM_SUCCESS);
    EXPECT_EQ(results[0].i64, 1);
}

struct JumpBoundsCheckParams {
    std::string testName;
    std::function<std::vector<uint8_t>()> buildBytecode;
};

class VirtualMachineJumpBoundsCheckTest :
        public VMInstructionTest,
        public testing::WithParamInterface<JumpBoundsCheckParams> {};

TEST_P(VirtualMachineJumpBoundsCheckTest, Execute) {
    auto p = GetParam();
    auto bytecode = p.buildBytecode();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);
    EXPECT_FALSE(npu_vm_call(engine, "test", /*num_arguments=*/0, /*arguments=*/nullptr) == NPU_VM_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
        JumpBoundsCheck, VirtualMachineJumpBoundsCheckTest,
        testing::Values(JumpBoundsCheckParams{"JmpPositiveOffsetBeyondEnd",
                                              [] {
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/0, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .jmp(10000)
                                                                               .ret())
                                                          .build();
                                              }},
                        JumpBoundsCheckParams{"JmpNegativeOffsetBeforeStart",
                                              [] {
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/0, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .jmp(-10000)
                                                                               .ret())
                                                          .build();
                                              }},
                        JumpBoundsCheckParams{"JmpZeroOffsetInfiniteLoopGuard",
                                              [] {
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/0, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .jmp(0)
                                                                               .ret())
                                                          .build();
                                              }},
                        JumpBoundsCheckParams{"JETakenOffsetBeyondEnd",
                                              [] {
                                                  constexpr int16_t r0 = 0;
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/1, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .je(10000, r0, r0)
                                                                               .ret())
                                                          .build();
                                              }},
                        JumpBoundsCheckParams{"JNETakenOffsetBeyondEnd",
                                              [] {
                                                  constexpr int16_t rZero = 0;
                                                  constexpr int16_t rOne = 1;
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/2, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .setImm(rOne, 1)
                                                                               .jne(10000, rZero, rOne)
                                                                               .ret())
                                                          .build();
                                              }}),
        [](const testing::TestParamInfo<JumpBoundsCheckParams>& info) {
            return info.param.testName;
        });

TEST_P(VirtualMachineAssertInstructionTest, Execute) {
    const auto& params = GetParam();

    const auto funcName = "assert_test";
    const auto bytecode =
            BytecodeBuilder{}
                    .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/2, {getInt64Type()},
                                                                  {getInt64Type()},
                                                                  /*isEntrypoint=*/true)
                                         .instruction(OpCode::ASSERT, {/*condition=*/1, params.messageIndex})
                                         .setImm(/*dst=*/0, /*imm=*/42)
                                         .retv({0}))
                    .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, funcName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(params.condition)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(-1)};
    const auto status = npu_vm_call_with_results(engine, funcName, parameters.size(), parameters.data(), results.size(),
                                                 results.data());

    EXPECT_EQ(status, params.expectedStatus);
    if (params.expectedResult.has_value()) {
        EXPECT_EQ(results[0].i64, *params.expectedResult);
    }
}

INSTANTIATE_TEST_SUITE_P(
        Assert, VirtualMachineAssertInstructionTest,
        testing::Values(AssertInstructionParams{/*condition=*/1, /*messageIndex=*/0, NPU_VM_SUCCESS,
                                                /*expectedResult=*/42},
                        AssertInstructionParams{/*condition=*/-1, /*messageIndex=*/0, NPU_VM_SUCCESS,
                                                /*expectedResult=*/42},
                        AssertInstructionParams{/*condition=*/0, /*messageIndex=*/0, NPU_VM_ERROR_FUNCTION_CALL_FAILED,
                                                /*expectedResult=*/std::nullopt},
                        AssertInstructionParams{/*condition=*/0, /*messageIndex=*/7, NPU_VM_ERROR_FUNCTION_CALL_FAILED,
                                                /*expectedResult=*/std::nullopt}),
        VirtualMachineAssertInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        AddF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(0.0), f64Bits(0.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-0.0), f64Bits(-0.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-0.0), f64Bits(0.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(1.0), f64Bits(0.0), double{1.0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(1.0), f64Bits(2.0), double{3.0}},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(1.5), f64Bits(2.5), double{4.0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-1.0), f64Bits(2.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-1.0), f64Bits(1.0), double{0.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-3.0), f64Bits(-4.0), double{-7.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(kInfF64), f64Bits(1.0), kInfF64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-kInfF64), f64Bits(1.0), -kInfF64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(kInfF64), f64Bits(kInfF64), kInfF64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-kInfF64), f64Bits(-kInfF64), -kInfF64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(kInfF64), f64Bits(-kInfF64), kNaNF64},
                // NaN
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(kNaNF64), f64Bits(1.0), kNaNF64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(1.0), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(kInfF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-kInfF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(kNaNF64), f64Bits(kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(kNaNF64), f64Bits(-kInfF64), kNaNF64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        SubF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(0.0), f64Bits(0.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-0.0), f64Bits(-0.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-0.0), f64Bits(0.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(1.0), f64Bits(0.0), double{1.0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(5.0), f64Bits(3.0), double{2.0}},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(1.0), f64Bits(1.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(1.0), f64Bits(3.0), double{-2.0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-1.0), f64Bits(2.0), double{-3.0}},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(1.0), f64Bits(-2.0), double{3.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-2.0), f64Bits(-5.0), double{3.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(kInfF64), f64Bits(1.0), kInfF64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-kInfF64), f64Bits(1.0), -kInfF64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(kInfF64), f64Bits(kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(kInfF64), f64Bits(-kInfF64), kInfF64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-kInfF64), f64Bits(kInfF64), -kInfF64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-kInfF64), f64Bits(-kInfF64), kNaNF64},
                // NaN
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(kNaNF64), f64Bits(1.0), kNaNF64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(1.0), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(kInfF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-kInfF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(kNaNF64), f64Bits(kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(kNaNF64), f64Bits(-kInfF64), kNaNF64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        MulF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(0.0), f64Bits(0.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(0.0), f64Bits(-0.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-0.0), f64Bits(0.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-0.0), f64Bits(-0.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(0.0), f64Bits(5.0), double{0.0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(1.0), f64Bits(1.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(2.0), f64Bits(3.0), double{6.0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-2.0), f64Bits(3.0), double{-6.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-2.0), f64Bits(-3.0), double{6.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(kInfF64), f64Bits(1.0), kInfF64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-kInfF64), f64Bits(1.0), -kInfF64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(kInfF64), f64Bits(kInfF64), kInfF64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(kInfF64), f64Bits(-kInfF64), -kInfF64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-kInfF64), f64Bits(kInfF64), -kInfF64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-kInfF64), f64Bits(-kInfF64), kInfF64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(kInfF64), f64Bits(0.0), kNaNF64},
                // NaN
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(kNaNF64), f64Bits(1.0), kNaNF64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(1.0), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(kInfF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-kInfF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(kNaNF64), f64Bits(kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(kNaNF64), f64Bits(-kInfF64), kNaNF64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DivF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(0.0), f64Bits(1.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(0.0), f64Bits(-1.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-0.0), f64Bits(1.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-0.0), f64Bits(-1.0), double{0.0}},
                // Division by zero (not a trap in floating point)
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(0.0), f64Bits(0.0), kNaNF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(1.0), f64Bits(0.0), kInfF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-1.0), f64Bits(0.0), -kInfF64},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(6.0), f64Bits(2.0), double{3.0}},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(1.0), f64Bits(4.0), double{0.25}},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(7.0), f64Bits(7.0), double{1.0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-6.0), f64Bits(2.0), double{-3.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-6.0), f64Bits(-2.0), double{3.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(kInfF64), f64Bits(2.0), kInfF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(kInfF64), f64Bits(-2.0), -kInfF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(kInfF64), f64Bits(kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(kInfF64), f64Bits(-kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-kInfF64), f64Bits(kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-kInfF64), f64Bits(-kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(kInfF64), f64Bits(0.0), kInfF64},
                // NaN
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(kNaNF64), f64Bits(1.0), kNaNF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(1.0), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(kInfF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-kInfF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(kNaNF64), f64Bits(kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(kNaNF64), f64Bits(-kInfF64), kNaNF64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        RemF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(0.0), f64Bits(1.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(0.0), f64Bits(-1.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-0.0), f64Bits(1.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-0.0), f64Bits(-1.0), double{0.0}},
                // Division by zero (not a trap in floating point)
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(0.0), f64Bits(0.0), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(1.0), f64Bits(0.0), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-1.0), f64Bits(0.0), kNaNF64},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(6.0), f64Bits(2.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(1.0), f64Bits(4.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(7.0), f64Bits(7.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(2.5), f64Bits(1.0), double{0.5}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-7.0), f64Bits(3.0), double{-1.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-7.0), f64Bits(-3.0), double{-1.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(kInfF64), f64Bits(2.0), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(kInfF64), f64Bits(-2.0), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(kInfF64), f64Bits(kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(kInfF64), f64Bits(-kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-kInfF64), f64Bits(kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-kInfF64), f64Bits(-kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(kInfF64), f64Bits(0.0), kNaNF64},
                // NaN
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(kNaNF64), f64Bits(1.0), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(1.0), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(kInfF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-kInfF64), f64Bits(kNaNF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(kNaNF64), f64Bits(kInfF64), kNaNF64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(kNaNF64), f64Bits(-kInfF64), kNaNF64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        MaxF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(0.0), f64Bits(0.0), double{0.0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(3.0), f64Bits(5.0), double{5.0}},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(5.0), f64Bits(3.0), double{5.0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(-1.0), f64Bits(1.0), double{1.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(-5.0), f64Bits(-3.0), double{-3.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(kInfF64), f64Bits(1.0), kInfF64},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(-kInfF64), f64Bits(1.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(1.0), f64Bits(kInfF64), kInfF64},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(1.0), f64Bits(-kInfF64), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(kInfF64), f64Bits(kInfF64), kInfF64},
                // NaN
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(kNaNF64), f64Bits(1.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(1.0), f64Bits(kNaNF64), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), kNaNF64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        MinF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(0.0), f64Bits(0.0), double{0.0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(3.0), f64Bits(5.0), double{3.0}},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(5.0), f64Bits(3.0), double{3.0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(-1.0), f64Bits(1.0), double{-1.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(-5.0), f64Bits(-3.0), double{-5.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(kInfF64), f64Bits(1.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(-kInfF64), f64Bits(1.0), -kInfF64},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(1.0), f64Bits(kInfF64), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(1.0), f64Bits(-kInfF64), -kInfF64},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(-kInfF64), f64Bits(-kInfF64), -kInfF64},
                // NaN
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(kNaNF64), f64Bits(1.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(1.0), f64Bits(kNaNF64), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), kNaNF64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(AbsF64, VirtualMachineTwoOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(0.0), double{0.0}},
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(-0.0), double{0.0}},
                                 // Basic positive
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(3.0), double{3.0}},
                                 // Basic negative
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(-3.0), double{3.0}},
                                 // Infinity
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(kInfF64), double{kInfF64}},
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(-kInfF64), double{kInfF64}},
                                 // NaN
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(kNaNF64), double{kNaNF64}},
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(-kNaNF64), double{kNaNF64}}),
                         VirtualMachineTwoOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(NegF64, VirtualMachineTwoOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(0.0), double{-0.0}},
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(-0.0), double{0.0}},
                                 // Basic positive
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(3.0), double{-3.0}},
                                 // Basic negative
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(-3.0), double{3.0}},
                                 // Infinity
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(kInfF64), double{-kInfF64}},
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(-kInfF64), double{kInfF64}},
                                 // NaN
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(kNaNF64), double{kNaNF64}},
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(-kNaNF64), double{kNaNF64}}),
                         VirtualMachineTwoOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(CeilF64, VirtualMachineTwoOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(0.0), double{0.0}},
                                 // Basic positive
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(1.1), double{2.0}},
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(2.0), double{2.0}},
                                 // Basic negative
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(-1.1), double{-1.0}},
                                 // Infinity
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(kInfF64), double{kInfF64}},
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(-kInfF64), double{-kInfF64}},
                                 // NaN
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(kNaNF64), double{kNaNF64}},
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(-kNaNF64), double{kNaNF64}}),
                         VirtualMachineTwoOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(FloorF64, VirtualMachineTwoOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(0.0), double{0.0}},
                                 // Basic positive
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(1.9), double{1.0}},
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(2.0), double{2.0}},
                                 // Basic negative
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(-1.1), double{-2.0}},
                                 // Infinity
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(kInfF64), double{kInfF64}},
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(-kInfF64), double{-kInfF64}},
                                 // NaN
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(kNaNF64), double{kNaNF64}},
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(-kNaNF64), double{kNaNF64}}),
                         VirtualMachineTwoOperandInstructionTest::getTestCaseName);

TEST_P(VirtualMachineRoundF64Test, Execute) {
    const auto& params = GetParam();

    const auto funcName = "test";
    auto bytecode = BytecodeBuilder{}
                            .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/2,
                                                                          {getF64Type()}, {getF64Type()},
                                                                          /*isEntrypoint=*/true)
                                                 .instruction(OpCode::ROUND_F64,
                                                              {/*dst=*/0, /*src=*/1, static_cast<int16_t>(params.flag)})
                                                 .retv(/*resultRegs=*/{0}))
                            .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, funcName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    double srcDouble{};
    std::memcpy(&srcDouble, &params.src, sizeof(srcDouble));
    std::vector<npu_vm_value> args = {intel_npu::vm::makeF64(srcDouble)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeF64(0.0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, funcName, args.size(), args.data(), results.size(), results.data()) ==
                NPU_VM_SUCCESS);

    const auto expected = params.expected;
    const auto actual = results[0].f64;
    if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual));
    } else if (std::isinf(expected)) {
        EXPECT_TRUE(std::isinf(actual));
        EXPECT_EQ(std::signbit(actual), std::signbit(expected));
    } else {
        EXPECT_DOUBLE_EQ(actual, expected);
    }
}

INSTANTIATE_TEST_SUITE_P(RoundF64_RNE, VirtualMachineRoundF64Test,
                         testing::Values(RoundF64Params{f64Bits(1.5), RoundingMode::RNE, 2.0},
                                         RoundF64Params{f64Bits(2.5), RoundingMode::RNE, 2.0},
                                         RoundF64Params{f64Bits(-1.5), RoundingMode::RNE, -2.0},
                                         RoundF64Params{f64Bits(-2.5), RoundingMode::RNE, -2.0},
                                         RoundF64Params{f64Bits(1.4), RoundingMode::RNE, 1.0},
                                         RoundF64Params{f64Bits(1.0), RoundingMode::RNE, 1.0},
                                         RoundF64Params{f64Bits(kInfF64), RoundingMode::RNE, kInfF64},
                                         RoundF64Params{f64Bits(kNaNF64), RoundingMode::RNE, kNaNF64}),
                         VirtualMachineRoundF64Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(RoundF64_RNA, VirtualMachineRoundF64Test,
                         testing::Values(RoundF64Params{f64Bits(1.5), RoundingMode::RNA, 2.0},
                                         RoundF64Params{f64Bits(2.5), RoundingMode::RNA, 3.0},
                                         RoundF64Params{f64Bits(-1.5), RoundingMode::RNA, -2.0},
                                         RoundF64Params{f64Bits(-2.5), RoundingMode::RNA, -3.0},
                                         RoundF64Params{f64Bits(1.4), RoundingMode::RNA, 1.0},
                                         RoundF64Params{f64Bits(1.0), RoundingMode::RNA, 1.0},
                                         RoundF64Params{f64Bits(kInfF64), RoundingMode::RNA, kInfF64},
                                         RoundF64Params{f64Bits(kNaNF64), RoundingMode::RNA, kNaNF64}),
                         VirtualMachineRoundF64Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(RoundF64_RDN, VirtualMachineRoundF64Test,
                         testing::Values(RoundF64Params{f64Bits(1.5), RoundingMode::RDN, 1.0},
                                         RoundF64Params{f64Bits(-1.1), RoundingMode::RDN, -2.0},
                                         RoundF64Params{f64Bits(2.0), RoundingMode::RDN, 2.0},
                                         RoundF64Params{f64Bits(kInfF64), RoundingMode::RDN, kInfF64},
                                         RoundF64Params{f64Bits(kNaNF64), RoundingMode::RDN, kNaNF64}),
                         VirtualMachineRoundF64Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(RoundF64_RUP, VirtualMachineRoundF64Test,
                         testing::Values(RoundF64Params{f64Bits(1.1), RoundingMode::RUP, 2.0},
                                         RoundF64Params{f64Bits(-1.5), RoundingMode::RUP, -1.0},
                                         RoundF64Params{f64Bits(2.0), RoundingMode::RUP, 2.0},
                                         RoundF64Params{f64Bits(kInfF64), RoundingMode::RUP, kInfF64},
                                         RoundF64Params{f64Bits(kNaNF64), RoundingMode::RUP, kNaNF64}),
                         VirtualMachineRoundF64Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(RoundF64_RTZ, VirtualMachineRoundF64Test,
                         testing::Values(RoundF64Params{f64Bits(1.9), RoundingMode::RTZ, 1.0},
                                         RoundF64Params{f64Bits(-1.9), RoundingMode::RTZ, -1.0},
                                         RoundF64Params{f64Bits(2.0), RoundingMode::RTZ, 2.0},
                                         RoundF64Params{f64Bits(kInfF64), RoundingMode::RTZ, kInfF64},
                                         RoundF64Params{f64Bits(kNaNF64), RoundingMode::RTZ, kNaNF64}),
                         VirtualMachineRoundF64Test::getTestCaseName);

TEST_P(VirtualMachineComparisonInstructionTest, Execute) {
    const auto& params = GetParam();

    auto flag = static_cast<int16_t>(params.predicate);
    if (params.isSigned.value_or(false)) {
        flag |= (1 << 8);
    }

    const auto funcName = "test";
    auto bytecode =
            BytecodeBuilder{}
                    .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/3,
                                                                  {getInt64Type(), getInt64Type()}, {getInt64Type()},
                                                                  /*isEntrypoint=*/true)
                                         .instruction(params.opcode, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2, /*flag=*/flag})
                                         .retv(/*resultRegs=*/{0}))
                    .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, funcName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {intel_npu::vm::makeI64(params.lhs), intel_npu::vm::makeI64(params.rhs)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, funcName, args.size(), args.data(), results.size(), results.data()) ==
                NPU_VM_SUCCESS);
    EXPECT_EQ(static_cast<bool>(results.at(0).i64), params.expected);
}

INSTANTIATE_TEST_SUITE_P(
        CMP_I64, VirtualMachineComparisonInstructionTest,
        testing::Values(
                // Equal
                CmpInstructionParams{OpCode::CMP_I64, 0, 0, CmpPredicate::EQ, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 0, 5, CmpPredicate::EQ, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, -5, 0, CmpPredicate::EQ, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, 5, 6, CmpPredicate::EQ, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 5, 5, CmpPredicate::EQ, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 6, 5, CmpPredicate::EQ, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, -5, -5, CmpPredicate::EQ, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, -6, -5, CmpPredicate::EQ, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, kMinI64, kMinI64, CmpPredicate::EQ, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, kMaxI64, kMaxI64, CmpPredicate::EQ, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, kMinI64, kMaxI64, CmpPredicate::EQ, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::EQ,
                                     /*isSigned=*/false, false},  // 0xFFFFFFFFFFFFFFFF is a large unsigned value
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::EQ,
                                     /*isSigned=*/true, false},  // 0xFFFFFFFFFFFFFFFF is -1
                // Not equal
                CmpInstructionParams{OpCode::CMP_I64, 0, 0, CmpPredicate::NE, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 0, 5, CmpPredicate::NE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, -5, 0, CmpPredicate::NE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, 5, 6, CmpPredicate::NE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 5, 5, CmpPredicate::NE, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 6, 5, CmpPredicate::NE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, -5, -5, CmpPredicate::NE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, -6, -5, CmpPredicate::NE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, kMinI64, kMinI64, CmpPredicate::NE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, kMaxI64, kMaxI64, CmpPredicate::NE, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, kMinI64, kMaxI64, CmpPredicate::NE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::NE,
                                     /*isSigned=*/false, true},  // 0xFFFFFFFFFFFFFFFF is a large unsigned value
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::NE,
                                     /*isSigned=*/true, true},  // 0xFFFFFFFFFFFFFFFF is -1
                // Greater than
                CmpInstructionParams{OpCode::CMP_I64, 0, 0, CmpPredicate::GT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 0, 5, CmpPredicate::GT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, -5, 0, CmpPredicate::GT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, 5, 6, CmpPredicate::GT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 5, 5, CmpPredicate::GT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 6, 5, CmpPredicate::GT, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, -5, -6, CmpPredicate::GT, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, -6, -5, CmpPredicate::GT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, kMinI64, kMinI64, CmpPredicate::GT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, kMaxI64, kMaxI64, CmpPredicate::GT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, kMinI64, kMaxI64, CmpPredicate::GT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, kMaxI64, kMinI64, CmpPredicate::GT, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::GT,
                                     /*isSigned=*/false, false},  // 0xFFFFFFFFFFFFFFFF is a large unsigned value
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::GT,
                                     /*isSigned=*/true, true},  // 0xFFFFFFFFFFFFFFFF is -1
                // Greater than or equal
                CmpInstructionParams{OpCode::CMP_I64, 0, 0, CmpPredicate::GTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 0, 5, CmpPredicate::GTE, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, -5, 0, CmpPredicate::GTE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, 5, 6, CmpPredicate::GTE, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 5, 5, CmpPredicate::GTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 6, 5, CmpPredicate::GTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, -5, -6, CmpPredicate::GTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, -6, -5, CmpPredicate::GTE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, kMinI64, kMinI64, CmpPredicate::GTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, kMaxI64, kMaxI64, CmpPredicate::GTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, kMinI64, kMaxI64, CmpPredicate::GTE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, kMaxI64, kMinI64, CmpPredicate::GTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::GTE,
                                     /*isSigned=*/false, false},  // 0xFFFFFFFFFFFFFFFF is a large unsigned value
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::GTE,
                                     /*isSigned=*/true, true},  // 0xFFFFFFFFFFFFFFFF is -1
                // Less than
                CmpInstructionParams{OpCode::CMP_I64, 0, 0, CmpPredicate::LT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 0, 5, CmpPredicate::LT, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, -5, 0, CmpPredicate::LT, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, 5, 6, CmpPredicate::LT, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 5, 5, CmpPredicate::LT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 6, 5, CmpPredicate::LT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, -5, -6, CmpPredicate::LT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, -6, -5, CmpPredicate::LT, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, kMinI64, kMinI64, CmpPredicate::LT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, kMaxI64, kMaxI64, CmpPredicate::LT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, kMinI64, kMaxI64, CmpPredicate::LT, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, kMaxI64, kMinI64, CmpPredicate::LT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::LT,
                                     /*isSigned=*/false, true},  // 0xFFFFFFFFFFFFFFFF is a large unsigned value
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::LT,
                                     /*isSigned=*/true, false},  // 0xFFFFFFFFFFFFFFFF is -1
                // Less than or equal
                CmpInstructionParams{OpCode::CMP_I64, 0, 0, CmpPredicate::LTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 0, 5, CmpPredicate::LTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, -5, 0, CmpPredicate::LTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, 5, 6, CmpPredicate::LTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 5, 5, CmpPredicate::LTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 6, 5, CmpPredicate::LTE, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, -5, -6, CmpPredicate::LTE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, -6, -5, CmpPredicate::LTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, kMinI64, kMinI64, CmpPredicate::LTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, kMaxI64, kMaxI64, CmpPredicate::LTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, kMinI64, kMaxI64, CmpPredicate::LTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, kMaxI64, kMinI64, CmpPredicate::LTE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::LTE,
                                     /*isSigned=*/false, true},  // 0xFFFFFFFFFFFFFFFF is a large unsigned value
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::LTE,
                                     /*isSigned=*/true, false}),  // 0xFFFFFFFFFFFFFFFF is -1
        VirtualMachineComparisonInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        CMP_F64, VirtualMachineComparisonInstructionTest,
        testing::Values(
                // Equal
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(0.0), CmpPredicate::EQ, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(-0.0), CmpPredicate::EQ, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(1.0), CmpPredicate::EQ, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(-1.0), CmpPredicate::EQ, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(1.0), CmpPredicate::EQ, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(-1.0), CmpPredicate::EQ, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kInfF64), f64Bits(kInfF64), CmpPredicate::EQ, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-kInfF64), f64Bits(-kInfF64), CmpPredicate::EQ, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kInfF64), f64Bits(-kInfF64), CmpPredicate::EQ, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), CmpPredicate::EQ, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kNaNF64), f64Bits(1.0), CmpPredicate::EQ, false},
                // Not equal
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(0.0), CmpPredicate::NE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(-0.0), CmpPredicate::NE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(1.0), CmpPredicate::NE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(-1.0), CmpPredicate::NE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(1.0), CmpPredicate::NE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(-1.0), CmpPredicate::NE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kInfF64), f64Bits(kInfF64), CmpPredicate::NE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-kInfF64), f64Bits(-kInfF64), CmpPredicate::NE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kInfF64), f64Bits(-kInfF64), CmpPredicate::NE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), CmpPredicate::NE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kNaNF64), f64Bits(1.0), CmpPredicate::NE, true},
                // Greater than
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(0.0), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(-0.0), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(1.0), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(-1.0), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(1.0), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(-1.0), CmpPredicate::GT, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kInfF64), f64Bits(kInfF64), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-kInfF64), f64Bits(-kInfF64), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kInfF64), f64Bits(-kInfF64), CmpPredicate::GT, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kNaNF64), f64Bits(1.0), CmpPredicate::GT, false},
                // Greater than or equal
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(0.0), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(-0.0), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(1.0), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(-1.0), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(1.0), CmpPredicate::GTE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(-1.0), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kInfF64), f64Bits(kInfF64), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-kInfF64), f64Bits(-kInfF64), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kInfF64), f64Bits(-kInfF64), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), CmpPredicate::GTE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kNaNF64), f64Bits(1.0), CmpPredicate::GTE, false},
                // Less than
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(0.0), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(-0.0), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(1.0), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(-1.0), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(1.0), CmpPredicate::LT, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(-1.0), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kInfF64), f64Bits(kInfF64), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-kInfF64), f64Bits(-kInfF64), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kInfF64), f64Bits(-kInfF64), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kNaNF64), f64Bits(1.0), CmpPredicate::LT, false},
                // Less than or equal
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(0.0), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(-0.0), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(1.0), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(-1.0), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(1.0), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(-1.0), CmpPredicate::LTE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kInfF64), f64Bits(kInfF64), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-kInfF64), f64Bits(-kInfF64), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kInfF64), f64Bits(-kInfF64), CmpPredicate::LTE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kNaNF64), f64Bits(kNaNF64), CmpPredicate::LTE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(kNaNF64), f64Bits(1.0), CmpPredicate::LTE, false}),
        VirtualMachineComparisonInstructionTest::getTestCaseName);

TEST_P(VirtualMachineSelectInstructionTest, Execute) {
    const auto& params = GetParam();

    const auto funcName = "test";
    auto bytecode =
            BytecodeBuilder{}
                    .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/4,
                                                                  {getInt64Type(), getInt64Type(), getInt64Type()},
                                                                  {getInt64Type()},
                                                                  /*isEntrypoint=*/true)
                                         .instruction(OpCode::SELECT, {/*dst=*/0, /*cond=*/1, /*true=*/2, /*false=*/3})
                                         .retv(/*resultRegs=*/{0}))
                    .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, funcName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {intel_npu::vm::makeI64(params.condition),
                                      intel_npu::vm::makeI64(params.trueValue),
                                      intel_npu::vm::makeI64(params.falseValue)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_TRUE(npu_vm_call_with_results(engine, funcName, args.size(), args.data(), results.size(), results.data()) ==
                NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, params.expected);
}

INSTANTIATE_TEST_SUITE_P(
        Select, VirtualMachineSelectInstructionTest,
        testing::Values(SelectInstructionParams{/*condition=*/0, /*trueValue=*/42, /*falseValue=*/84, /*expected=*/84},
                        SelectInstructionParams{/*condition=*/1, /*trueValue=*/42, /*falseValue=*/84, /*expected=*/42},
                        SelectInstructionParams{/*condition=*/-1, /*trueValue=*/42, /*falseValue=*/84,
                                                /*expected=*/42}),
        VirtualMachineSelectInstructionTest::getTestCaseName);

// Execute a single two-operand conversion instruction and compare the result.
// The function signature uses srcType/dstType so that npu_vm_call_with_results
// marshals values correctly (i64, f32, or f64).
TEST_P(VirtualMachineConvertInstructionTest, Execute) {
    const auto& params = GetParam();

    const auto funcName = "test_convert";
    auto bytecode = BytecodeBuilder{}
                            .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/2,
                                                                          {params.srcType}, {params.dstType},
                                                                          /*isEntrypoint=*/true)
                                                 .instruction(params.opcode, {/*dst=*/0, /*src=*/1})
                                                 .retv(/*resultRegs=*/{0}))
                            .build();
    ASSERT_TRUE(npu_vm_parse_module(bytecode.data(), bytecode.size(), &module) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_get_function_info(module, funcName, &funcInfo) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_new_engine(&engine) == NPU_VM_SUCCESS);
    ASSERT_TRUE(npu_vm_load_module(engine, module) == NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {params.src};
    npu_vm_value resultInit{};
    std::vector<npu_vm_value> results = {resultInit};
    ASSERT_TRUE(npu_vm_call_with_results(engine, funcName, args.size(), args.data(), results.size(), results.data()) ==
                NPU_VM_SUCCESS);

    const auto dstTypeCode = intel_npu::vm::getTypeCode(params.dstType);
    if (dstTypeCode == intel_npu::vm::TypeCode::INTEGER) {
        EXPECT_EQ(results[0].i64, params.expected.i64);
    } else {
        const auto dstBitWidth = intel_npu::vm::getBitWidth(params.dstType);
        if (dstBitWidth == sizeof(float) * CHAR_BIT) {
            if (std::isnan(params.expected.f32)) {
                EXPECT_TRUE(std::isnan(results[0].f32));
            } else {
                EXPECT_FLOAT_EQ(results[0].f32, params.expected.f32);
            }
        } else {
            if (std::isnan(params.expected.f64)) {
                EXPECT_TRUE(std::isnan(results[0].f64));
            } else {
                EXPECT_DOUBLE_EQ(results[0].f64, params.expected.f64);
            }
        }
    }
}

constexpr int64_t kMinI8 = std::numeric_limits<int8_t>::min();
constexpr int64_t kMaxI8 = std::numeric_limits<int8_t>::max();
constexpr int64_t kMinI16 = std::numeric_limits<int16_t>::min();
constexpr int64_t kMaxI16 = std::numeric_limits<int16_t>::max();
constexpr int64_t kMinI32 = std::numeric_limits<int32_t>::min();
constexpr int64_t kMaxI32 = std::numeric_limits<int32_t>::max();

constexpr float kInfF32 = std::numeric_limits<float>::infinity();
constexpr float kNaNF32 = std::numeric_limits<float>::quiet_NaN();

// CONVERT_I8_TO_F32: sign-extend low 8 bits to i8, then convert to float32
INSTANTIATE_TEST_SUITE_P(ConvertI8ToF32, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF32(OpCode::CONVERT_I8_TO_F32, 0, 0.0f),
                                         intToF32(OpCode::CONVERT_I8_TO_F32, 1, 1.0f),
                                         intToF32(OpCode::CONVERT_I8_TO_F32, kMaxI8, static_cast<float>(kMaxI8)),
                                         intToF32(OpCode::CONVERT_I8_TO_F32, -1, -1.0f),
                                         intToF32(OpCode::CONVERT_I8_TO_F32, kMinI8, static_cast<float>(kMinI8)),
                                         // 0xFF low byte → i8(-1) → -1.0f
                                         intToF32(OpCode::CONVERT_I8_TO_F32, static_cast<int64_t>(0xFF), -1.0f)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I8_TO_F64: sign-extend low 8 bits to i8, then convert to float64
INSTANTIATE_TEST_SUITE_P(ConvertI8ToF64, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF64(OpCode::CONVERT_I8_TO_F64, 0, 0.0),
                                         intToF64(OpCode::CONVERT_I8_TO_F64, 1, 1.0),
                                         intToF64(OpCode::CONVERT_I8_TO_F64, kMaxI8, static_cast<double>(kMaxI8)),
                                         intToF64(OpCode::CONVERT_I8_TO_F64, -1, -1.0),
                                         intToF64(OpCode::CONVERT_I8_TO_F64, kMinI8, static_cast<double>(kMinI8)),
                                         intToF64(OpCode::CONVERT_I8_TO_F64, static_cast<int64_t>(0xFF), -1.0),
                                         intToF64(OpCode::CONVERT_I8_TO_F64, static_cast<int64_t>(0x1FF), -1.0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I16_TO_I8: truncate i16 to i8 (low 8 bits), sign-extend to i64
INSTANTIATE_TEST_SUITE_P(
        ConvertI16ToI8, VirtualMachineConvertInstructionTest,
        testing::Values(intToInt(OpCode::CONVERT_I16_TO_I8, 0, 0), intToInt(OpCode::CONVERT_I16_TO_I8, 1, 1),
                        intToInt(OpCode::CONVERT_I16_TO_I8, kMaxI8, kMaxI8),
                        // 128 as i16 → low 8 bits = 0x80 = i8(-128)
                        intToInt(OpCode::CONVERT_I16_TO_I8, 128, kMinI8), intToInt(OpCode::CONVERT_I16_TO_I8, -1, -1),
                        // 256 → low 8 bits = 0x00 = 0
                        intToInt(OpCode::CONVERT_I16_TO_I8, 256, 0), intToInt(OpCode::CONVERT_I16_TO_I8, kMaxI16, -1)),
        VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I16_TO_F32: sign-extend low 16 bits to i16, then convert to float32
INSTANTIATE_TEST_SUITE_P(ConvertI16ToF32, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF32(OpCode::CONVERT_I16_TO_F32, 0, 0.0f),
                                         intToF32(OpCode::CONVERT_I16_TO_F32, 1, 1.0f),
                                         intToF32(OpCode::CONVERT_I16_TO_F32, kMaxI16, static_cast<float>(kMaxI16)),
                                         intToF32(OpCode::CONVERT_I16_TO_F32, -1, -1.0f),
                                         intToF32(OpCode::CONVERT_I16_TO_F32, kMinI16, static_cast<float>(kMinI16)),
                                         intToF32(OpCode::CONVERT_I16_TO_F32, static_cast<int64_t>(0xFFFF), -1.0f),
                                         intToF32(OpCode::CONVERT_I16_TO_F32, static_cast<int64_t>(0x8000), -32768.0f)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I16_TO_F64: sign-extend low 16 bits to i16, then convert to float64
INSTANTIATE_TEST_SUITE_P(ConvertI16ToF64, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF64(OpCode::CONVERT_I16_TO_F64, 0, 0.0),
                                         intToF64(OpCode::CONVERT_I16_TO_F64, 1, 1.0),
                                         intToF64(OpCode::CONVERT_I16_TO_F64, kMaxI16, static_cast<double>(kMaxI16)),
                                         intToF64(OpCode::CONVERT_I16_TO_F64, -1, -1.0),
                                         intToF64(OpCode::CONVERT_I16_TO_F64, kMinI16, static_cast<double>(kMinI16)),
                                         intToF64(OpCode::CONVERT_I16_TO_F64, static_cast<int64_t>(0xFFFF), -1.0),
                                         intToF64(OpCode::CONVERT_I16_TO_F64, static_cast<int64_t>(0x18000), -32768.0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I32_TO_I8: truncate i32 to i8 (low 8 bits), sign-extend to i64
INSTANTIATE_TEST_SUITE_P(ConvertI32ToI8, VirtualMachineConvertInstructionTest,
                         testing::Values(intToInt(OpCode::CONVERT_I32_TO_I8, 0, 0),
                                         intToInt(OpCode::CONVERT_I32_TO_I8, 1, 1),
                                         intToInt(OpCode::CONVERT_I32_TO_I8, kMaxI8, kMaxI8),
                                         intToInt(OpCode::CONVERT_I32_TO_I8, 128, kMinI8),
                                         intToInt(OpCode::CONVERT_I32_TO_I8, -1, -1),
                                         intToInt(OpCode::CONVERT_I32_TO_I8, 256, 0),
                                         // INT32_MAX low 8 bits = 0xFF → i8(-1)
                                         intToInt(OpCode::CONVERT_I32_TO_I8, kMaxI32, -1)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I32_TO_I16: truncate i32 to i16 (low 16 bits), sign-extend to i64
INSTANTIATE_TEST_SUITE_P(ConvertI32ToI16, VirtualMachineConvertInstructionTest,
                         testing::Values(intToInt(OpCode::CONVERT_I32_TO_I16, 0, 0),
                                         intToInt(OpCode::CONVERT_I32_TO_I16, 1, 1),
                                         intToInt(OpCode::CONVERT_I32_TO_I16, kMaxI16, kMaxI16),
                                         // 32768 = 0x8000 as i16 → i16(-32768)
                                         intToInt(OpCode::CONVERT_I32_TO_I16, 32768, kMinI16),
                                         intToInt(OpCode::CONVERT_I32_TO_I16, -1, -1),
                                         intToInt(OpCode::CONVERT_I32_TO_I16, 65536, 0),
                                         // INT32_MAX low 16 bits = 0xFFFF → i16(-1)
                                         intToInt(OpCode::CONVERT_I32_TO_I16, kMaxI32, -1)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I32_TO_F32: sign-extend low 32 bits to i32, then convert to float32
INSTANTIATE_TEST_SUITE_P(ConvertI32ToF32, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF32(OpCode::CONVERT_I32_TO_F32, 0, 0.0f),
                                         intToF32(OpCode::CONVERT_I32_TO_F32, 1, 1.0f),
                                         intToF32(OpCode::CONVERT_I32_TO_F32, -1, -1.0f),
                                         intToF32(OpCode::CONVERT_I32_TO_F32, kMaxI32, static_cast<float>(kMaxI32)),
                                         intToF32(OpCode::CONVERT_I32_TO_F32, kMinI32, static_cast<float>(kMinI32)),
                                         intToF32(OpCode::CONVERT_I32_TO_F32, static_cast<int64_t>(0xFFFFFFFF), -1.0f),
                                         intToF32(OpCode::CONVERT_I32_TO_F32, static_cast<int64_t>(0x80000000),
                                                  static_cast<float>(kMinI32))),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I32_TO_F64: sign-extend low 32 bits to i32, then convert to float64
INSTANTIATE_TEST_SUITE_P(ConvertI32ToF64, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF64(OpCode::CONVERT_I32_TO_F64, 0, 0.0),
                                         intToF64(OpCode::CONVERT_I32_TO_F64, 1, 1.0),
                                         intToF64(OpCode::CONVERT_I32_TO_F64, -1, -1.0),
                                         intToF64(OpCode::CONVERT_I32_TO_F64, kMaxI32, static_cast<double>(kMaxI32)),
                                         intToF64(OpCode::CONVERT_I32_TO_F64, kMinI32, static_cast<double>(kMinI32)),
                                         intToF64(OpCode::CONVERT_I32_TO_F64, static_cast<int64_t>(0xFFFFFFFF), -1.0),
                                         intToF64(OpCode::CONVERT_I32_TO_F64, static_cast<int64_t>(0x1FFFFFFFF), -1.0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I64_TO_I8: truncate i64 to i8 (low 8 bits), sign-extend to i64
INSTANTIATE_TEST_SUITE_P(ConvertI64ToI8, VirtualMachineConvertInstructionTest,
                         testing::Values(intToInt(OpCode::CONVERT_I64_TO_I8, 0, 0),
                                         intToInt(OpCode::CONVERT_I64_TO_I8, 1, 1),
                                         intToInt(OpCode::CONVERT_I64_TO_I8, kMaxI8, kMaxI8),
                                         intToInt(OpCode::CONVERT_I64_TO_I8, 128, kMinI8),
                                         intToInt(OpCode::CONVERT_I64_TO_I8, -1, -1),
                                         // INT64_MAX low 8 bits = 0xFF → i8(-1)
                                         intToInt(OpCode::CONVERT_I64_TO_I8, kMaxI64, -1)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I64_TO_I16: truncate i64 to i16 (low 16 bits), sign-extend to i64
INSTANTIATE_TEST_SUITE_P(ConvertI64ToI16, VirtualMachineConvertInstructionTest,
                         testing::Values(intToInt(OpCode::CONVERT_I64_TO_I16, 0, 0),
                                         intToInt(OpCode::CONVERT_I64_TO_I16, 1, 1),
                                         intToInt(OpCode::CONVERT_I64_TO_I16, kMaxI16, kMaxI16),
                                         intToInt(OpCode::CONVERT_I64_TO_I16, 32768, kMinI16),
                                         intToInt(OpCode::CONVERT_I64_TO_I16, -1, -1),
                                         // INT64_MAX low 16 bits = 0xFFFF → i16(-1)
                                         intToInt(OpCode::CONVERT_I64_TO_I16, kMaxI64, -1)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I64_TO_I32: truncate i64 to i32 (low 32 bits), sign-extend to i64
INSTANTIATE_TEST_SUITE_P(ConvertI64ToI32, VirtualMachineConvertInstructionTest,
                         testing::Values(intToInt(OpCode::CONVERT_I64_TO_I32, 0, 0),
                                         intToInt(OpCode::CONVERT_I64_TO_I32, 1, 1),
                                         intToInt(OpCode::CONVERT_I64_TO_I32, kMaxI32, kMaxI32),
                                         intToInt(OpCode::CONVERT_I64_TO_I32, kMinI32, kMinI32),
                                         intToInt(OpCode::CONVERT_I64_TO_I32, -1, -1),
                                         // INT64_MAX low 32 bits = 0xFFFFFFFF → i32(-1)
                                         intToInt(OpCode::CONVERT_I64_TO_I32, kMaxI64, -1)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I64_TO_F32: i64 to float32
INSTANTIATE_TEST_SUITE_P(ConvertI64ToF32, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF32(OpCode::CONVERT_I64_TO_F32, 0, 0.0f),
                                         intToF32(OpCode::CONVERT_I64_TO_F32, 1, 1.0f),
                                         intToF32(OpCode::CONVERT_I64_TO_F32, -1, -1.0f),
                                         intToF32(OpCode::CONVERT_I64_TO_F32, kMaxI64, static_cast<float>(kMaxI64)),
                                         intToF32(OpCode::CONVERT_I64_TO_F32, kMinI64, static_cast<float>(kMinI64))),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I64_TO_F64: i64 to float64
INSTANTIATE_TEST_SUITE_P(ConvertI64ToF64, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF64(OpCode::CONVERT_I64_TO_F64, 0, 0.0),
                                         intToF64(OpCode::CONVERT_I64_TO_F64, 1, 1.0),
                                         intToF64(OpCode::CONVERT_I64_TO_F64, -1, -1.0),
                                         intToF64(OpCode::CONVERT_I64_TO_F64, kMaxI64, static_cast<double>(kMaxI64)),
                                         intToF64(OpCode::CONVERT_I64_TO_F64, kMinI64, static_cast<double>(kMinI64))),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// Float-to-integer conversions use saturating semantics: NaN → 0, overflow → INT_MAX/MIN.

// CONVERT_F32_TO_I8
INSTANTIATE_TEST_SUITE_P(ConvertF32ToI8, VirtualMachineConvertInstructionTest,
                         testing::Values(f32ToInt(OpCode::CONVERT_F32_TO_I8, 0.0f, 0),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, 1.0f, 1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, -1.0f, -1),
                                         // Truncation toward zero
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, 1.9f, 1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, -1.9f, -1),
                                         // In-range boundaries
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, static_cast<float>(kMaxI8), kMaxI8),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, static_cast<float>(kMinI8), kMinI8),
                                         // Overflow saturation
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, 128.0f, kMaxI8),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, -129.0f, kMinI8),
                                         // Special values
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, kInfF32, kMaxI8),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, -kInfF32, kMinI8),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, kNaNF32, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F32_TO_I16
INSTANTIATE_TEST_SUITE_P(ConvertF32ToI16, VirtualMachineConvertInstructionTest,
                         testing::Values(f32ToInt(OpCode::CONVERT_F32_TO_I16, 0.0f, 0),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, 1.0f, 1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, -1.0f, -1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, 1.9f, 1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, -1.9f, -1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, static_cast<float>(kMaxI16), kMaxI16),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, static_cast<float>(kMinI16), kMinI16),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, 32768.0f, kMaxI16),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, -32769.0f, kMinI16),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, kInfF32, kMaxI16),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, -kInfF32, kMinI16),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, kNaNF32, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F32_TO_I32
INSTANTIATE_TEST_SUITE_P(
        ConvertF32ToI32, VirtualMachineConvertInstructionTest,
        testing::Values(f32ToInt(OpCode::CONVERT_F32_TO_I32, 0.0f, 0), f32ToInt(OpCode::CONVERT_F32_TO_I32, 1.5f, 1),
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, -1.5f, -1),
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, 1.0e9f, 1000000000),
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, 1.0e10f, kMaxI32),   // overflow saturation
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, -1.0e10f, kMinI32),  // overflow saturation
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, kInfF32, kMaxI32),
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, -kInfF32, kMinI32),
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, kNaNF32, 0)),
        VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F32_TO_I64
INSTANTIATE_TEST_SUITE_P(ConvertF32ToI64, VirtualMachineConvertInstructionTest,
                         testing::Values(f32ToInt(OpCode::CONVERT_F32_TO_I64, 0.0f, 0),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I64, 1.0f, 1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I64, -1.0f, -1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I64, 1.0e18f, static_cast<int64_t>(1.0e18f)),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I64, kInfF32, kMaxI64),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I64, -kInfF32, kMinI64),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I64, kNaNF32, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F32_TO_F64: widen float32 to float64
INSTANTIATE_TEST_SUITE_P(ConvertF32ToF64, VirtualMachineConvertInstructionTest,
                         testing::Values(f32ToF64(OpCode::CONVERT_F32_TO_F64, 0.0f, 0.0),
                                         f32ToF64(OpCode::CONVERT_F32_TO_F64, 1.5f, static_cast<double>(1.5f)),
                                         f32ToF64(OpCode::CONVERT_F32_TO_F64, -3.14f, static_cast<double>(-3.14f)),
                                         f32ToF64(OpCode::CONVERT_F32_TO_F64, kInfF32, kInfF64),
                                         f32ToF64(OpCode::CONVERT_F32_TO_F64, -kInfF32, -kInfF64),
                                         f32ToF64(OpCode::CONVERT_F32_TO_F64, kNaNF32, kNaNF64)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F64_TO_I8
INSTANTIATE_TEST_SUITE_P(ConvertF64ToI8, VirtualMachineConvertInstructionTest,
                         testing::Values(f64ToInt(OpCode::CONVERT_F64_TO_I8, 0.0, 0),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, 1.0, 1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, -1.0, -1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, 1.9, 1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, -1.9, -1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, static_cast<double>(kMaxI8), kMaxI8),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, static_cast<double>(kMinI8), kMinI8),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, 128.0, kMaxI8),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, -129.0, kMinI8),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, kInfF64, kMaxI8),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, -kInfF64, kMinI8),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, kNaNF64, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F64_TO_I16
INSTANTIATE_TEST_SUITE_P(ConvertF64ToI16, VirtualMachineConvertInstructionTest,
                         testing::Values(f64ToInt(OpCode::CONVERT_F64_TO_I16, 0.0, 0),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, 1.0, 1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, -1.0, -1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, static_cast<double>(kMaxI16), kMaxI16),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, static_cast<double>(kMinI16), kMinI16),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, 32768.0, kMaxI16),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, -32769.0, kMinI16),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, kInfF64, kMaxI16),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, -kInfF64, kMinI16),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, kNaNF64, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F64_TO_I32
INSTANTIATE_TEST_SUITE_P(ConvertF64ToI32, VirtualMachineConvertInstructionTest,
                         testing::Values(f64ToInt(OpCode::CONVERT_F64_TO_I32, 0.0, 0),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, 1.5, 1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, -1.5, -1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, 1.0e9, 1000000000),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, 1.0e10, kMaxI32),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, -1.0e10, kMinI32),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, kInfF64, kMaxI32),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, -kInfF64, kMinI32),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, kNaNF64, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F64_TO_I64
INSTANTIATE_TEST_SUITE_P(ConvertF64ToI64, VirtualMachineConvertInstructionTest,
                         testing::Values(f64ToInt(OpCode::CONVERT_F64_TO_I64, 0.0, 0),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I64, 1.0, 1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I64, -1.0, -1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I64, 1.0e18, static_cast<int64_t>(1.0e18)),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I64, kInfF64, kMaxI64),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I64, -kInfF64, kMinI64),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I64, kNaNF64, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F64_TO_F32: narrow float64 to float32
INSTANTIATE_TEST_SUITE_P(ConvertF64ToF32, VirtualMachineConvertInstructionTest,
                         testing::Values(f64ToF32(OpCode::CONVERT_F64_TO_F32, 0.0, 0.0f),
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, 1.5, 1.5f),
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, -3.14, static_cast<float>(-3.14)),
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, kInfF64, kInfF32),
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, -kInfF64, -kInfF32),
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, kNaNF64, kNaNF32),
                                         // Rounding: midpoint between 1.0f and nextafterf(1.0f, 2.0f)
                                         // ties-to-even rounds to 1.0f (even mantissa LSB)
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, 0x1.0000010000000p+0, 1.0f),
                                         // Midpoint between 0x1.000002p+0 and 0x1.000004p+0
                                         // ties-to-even rounds to 0x1.000004p+0 (even mantissa LSB)
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, 0x1.0000030000000p+0, 0x1.000004p+0f)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

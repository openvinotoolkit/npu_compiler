//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bytecode_builder.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "npu_interpreter_runtime/virtual_machine.h"
#include "vm_value_helpers.hpp"

#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

using namespace intel_npu::vm;
using namespace utils;

namespace {

Type getInt64Type() {
    return Type{IntegerType{sizeof(int64_t) * CHAR_BIT, false}};
}

std::optional<std::vector<npu_vm_value>> buildAndRun(const BytecodeBuilder::FunctionBuilder& functionBuilder,
                                                     const std::vector<npu_vm_value>& initialResults,
                                                     const std::vector<npu_vm_value>& args = {},
                                                     size_t expectedParamCount = 0) {
    EXPECT_EQ(args.size(), expectedParamCount) << "args must provide one value per function parameter";
    if (args.size() != expectedParamCount) {
        return std::nullopt;
    }

    const auto bytecodeBytes = BytecodeBuilder{}.addFunction(functionBuilder).build();
    if (bytecodeBytes.empty()) {
        ADD_FAILURE() << "Failed to build bytecode";
        return std::nullopt;
    }

    npu_vm_module* module = nullptr;
    if (npu_vm_parse_module(bytecodeBytes.data(), bytecodeBytes.size(), &module) != NPU_VM_SUCCESS) {
        ADD_FAILURE() << "Failed to parse serialized bytecode";
        return std::nullopt;
    }
    npu_vm_function_info* funcInfo = nullptr;
    if (npu_vm_get_function_info(module, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME, &funcInfo) != NPU_VM_SUCCESS) {
        npu_vm_destroy_module(module);
        ADD_FAILURE() << "Bytecode does not contain function 'main'";
        return std::nullopt;
    }
    EXPECT_EQ(npu_vm_destroy_function_info(funcInfo), NPU_VM_SUCCESS);

    npu_vm_engine* engine = nullptr;
    if (npu_vm_new_engine(&engine) != NPU_VM_SUCCESS) {
        npu_vm_destroy_module(module);
        ADD_FAILURE() << "Failed to create VM engine";
        return std::nullopt;
    }
    if (npu_vm_load_module(engine, module) != NPU_VM_SUCCESS) {
        npu_vm_destroy_engine(engine);
        npu_vm_destroy_module(module);
        ADD_FAILURE() << "Failed to load module into engine";
        return std::nullopt;
    }

    auto results = initialResults;
    if (npu_vm_call_with_results(engine, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME, args.size(), args.data(), results.size(),
                                 results.data()) != NPU_VM_SUCCESS) {
        npu_vm_destroy_engine(engine);
        npu_vm_destroy_module(module);
        ADD_FAILURE() << "VirtualMachine call failed";
        return std::nullopt;
    }
    npu_vm_destroy_engine(engine);
    npu_vm_destroy_module(module);
    return results;
}

TEST(VirtualMachineInstructionTest, BufferCreateGetDim) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/9,
                                                     /*paramTypes=*/{},
                                                     /*resultTypes=*/{i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/4)
                            .setImm(/*dst=*/2, /*imm=*/6)
                            .setImm(/*dst=*/3, /*imm=*/6)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/2,
                                                                 /*shape=*/1, 2,
                                                                 /*strides=*/3, 4})
                            .setImm(/*dst=*/5, /*imm=*/0)
                            .setImm(/*dst=*/6, /*imm=*/1)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/7, /*buffer=*/0, /*dim=*/5})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/8, /*buffer=*/0, /*dim=*/6})
                            .retv({7, 8});

    const auto results = buildAndRun(function, {intel_npu::vm::makeI64(0), intel_npu::vm::makeI64(0)});

    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 2u);
    EXPECT_EQ(results->at(0).i64, 4);
    EXPECT_EQ(results->at(1).i64, 6);
}

TEST(VirtualMachineInstructionTest, BufferSubviewMatchesRequestedSizes) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/16,
                                                     /*paramTypes=*/{},
                                                     /*resultTypes=*/{i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/4)
                            .setImm(/*dst=*/2, /*imm=*/6)
                            .setImm(/*dst=*/3, /*imm=*/6)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/2,
                                                                 /*shape=*/1, 2,
                                                                 /*strides=*/3, 4})
                            .setImm(/*dst=*/5, /*imm=*/1)
                            .setImm(/*dst=*/6, /*imm=*/1)
                            .setImm(/*dst=*/7, /*imm=*/2)
                            .setImm(/*dst=*/8, /*imm=*/3)
                            .setImm(/*dst=*/9, /*imm=*/1)
                            .setImm(/*dst=*/10, /*imm=*/1)
                            .instruction(OpCode::BUFFER_SUBVIEW, {/*dst=*/11, /*src=*/0, /*rank=*/2,
                                                                  /*offsets=*/5, 6,
                                                                  /*sizes=*/7, 8,
                                                                  /*strides=*/9, 10})
                            .setImm(/*dst=*/12, /*imm=*/0)
                            .setImm(/*dst=*/13, /*imm=*/1)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/14, /*buffer=*/11, /*dim=*/12})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/15, /*buffer=*/11, /*dim=*/13})
                            .retv({14, 15});

    const auto results = buildAndRun(function, {intel_npu::vm::makeI64(0), intel_npu::vm::makeI64(0)});

    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 2u);
    EXPECT_EQ(results->at(0).i64, 2);
    EXPECT_EQ(results->at(1).i64, 3);
}

// Smoke test for buffer.store: build a function that allocates an owned i64 memref via buffer.create, stores a
// parameter value at static indices, and returns. Asserts the VM finalizes without halting. No buffer.load opcode
// exists yet, so the stored value cannot be read back; this exercises the executor's metadata lookup, bounds checking,
// permission check, and write path end-to-end through bytecode parsing and execution.
TEST(VirtualMachineInstructionTest, BufferStoreExecutes) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;
    constexpr int16_t kValueParamReg = 7;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/8,
                                                     /*paramTypes=*/{i64},
                                                     /*resultTypes=*/{},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/4)
                            .setImm(/*dst=*/2, /*imm=*/6)
                            .setImm(/*dst=*/3, /*imm=*/6)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/2,
                                                                 /*shape=*/1, 2,
                                                                 /*strides=*/3, 4})
                            .setImm(/*dst=*/5, /*imm=*/1)
                            .setImm(/*dst=*/6, /*imm=*/2)
                            .instruction(OpCode::BUFFER_STORE, {/*buffer=*/0, /*value=*/kValueParamReg, /*rank=*/2,
                                                                /*indices=*/5, 6})
                            .ret();

    const auto results = buildAndRun(function, /*initialResults=*/{},
                                     /*args=*/{intel_npu::vm::makeI64(static_cast<int64_t>(0x123456789ABCDEF))},
                                     /*expectedParamCount=*/1);

    EXPECT_TRUE(results.has_value());
}

// Build a buffer whose shape is supplied as i64 parameters at call time, then read each dimension back through
// buffer.get_dim. This exercises the runtime path for dynamically-shaped buffers: : the dim values stored in the buffer
// descriptor are whatever shape registers held at buffer.create time, and buffer.get_dim returns those runtime values.
TEST(VirtualMachineInstructionTest, BufferCreateGetDimDynamicShape) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;
    constexpr int16_t kDim0ParamReg = 10;
    constexpr int16_t kDim1ParamReg = 11;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/12,
                                                     /*paramTypes=*/{i64, i64},
                                                     /*resultTypes=*/{i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/2, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/2,
                                                                 /*shape=*/kDim0ParamReg, kDim1ParamReg,
                                                                 /*strides=*/kDim1ParamReg, 2})
                            .setImm(/*dst=*/3, /*imm=*/0)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/5, /*buffer=*/0, /*dim=*/3})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/6, /*buffer=*/0, /*dim=*/4})
                            .retv({5, 6});

    const auto results = buildAndRun(function, {intel_npu::vm::makeI64(0), intel_npu::vm::makeI64(0)},
                                     /*args=*/{intel_npu::vm::makeI64(7), intel_npu::vm::makeI64(11)},
                                     /*expectedParamCount=*/2);

    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 2u);
    EXPECT_EQ(results->at(0).i64, 7);
    EXPECT_EQ(results->at(1).i64, 11);
}

// The source memref's dim 2 and the subview's dim-2 offset are both runtime values; sizes and strides are static.
// This builds a buffer.create with a parameter register in the dim-2 slot followed by a buffer.subview whose dim-2
// offset is the second parameter register. The view's dims must equal the requested sizes [1, 16, 31, 1280] regardless
// of the runtime input values.
TEST(VirtualMachineInstructionTest, BufferSubviewDynamicOffsetReturnsRequestedSizes) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;
    constexpr int16_t kDynSrcDim2ParamReg = 28;
    constexpr int16_t kDynOffsetParamReg = 29;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/30,
                                                     /*paramTypes=*/{i64, i64},
                                                     /*resultTypes=*/{i64, i64, i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/1)
                            .setImm(/*dst=*/2, /*imm=*/16)
                            .setImm(/*dst=*/3, /*imm=*/1280)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/4,
                                                                 /*shape=*/1, 2, kDynSrcDim2ParamReg, 3,
                                                                 /*strides=*/4, 4, 4, 4})
                            .setImm(/*dst=*/6, /*imm=*/0)
                            .setImm(/*dst=*/7, /*imm=*/0)
                            .setImm(/*dst=*/8, /*imm=*/0)
                            .setImm(/*dst=*/9, /*imm=*/1)
                            .setImm(/*dst=*/10, /*imm=*/16)
                            .setImm(/*dst=*/11, /*imm=*/31)
                            .setImm(/*dst=*/12, /*imm=*/1280)
                            .setImm(/*dst=*/13, /*imm=*/1)
                            .setImm(/*dst=*/14, /*imm=*/1)
                            .setImm(/*dst=*/15, /*imm=*/1)
                            .setImm(/*dst=*/16, /*imm=*/1)
                            .instruction(OpCode::BUFFER_SUBVIEW, {/*dst=*/5, /*src=*/0, /*rank=*/4,
                                                                  /*offsets=*/6, 7, kDynOffsetParamReg, 8,
                                                                  /*sizes=*/9, 10, 11, 12,
                                                                  /*strides=*/13, 14, 15, 16})
                            .setImm(/*dst=*/17, /*imm=*/0)
                            .setImm(/*dst=*/18, /*imm=*/1)
                            .setImm(/*dst=*/19, /*imm=*/2)
                            .setImm(/*dst=*/20, /*imm=*/3)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/21, /*buffer=*/5, /*dim=*/17})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/22, /*buffer=*/5, /*dim=*/18})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/23, /*buffer=*/5, /*dim=*/19})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/24, /*buffer=*/5, /*dim=*/20})
                            .retv({21, 22, 23, 24});

    const auto results = buildAndRun(function,
                                     {intel_npu::vm::makeI64(0), intel_npu::vm::makeI64(0), intel_npu::vm::makeI64(0),
                                      intel_npu::vm::makeI64(0)},
                                     /*args=*/{intel_npu::vm::makeI64(100), intel_npu::vm::makeI64(31)},
                                     /*expectedParamCount=*/2);

    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 4u);
    EXPECT_EQ(results->at(0).i64, 1);
    EXPECT_EQ(results->at(1).i64, 16);
    EXPECT_EQ(results->at(2).i64, 31);
    EXPECT_EQ(results->at(3).i64, 1280);
}

// Verify that buffer.view creates a view with the exact shape and strides provided.
// Source is a flat 8-element i64 buffer reinterpreted as a 2x4 row-major view.
TEST(VirtualMachineInstructionTest, BufferViewMatchesRequestedShape) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/13,
                                                     /*paramTypes=*/{},
                                                     /*resultTypes=*/{i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/8)
                            .setImm(/*dst=*/2, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/1,
                                                                 /*shape=*/1,
                                                                 /*strides=*/2})
                            .setImm(/*dst=*/3, /*imm=*/0)
                            .setImm(/*dst=*/4, /*imm=*/2)
                            .setImm(/*dst=*/5, /*imm=*/4)
                            .setImm(/*dst=*/6, /*imm=*/4)
                            .setImm(/*dst=*/7, /*imm=*/1)
                            .instruction(OpCode::BUFFER_VIEW,
                                         {/*dst=*/8, /*src=*/0, /*byteOffset=*/3, kElemI64TypeIndex, /*rank=*/2,
                                          /*shape=*/4, 5,
                                          /*strides=*/6, 7})
                            .setImm(/*dst=*/9, /*imm=*/0)
                            .setImm(/*dst=*/10, /*imm=*/1)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/11, /*buffer=*/8, /*dim=*/9})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/12, /*buffer=*/8, /*dim=*/10})
                            .retv({11, 12});

    const auto results = buildAndRun(function, {intel_npu::vm::makeI64(0), intel_npu::vm::makeI64(0)});

    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 2u);
    EXPECT_EQ(results->at(0).i64, 2);
    EXPECT_EQ(results->at(1).i64, 4);
}

// Verify buffer.view with runtime-supplied shape registers: both shape dims are function parameters,
// exercising the dynamic-register path through the VM's buffer.view handler.
// Source is a flat 16-element i64 buffer; view shape {rows, cols} = {4, 4} is passed at call time.
TEST(VirtualMachineInstructionTest, BufferViewDynamicShapeMatchesRuntimeValues) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;
    constexpr int16_t kRowsParamReg = 12;
    constexpr int16_t kColsParamReg = 13;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/14,
                                                     /*paramTypes=*/{i64, i64},
                                                     /*resultTypes=*/{i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/16)
                            .setImm(/*dst=*/2, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/1,
                                                                 /*shape=*/1,
                                                                 /*strides=*/2})
                            .setImm(/*dst=*/3, /*imm=*/0)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .instruction(OpCode::BUFFER_VIEW,
                                         {/*dst=*/5, /*src=*/0, /*byteOffset=*/3, kElemI64TypeIndex, /*rank=*/2,
                                          /*shape=*/kRowsParamReg, kColsParamReg,
                                          /*strides=*/kColsParamReg, 4})
                            .setImm(/*dst=*/6, /*imm=*/0)
                            .setImm(/*dst=*/7, /*imm=*/1)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/8, /*buffer=*/5, /*dim=*/6})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/9, /*buffer=*/5, /*dim=*/7})
                            .retv({8, 9});

    const auto results = buildAndRun(function, {intel_npu::vm::makeI64(0), intel_npu::vm::makeI64(0)},
                                     /*args=*/{intel_npu::vm::makeI64(4), intel_npu::vm::makeI64(4)},
                                     /*expectedParamCount=*/2);

    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 2u);
    EXPECT_EQ(results->at(0).i64, 4);
    EXPECT_EQ(results->at(1).i64, 4);
}

TEST(VirtualMachineInstructionTest, BufferViewWithNonZeroByteOffset) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/9,
                                                     /*paramTypes=*/{},
                                                     /*resultTypes=*/{i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/4)
                            .setImm(/*dst=*/2, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/1,
                                                                 /*shape=*/1,
                                                                 /*strides=*/2})
                            // Byte offset 16 skips two i64 elements (2x8 bytes), leaving a 2-element view.
                            .setImm(/*dst=*/3, /*imm=*/16)
                            .setImm(/*dst=*/4, /*imm=*/2)
                            .setImm(/*dst=*/5, /*imm=*/1)
                            .instruction(OpCode::BUFFER_VIEW,
                                         {/*dst=*/6, /*src=*/0, /*byteOffset=*/3, kElemI64TypeIndex, /*rank=*/1,
                                          /*shape=*/4,
                                          /*strides=*/5})
                            .setImm(/*dst=*/7, /*imm=*/0)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/8, /*buffer=*/6, /*dim=*/7})
                            .retv({8});

    const auto results = buildAndRun(function, {intel_npu::vm::makeI64(0)});

    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 1u);
    EXPECT_EQ(results->at(0).i64, 2);
}

}  // namespace

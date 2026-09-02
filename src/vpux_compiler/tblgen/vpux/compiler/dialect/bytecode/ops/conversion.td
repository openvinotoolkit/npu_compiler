//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_CONVERSION
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_CONVERSION

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"

// Conversion instructions (integer narrowing, integer<->float, float widening/narrowing)
// Integer narrowing: the source value is truncated to the destination bit width, then sign-extended to 64 bits for storage.

def Bytecode_ConvertI8ToF32Op : Bytecode_TwoOperandsInstructionBase<"convert.i8tof32", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I8_TO_F32"> {
    let summary = "Convert an 8-bit integer to a 32-bit float";
    let description = [{
        Interprets the low 8 bits of `src` as a signed 8-bit integer and converts it to the nearest IEEE 754
        single-precision value, stored in the low 32 bits of `dst`.

        Example:
        ```mlir
        bytecode.convert.i8tof32 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI8ToF64Op : Bytecode_TwoOperandsInstructionBase<"convert.i8tof64", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I8_TO_F64"> {
    let summary = "Convert an 8-bit integer to a 64-bit float";
    let description = [{
        Interprets the low 8 bits of `src` as a signed 8-bit integer and converts it to the nearest IEEE 754
        double-precision value stored in `dst`.

        Example:
        ```mlir
        bytecode.convert.i8tof64 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI16ToI8Op : Bytecode_TwoOperandsInstructionBase<"convert.i16toi8", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I16_TO_I8"> {
    let summary = "Truncate a 16-bit integer to an 8-bit integer";
    let description = [{
        Interprets the low 16 bits of `src` as a signed 16-bit integer, truncates to 8 bits, then sign-extends
        to 64 bits for storage in `dst`.

        Example:
        ```mlir
        bytecode.convert.i16toi8 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI16ToF32Op : Bytecode_TwoOperandsInstructionBase<"convert.i16tof32", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I16_TO_F32"> {
    let summary = "Convert a 16-bit integer to a 32-bit float";
    let description = [{
        Interprets the low 16 bits of `src` as a signed 16-bit integer and converts it to the nearest IEEE 754
        single-precision value, stored in the low 32 bits of `dst`.

        Example:
        ```mlir
        bytecode.convert.i16tof32 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI16ToF64Op : Bytecode_TwoOperandsInstructionBase<"convert.i16tof64", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I16_TO_F64"> {
    let summary = "Convert a 16-bit integer to a 64-bit float";
    let description = [{
        Interprets the low 16 bits of `src` as a signed 16-bit integer and converts it to the nearest IEEE 754
        double-precision value stored in `dst`.

        Example:
        ```mlir
        bytecode.convert.i16tof64 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI32ToI8Op : Bytecode_TwoOperandsInstructionBase<"convert.i32toi8", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I32_TO_I8"> {
    let summary = "Truncate a 32-bit integer to an 8-bit integer";
    let description = [{
        Interprets the low 32 bits of `src` as a signed 32-bit integer, truncates to 8 bits, then sign-extends
        to 64 bits for storage in `dst`.

        Example:
        ```mlir
        bytecode.convert.i32toi8 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI32ToI16Op : Bytecode_TwoOperandsInstructionBase<"convert.i32toi16", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I32_TO_I16"> {
    let summary = "Truncate a 32-bit integer to a 16-bit integer";
    let description = [{
        Interprets the low 32 bits of `src` as a signed 32-bit integer, truncates to 16 bits, then sign-extends
        to 64 bits for storage in `dst`.

        Example:
        ```mlir
        bytecode.convert.i32toi16 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI32ToF32Op : Bytecode_TwoOperandsInstructionBase<"convert.i32tof32", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I32_TO_F32"> {
    let summary = "Convert a 32-bit integer to a 32-bit float";
    let description = [{
        Interprets the low 32 bits of `src` as a signed 32-bit integer and converts it to the nearest IEEE 754
        single-precision value, stored in the low 32 bits of `dst`.

        Example:
        ```mlir
        bytecode.convert.i32tof32 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI32ToF64Op : Bytecode_TwoOperandsInstructionBase<"convert.i32tof64", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I32_TO_F64"> {
    let summary = "Convert a 32-bit integer to a 64-bit float";
    let description = [{
        Interprets the low 32 bits of `src` as a signed 32-bit integer and converts it to the nearest IEEE 754
        double-precision value stored in `dst`.

        Example:
        ```mlir
        bytecode.convert.i32tof64 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI64ToI8Op : Bytecode_TwoOperandsInstructionBase<"convert.i64toi8", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I64_TO_I8"> {
    let summary = "Truncate a 64-bit integer to an 8-bit integer";
    let description = [{
        Truncates `src` to 8 bits, then sign-extends to 64 bits for storage in `dst`.

        Example:
        ```mlir
        bytecode.convert.i64toi8 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI64ToI16Op : Bytecode_TwoOperandsInstructionBase<"convert.i64toi16", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I64_TO_I16"> {
    let summary = "Truncate a 64-bit integer to a 16-bit integer";
    let description = [{
        Truncates `src` to 16 bits, then sign-extends to 64 bits for storage in `dst`.

        Example:
        ```mlir
        bytecode.convert.i64toi16 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI64ToI32Op : Bytecode_TwoOperandsInstructionBase<"convert.i64toi32", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I64_TO_I32"> {
    let summary = "Truncate a 64-bit integer to a 32-bit integer";
    let description = [{
        Truncates `src` to 32 bits, then sign-extends to 64 bits for storage in `dst`.

        Example:
        ```mlir
        bytecode.convert.i64toi32 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI64ToF32Op : Bytecode_TwoOperandsInstructionBase<"convert.i64tof32", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I64_TO_F32"> {
    let summary = "Convert a 64-bit integer to a 32-bit float";
    let description = [{
        Converts `src` (treated as a signed 64-bit integer) to the nearest IEEE 754 single-precision value,
        stored in the low 32 bits of `dst`.

        Example:
        ```mlir
        bytecode.convert.i64tof32 %dst, %src
        ```
    }];
}

def Bytecode_ConvertI64ToF64Op : Bytecode_TwoOperandsInstructionBase<"convert.i64tof64", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_I64_TO_F64"> {
    let summary = "Convert a 64-bit integer to a 64-bit float";
    let description = [{
        Converts `src` (treated as a signed 64-bit integer) to the nearest IEEE 754 double-precision value
        stored in `dst`.

        Example:
        ```mlir
        bytecode.convert.i64tof64 %dst, %src
        ```
    }];
}

// Float-to-integer conversions.
// The conversion truncates toward zero for finite in-range inputs.
// Saturating semantics are applied for all other inputs:
//   - NaN                              -> 0
//   - finite or infinite, > INT_MAX    -> INT_MAX of the destination type
//   - finite or infinite, < INT_MIN    -> INT_MIN of the destination type

def Bytecode_ConvertF32ToI8Op : Bytecode_TwoOperandsInstructionBase<"convert.f32toi8", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_F32_TO_I8"> {
    let summary = "Convert a 32-bit float to an 8-bit integer";
    let description = [{
        Reads the low 32 bits of `src` as an IEEE 754 single-precision float and converts to an 8-bit signed
        integer using saturating truncation-toward-zero semantics. The result is sign-extended to 64 bits for
        storage in `dst`.
        Saturating semantics: NaN -> 0; src > INT8_MAX -> INT8_MAX; src < INT8_MIN -> INT8_MIN.

        Example:
        ```mlir
        bytecode.convert.f32toi8 %dst, %src
        ```
    }];
}

def Bytecode_ConvertF32ToI16Op : Bytecode_TwoOperandsInstructionBase<"convert.f32toi16", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_F32_TO_I16"> {
    let summary = "Convert a 32-bit float to a 16-bit integer";
    let description = [{
        Reads the low 32 bits of `src` as an IEEE 754 single-precision float and converts to a 16-bit signed
        integer using saturating truncation-toward-zero semantics. The result is sign-extended to 64 bits for
        storage in `dst`.
        Saturating semantics: NaN -> 0; src > INT16_MAX -> INT16_MAX; src < INT16_MIN -> INT16_MIN.

        Example:
        ```mlir
        bytecode.convert.f32toi16 %dst, %src
        ```
    }];
}

def Bytecode_ConvertF32ToI32Op : Bytecode_TwoOperandsInstructionBase<"convert.f32toi32", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_F32_TO_I32"> {
    let summary = "Convert a 32-bit float to a 32-bit integer";
    let description = [{
        Reads the low 32 bits of `src` as an IEEE 754 single-precision float and converts to a 32-bit signed
        integer using saturating truncation-toward-zero semantics. The result is sign-extended to 64 bits for
        storage in `dst`.
        Saturating semantics: NaN -> 0; src > INT32_MAX -> INT32_MAX; src < INT32_MIN -> INT32_MIN.

        Example:
        ```mlir
        bytecode.convert.f32toi32 %dst, %src
        ```
    }];
}

def Bytecode_ConvertF32ToI64Op : Bytecode_TwoOperandsInstructionBase<"convert.f32toi64", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_F32_TO_I64"> {
    let summary = "Convert a 32-bit float to a 64-bit integer";
    let description = [{
        Reads the low 32 bits of `src` as an IEEE 754 single-precision float and converts to a 64-bit signed
        integer using saturating truncation-toward-zero semantics, stored in `dst`.
        Saturating semantics: NaN -> 0; src > INT64_MAX -> INT64_MAX; src < INT64_MIN -> INT64_MIN.

        Example:
        ```mlir
        bytecode.convert.f32toi64 %dst, %src
        ```
    }];
}

def Bytecode_ConvertF32ToF64Op : Bytecode_TwoOperandsInstructionBase<"convert.f32tof64", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_F32_TO_F64"> {
    let summary = "Widen a 32-bit float to a 64-bit float";
    let description = [{
        Reads the low 32 bits of `src` as an IEEE 754 single-precision float and widens it to double precision
        stored in `dst`. This conversion is exact.

        Example:
        ```mlir
        bytecode.convert.f32tof64 %dst, %src
        ```
    }];
}

def Bytecode_ConvertF64ToI8Op : Bytecode_TwoOperandsInstructionBase<"convert.f64toi8", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_F64_TO_I8"> {
    let summary = "Convert a 64-bit float to an 8-bit integer";
    let description = [{
        Reads `src` as an IEEE 754 double-precision float and converts to an 8-bit signed integer using
        saturating truncation-toward-zero semantics. The result is sign-extended to 64 bits for storage in `dst`.
        Saturating semantics: NaN -> 0; src > INT8_MAX -> INT8_MAX; src < INT8_MIN -> INT8_MIN.

        Example:
        ```mlir
        bytecode.convert.f64toi8 %dst, %src
        ```
    }];
}

def Bytecode_ConvertF64ToI16Op : Bytecode_TwoOperandsInstructionBase<"convert.f64toi16", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_F64_TO_I16"> {
    let summary = "Convert a 64-bit float to a 16-bit integer";
    let description = [{
        Reads `src` as an IEEE 754 double-precision float and converts to a 16-bit signed integer using
        saturating truncation-toward-zero semantics. The result is sign-extended to 64 bits for storage in `dst`.
        Saturating semantics: NaN -> 0; src > INT16_MAX -> INT16_MAX; src < INT16_MIN -> INT16_MIN.

        Example:
        ```mlir
        bytecode.convert.f64toi16 %dst, %src
        ```
    }];
}

def Bytecode_ConvertF64ToI32Op : Bytecode_TwoOperandsInstructionBase<"convert.f64toi32", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_F64_TO_I32"> {
    let summary = "Convert a 64-bit float to a 32-bit integer";
    let description = [{
        Reads `src` as an IEEE 754 double-precision float and converts to a 32-bit signed integer using
        saturating truncation-toward-zero semantics. The result is sign-extended to 64 bits for storage in `dst`.
        Saturating semantics: NaN -> 0; src > INT32_MAX -> INT32_MAX; src < INT32_MIN -> INT32_MIN.

        Example:
        ```mlir
        bytecode.convert.f64toi32 %dst, %src
        ```
    }];
}

def Bytecode_ConvertF64ToI64Op : Bytecode_TwoOperandsInstructionBase<"convert.f64toi64", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_F64_TO_I64"> {
    let summary = "Convert a 64-bit float to a 64-bit integer";
    let description = [{
        Reads `src` as an IEEE 754 double-precision float and converts to a 64-bit signed integer using
        saturating truncation-toward-zero semantics, stored in `dst`.
        Saturating semantics: NaN -> 0; src > INT64_MAX -> INT64_MAX; src < INT64_MIN -> INT64_MIN.

        Example:
        ```mlir
        bytecode.convert.f64toi64 %dst, %src
        ```
    }];
}

def Bytecode_ConvertF64ToF32Op : Bytecode_TwoOperandsInstructionBase<"convert.f64tof32", [], "1.0.0", "intel_npu::vm::OpCode::CONVERT_F64_TO_F32"> {
    let summary = "Narrow a 64-bit float to a 32-bit float";
    let description = [{
        Reads `src` as an IEEE 754 double-precision float and narrows it to single precision using the
        IEEE 754 conversion rules (round to nearest, ties to even). The result is stored in the low 32 bits of `dst`.

        Example:
        ```mlir
        bytecode.convert.f64tof32 %dst, %src
        ```
    }];
}

#endif

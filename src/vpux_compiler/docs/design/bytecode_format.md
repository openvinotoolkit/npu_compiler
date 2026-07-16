# Bytecode Format

This document describes the bytecode format used by the NPU compiler for representing the orchestration of kernel execution (i.e. NPU blobs). This binary format is intended to be interpreted by a Virtual Machine (VM), which performs the actual execution during inference. As this format is able to describe the control flow, it is able to support multiple complex use-cases, such as:

- dynamic models, whose data shape is only known during inference
- repeated parametrized function calls, such as for functions encountered in the architecture of LLMs (also known as repeating blocks execution)
- dynamically-dispatched kernels, based on the NPU platform, when diverse platform-specific kernels are contained within the same bytecode file
- weights separation, where the schedule is split between an initialization stage and execution stage

## Bytecode Version History

| Version | Details |
| -- | -- |
| 1.0.0 | Initial release |

## Table of Contents
* [High-Level Overview](#1-high-level-overview)
* [Bytecode Format Prerequisites](#2-bytecode-format-prerequisites)
* [Bytecode File Structure](#3-bytecode-file-structure)
* [Opcodes](#4-opcodes)
* [Bytecode Dialect](#5-bytecode-dialect)
* [Virtual Machine](#6-virtual-machine)

## 1. High-Level Overview

![Bytecode dialect -> Bytecode file -> NPU VM Runtime](../assets/bytecode_overview.svg "Bytecode High-Level Overview")

## 2. Bytecode Format Prerequisites

### 2.1. Endianness

All multi-byte elements inside the format are stored using the little-endian byte order. This includes primitive types, such as integers, as well as the instructions themselves (opcode + operands).

## 3. Bytecode File Structure

The bytecode format is comprised of a file header, followed by multiple sections. The file header contains the following entries:

```
magic_number: "NPUByte\x00",
version: major.minor.patch (uint16_t.uint16_t.uint16_t),
section_header_table: section_header_table,
sections: uint8_t[]
```

### 3.1. Magic Number

The header begins with a magic number, which is a unique binary identifier used for this particular bytecode format. The chosen magic number is represented by the following 8 bytes:

```
"NPUByte\x00"
```

### 3.2. Version

The bytecode format follows the [Semantic Versioning](https://semver.org/) scheme. The `major` number is used to represent changes that break backward compatibility, the `minor` number represents backward compatible, but not forward compatible changes, and the `patch` number represents backward compatible changes.

#### Compatibility

The format aims to provide the following compatibility guarantees:
- backward compatibility: binaries serialized by older versions of the compiler are compatible with newer runtimes
- forward compatibility: binaries serialized by newer versions of the compiler are compatible with older runtimes

Currently, there is no fixed compatibility window for this format. The intention is to provide compatibility as long as possible. If this changes, it will be documented here.

In order to ensure compatibility, some constraints have to be followed. On the compiler side:
- Every operation in the bytecode contains the version where it has been introduced, which should not be changed. Once an operation has been added, it cannot be removed or changed, in a way that would break compatibility (e.g. no semantic changes, no operator mutations, no attribute changes etc). Attributes and types are also versioned. An operation can only use attributes and types that have a version lower or equal to its own.
- Adding a new operation implies increasing the minor number, as the new operation must have this newer version set as the version where it was introduced.
- The compiler has a bytecode target version for serialization. It must only use operations, attributes and types that have the version lower or equal to the target version.
- Every supported NPU platform has a default target version which is used for compilation. Increasing this target version means that forward compatibility is broken. During compilation, a user may toggle specific features which can affect the target bytecode version. If this happens, the compiler reports a warning to inform the user about the minimum runtime version required.

On the runtime side:
- The virtual machine knows what is the latest bytecode version it supports. It will check if the version in the file is compatible and stop the execution otherwise.
- The virtual machine must maintain the support for all previous versions of the bytecode format, as long as backward compatibility is intended to be maintained.

##### Compatibility Testing

On the compiler side, every operation has its serialization tested against all supported target versions. For an operation introduced in version N, the following must be tested:

- The serialization from bytecode IR with targets N, N+1, N+2 etc.
    - these tests should produce the equivalent opcode in binary form
- The serialization from bytecode IR with targets N-1, N-2 etc.
    - these tests will check for expected failures

On the runtime side, pre-compiled bytecode binaries are used to test the compatibility of the VM. These binaries cover all of the existing bytecode versions, which ensure that the interpreter remains compatible with previous versions. All operations are covered by these binaries (e.g. all instructions).

### 3.3. Section Header Table

The sections header table describes every section found within the file. It contains the following:

```
section_header_table {
    num_sections: uint64_t,            // The number of sections inside the file
    section_headers: section_header[]  // The header of every section. The number of headers corresponds with the number of sections
}

section_header {
    type: uint8_t,         // The type of the section
    name_index: uint64_t,  // The name of the section, specified using an index inside the string section
    offset: uint64_t,      // The offset of the section's data within the file
    size: uint64_t,        // The size in bytes of the section
    info: uint8_t[]        // Extra information about the section, which helps interpret the section's data. The information contained depends on the section type
}
```

Each section type has a unique identifier. The format supports the following section types:

#### Function Section

This section contains all of the functions that are executed by the VM. It has the type identifier `0x00`. The info part of the section header has the following structure:

```
info {
    num_functions: uint64_t,              // The number of functions present in the section
    entrypoint_function_index: uint64_t,  // The index of the entrypoint function for execution
    function_info: function_info[]        // Per-function information fields. The number of information fields corresponds with the number of functions
}

function_info {
    name_index: uint64_t,             // The unique name of the function, specified using an index inside the string section
    function_type_index: uint64_t,    // The signature type of the function, specified using an index inside the type section
    num_general_registers: uint64_t,  // The total number of register slots used by the function, including the parameter register slots (i.e. G + P). The number of parameter registers (P) is derived from the referenced function type; the VM subtracts P from this value to obtain the number of scratch general registers (G)
    body_offset: uint64_t,            // The starting offset of the function's body within the section
    body_size: uint64_t               // The size in bytes of the function's body
}
```

#### Constant Section

This section contains constants that are used by functions during execution. It has the type identifier `0x01`. The info part of the section header has the following structure:

```
info {
    num_constants: uint64_t,        // The number of constants present in the section
    constant_info: constant_info[]  // Per-constant information fields. The number of information fields corresponds with the number of constants
}

constant_info {
    constant_offset: uint64_t,  // The starting offset of the constant within the section
    constant_size: uint64_t,    // The size in bytes of the constant
}
```

#### String Section

This section contains strings referenced by the other sections, such as the function section. It contains strings used during execution (e.g. assert messages), as well as the names of the functions. The strings are null-terminated. Having the strings placed in this dedicated section, instead of being inlined, can help us avoid duplicating their value and therefore reduce the potential size of the bytecode format.

It has the type identifier `0x02`. The info part of the section header has the following structure:

```
info {
    num_strings: uint64_t,      // The number of strings present in the section
    string_info: string_info[]  // Per-string information fields. The number of information fields corresponds with the number of strings
}

string_info {
    string_offset: uint64_t,  // The starting offset of the string within the section
    string_size: uint64_t,    // The size in bytes of the string
}
```

#### Kernel Section

This section contains the binary values of the kernels that will be executed on the NPU device. These kernels are referenced by the function bodies, when they are called for execution.

It has the type identifier `0x03`. The info part of the section header has the following structure:

```
info {
    num_kernels: uint64_t,      // The number of kernels present in the section
    kernel_info: kernel_info[]  // Per-kernel information fields. The number of information fields corresponds with the number of kernels
}

kernel_info {
    kernel_offset: uint64_t,  // The starting offset of the kernel within the section
    kernel_size: uint64_t,    // The size in bytes of the kernel
}
```

#### Type Section

This section contains the type definitions used throughout the file. This includes data types (e.g. integer, floating-point), buffer types, function signature types etc.

It has the type identifier `0x04`. The info part of the section header has the following structure:

```
info {
    num_types: uint64_t,    // The number of types present in the section
    type_info: type_info[]  // Per-type information fields. The number of information fields corresponds with the number of types
}

type_info {
    type_offset: uint64_t,  // The starting offset of the type within the section
    type_size: uint64_t,    // The size in bytes of the type
}
```

The type data itself is encoded differently for each supported type definition. Each type definition has an identifier that allows the bytecode parser to interpret the rest of the fields. There can be multiple entries inside the section with the same identifier, but with different fields (e.g. one entry for 32-bit integers, one entry for 64-bit integers). The types used in the rest of the bytecode file are referenced by the type's index within this section.

Below are enumerated the supported types, as well as their binary encoding within the type section:

##### Integer Type

Any integer type, whose width is specified explicitly.

```
integer_type {
    id: uint8_t,        // 0x01
    width: uint8_t,     // The number of bits inside the type
    is_signed: uint8_t  // Whether the integer type is signed (1) or unsigned (0)
}
```

##### Floating-Point Type

Any supported floating-point type, whose width is specified explicitly.

```
float_type {
    id: uint8_t,     // 0x02
    width: uint8_t,  // The number of bits inside the type
    format: uint8_t  // The specific float format:
                     // - float (IEEE 754):              0x00
                     // - bfloat (Brain Floating Point): 0x01
                     // - tfloat (TensorFloat):          0x02
                     // - E4M3 (float8):                 0x03
                     // - E5M2 (float8):                 0x04
                     // - E2M1 (float4):                 0x05
                     // - E8M0:                          0x06
                     // - NF4:                           0x07
}
```

##### Opaque Type

An opaque type, whose semantics are unknown. The width is specified explicitly. This type can be used when the other existing types cannot express it; e.g. opaque data that is passed as input to a kernel call.

```
opaque_type {
    id: uint8_t,    // 0x03
    width: uint8_t  // The number of bits inside the type
}
```

##### Buffer Type

```
buffer_type {
    id: uint8_t,                // 0x04
    data_type_index: uint64_t,  // The index of the data type inside the type section
    rank: uint8_t,              // The number of dimensions of the buffer
    shape: int64_t[],           // The shape of the buffer (-1 represents a dynamic dimension)
    strides: int64_t[]          // The strides of the buffer (-1 represents a dynamic dimension)
}
```

##### Function Type

The function type describes the signature of a function whose code is interpreted by the VM. It describes the number of arguments and return values, as well as their types.

```
function_type {
    id: uint8_t,                      // 0x05
    num_params: uint16_t,             // The number of parameters passed to the function
    param_type_indices: uint64_t[],   // Array of type indices for each parameter
    num_results: uint16_t,            // The number of results returned by the function
    result_type_indices: uint64_t[],  // Array of type indices for each result
}
```

#### Metadata Section

This section contains the network metadata, which describes the network-level information (e.g. its name, the number of streams and command lists), as well as the input, output and profiling output descriptors. This metadata is consumed by the driver and the plugin in order to identify and interpret the network's I/O buffers during inference.

It has the type identifier `0x05`. The info part of the section header has the following structure:

```
info {
    num_metadata: uint64_t,         // The number of metadata entries present in the section
    metadata_info: metadata_info[]  // Per-entry information fields. The number of information fields corresponds with the number of metadata entries
}

metadata_info {
    metadata_offset: uint64_t,  // The starting offset of the metadata entry within the section
    metadata_size: uint64_t,    // The size in bytes of the metadata entry
}
```

Each metadata entry begins with a single-byte record kind, which dictates how the rest of the entry is interpreted. The supported record kinds are:

- network metadata: value `0x00`
- input descriptor: value `0x01`
- output descriptor: value `0x02`
- profiling output descriptor: value `0x03`

There is exactly one network metadata entry, followed by one entry per input, output and profiling output descriptor.

> Note: The network metadata in the Bytecode format matches the existing network metadata used between the NPU compiler and NPU plugin (e.g. for the ELF blob format). That is the reason it includes all of the fields mentioned in the following sections.

##### Network Metadata

The network metadata describes network-level information. Its binary encoding is the following:

```
network_metadata {
    kind: uint8_t,            // 0x00
    name_index: uint64_t,     // The name of the network, specified using an index inside the string section
    num_streams: uint64_t,    // The number of streams used by the network
    num_cmdlists: uint64_t    // The number of command lists used by the network
}
```

##### Descriptor Metadata

The same encoding is used for the input, output and profiling output descriptors, the only difference being the record kind. Each descriptor describes one I/O buffer of the network. Its binary encoding is the following:

```
descriptor_metadata {
    kind: uint8_t,                        // 0x01 (input), 0x02 (output) or 0x03 (profiling output)
    name_index: uint64_t,                 // The name of the descriptor, specified using an index inside the string section
    precision_index: uint64_t,            // The precision (data type) of the descriptor, specified using an index inside the type section
    shape_index: uint64_t,                // The shape of the descriptor, specified using an index inside the constant section
    index_used_by_driver: uint32_t,       // The index used by the driver to identify the I/O buffer
    has_dynamic_strides: uint8_t,         // Whether the descriptor supports a strided (non-contiguous) memory layout (1) or not (0)
    num_tensor_names: uint8_t,            // The number of tensor names associated with the descriptor
    tensor_name_indices: uint64_t[],      // The tensor names, each specified using an index inside the string section. The number of indices corresponds with `num_tensor_names`
    has_shape_from_ir_model: uint8_t,     // Whether the descriptor has a shape extracted from the original IR model (1) or not (0)
    shape_from_ir_model_index: uint64_t,  // The shape extracted from the original IR model, specified using an index inside the constant section. Present only when `has_shape_from_ir_model` is 1
    has_node_friendly_name: uint8_t,      // Whether the descriptor has a friendly node name (1) or not (0)
    node_friendly_name_index: uint64_t    // The friendly node name, specified using an index inside the string section. Present only when `has_node_friendly_name` is 1
}
```

The `shape_from_ir_model_index` field is present in the entry only when `has_shape_from_ir_model` is `1`. Similarly, the `node_friendly_name_index` field is present only when `has_node_friendly_name` is `1`. These optional fields allow reconstructing the original I/O metadata for entries that originate from the IR model, while keeping the encoding compact for compiler-added entries.

## 4. Opcodes

The bytecode format supports a predefined set of instructions, each identified by a unique opcode. Every instruction consists of the opcode and zero or more operands supplying registers or data that are used by the operation. The number of the operands is determined by the opcode.

All of the registers that are used by the instructions have 64 bits. This allows us to reuse the same registers for all instructions, even if only part of the register is used for storing the data. Every opcode has a clear specification on how many bits are used out of the operand registers (e.g. treat the value inside the register as a 32-bit or 64-bit floating-point number). In case an instruction uses only part of the register's data, the data is expected to be placed in the least-significant part of the register.

Each register is identified by a unique register number, which is represented by a signed 16-bit integer.

### Binary Representation

The binary representation of an instruction depends on its number of operands. Every instruction utilizes the following format, from the least-significant bits to the most-significant bits:
- the opcode: the unique identifier of the instruction, stored using 16 bits
- operands: zero or more operands, where each operand is stored using 16 bits
    - Note: the `set.imm`, `jmp`, `je`, and `jne` instructions are exceptions to this, as their immediate operand is a 64-bit signed integer

### Operand Notation

Instruction signatures in the sections below use a uniform notation so that the role of each operand is visible without reading the prose:

| Notation | Meaning |
| -- | -- |
| `rd` | destination register (written by the instruction) |
| `rs` | generic source register |
| `rlhs`, `rrhs` | left and right source operands of a binary instruction (`x[rd] = x[rlhs] OP x[rrhs]`) |
| `r<role>` | semantic source register, named after its role (e.g. `rbuf`, `rcond`) |
| `#imm`, `#<role>` | immediate value, optionally named after its role (e.g. `#off`, `#msg`) |
| `#n<role>` | count immediate value that introduces a variadic group (e.g. `#ndst`, `#nargs`) |
| `r<role>…` | variadic register group whose length is given by the preceding count immediate (e.g. `rdst…`, `rarg…`) |

Each variadic group is named distinctly so that two groups in the same instruction (such as the return and argument lists of `call`) are unambiguous. Unless stated otherwise, immediates are 16 bits; the 64-bit signed exceptions are called out explicitly.

### Arithmetic - Integers

There are dedicated instructions for 64-bit integer arithmetic computation. Signed integer values are expected to be stored in two's complement.

Signatures follow the conventions in [Operand Notation](#operand-notation). `Effect` states the happy-path computation; `Traps` lists the conditions under which the VM halts execution.

Instruction | Signature | Effect | Traps
-- | -- | -- | --
abs.i64 | `abs.i64 rd, rs` | `x[rd] = \|x[rs]\|` (`INT64_MIN` maps to `INT64_MIN`) | -
add.i64 | `add.i64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] + x[rrhs]` (wraps on overflow) | -
div.i64 | `div.i64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] / x[rrhs]` (`INT64_MIN / -1` maps to `INT64_MIN`) | division by zero
max.i64 | `max.i64 rd, rlhs, rrhs` | `x[rd] = max(x[rlhs], x[rrhs])` | -
min.i64 | `min.i64 rd, rlhs, rrhs` | `x[rd] = min(x[rlhs], x[rrhs])` | -
mul.i64 | `mul.i64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] * x[rrhs]` (wraps on overflow) | -
rem.i64 | `rem.i64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] % x[rrhs]` (`INT64_MIN % -1` yields `0`) | division by zero
sub.i64 | `sub.i64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] - x[rrhs]` (wraps on overflow) | -
add.u64 | `add.u64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] + x[rrhs]` (unsigned; wraps on overflow) | -
div.u64 | `div.u64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] / x[rrhs]` (unsigned) | division by zero
max.u64 | `max.u64 rd, rlhs, rrhs` | `x[rd] = max(x[rlhs], x[rrhs])` (unsigned) | -
min.u64 | `min.u64 rd, rlhs, rrhs` | `x[rd] = min(x[rlhs], x[rrhs])` (unsigned) | -
mul.u64 | `mul.u64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] * x[rrhs]` (unsigned; wraps on overflow) | -
rem.u64 | `rem.u64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] % x[rrhs]` (unsigned) | division by zero
sub.u64 | `sub.u64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] - x[rrhs]` (unsigned; wraps on underflow) | -

### Arithmetic - Floating-Point Numbers

There are dedicated instructions for 64-bit floating-point arithmetic computation. The floating-point values are expected to be stored in the [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) format. In case of overflow, the standard IEEE 754 rules apply.

Instruction | Signature | Effect | Traps
-- | -- | -- | --
abs.f64 | `abs.f64 rd, rs` | `x[rd] = \|x[rs]\|` | -
add.f64 | `add.f64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] + x[rrhs]` | -
ceil.f64 | `ceil.f64 rd, rs` | `x[rd] = ceil(x[rs])` (rounds up to the nearest integer) | -
div.f64 | `div.f64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] / x[rrhs]` | -
floor.f64 | `floor.f64 rd, rs` | `x[rd] = floor(x[rs])` (rounds down to the nearest integer) | -
max.f64 | `max.f64 rd, rlhs, rrhs` | `x[rd] = max(x[rlhs], x[rrhs])` (`NaN` operand yields the other operand) | -
min.f64 | `min.f64 rd, rlhs, rrhs` | `x[rd] = min(x[rlhs], x[rrhs])` (`NaN` operand yields the other operand) | -
mul.f64 | `mul.f64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] * x[rrhs]` | -
neg.f64 | `neg.f64 rd, rs` | `x[rd] = -x[rs]` | -
rem.f64 | `rem.f64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] % x[rrhs]` | -
round.f64 | `round.f64 rd, rs, #mode` | `x[rd] = round(x[rs])` using the rounding mode in `#mode` | unrecognized rounding mode
sub.f64 | `sub.f64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] - x[rrhs]` | -

For the `round.f64` instruction, the rounding mode is specified via the `#mode` immediate and has the following meaning:
- RNE (round to nearest, ties to even): value `0x0`
- RNA (round to nearest, ties away from zero): value `0x1`
- RDN (round down): value `0x2`
- RUP (round up): value `0x3`
- RTZ (round toward zero): value `0x4`

### Bitwise

There are dedicated instructions for bitwise manipulation of 64-bit values. None of these instructions trap.

Instruction | Signature | Effect | Traps
-- | -- | -- | --
and.64 | `and.64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] & x[rrhs]` | -
not.64 | `not.64 rd, rs` | `x[rd] = ~x[rs]` | -
or.64 | `or.64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] \| x[rrhs]` | -
xor.64 | `xor.64 rd, rlhs, rrhs` | `x[rd] = x[rlhs] ^ x[rrhs]` | -
sll.64 | `sll.64 rd, rval, rcnt` | `x[rd] = x[rval] << x[rcnt]` (logical left; shift count masked to [0, 63]) | -
srl.64 | `srl.64 rd, rval, rcnt` | `x[rd] = x[rval] >> x[rcnt]` (logical right; shift count masked to [0, 63]) | -
sra.64 | `sra.64 rd, rval, rcnt` | `x[rd] = x[rval] >> x[rcnt]` (arithmetic right, sign-preserving; shift count masked to [0, 63]) | -

### Comparison

There are dedicated instructions for comparing 64-bit integers or 64-bit floating-point numbers. All signed integer values are expected to be stored in two's complement. The instructions make use of a flag to denote whether to treat the values as signed or unsigned, as well as the type of comparison that should be done. The flag is passed as the `#flag` immediate, which has the following binary representation:

15-9 | 8 | 7-0
-- | -- | --
reserved | SIGN | CMP

The meaning of the fields is the following:

- `CMP` specifies the comparison function:
    - `EQ` (equal): value `0x00`
    - `NE` (not equal): value `0x01`
    - `GT` (greater than): value `0x02`
    - `GTE` (greater than or equal): value `0x03`
    - `LT` (less than): value `0x04`
    - `LTE` (less than or equal): value `0x05`
- the `SIGN` bit determines whether to treat the operands as signed (value `1`) or unsigned (value `0`); this sign bit only has meaning for comparisons between integers and is ignored for comparisons between floating-point numbers.

The reserved bits in the flag operand are ignored.

In case the `CMP` function is invalid, the execution halts.

Instruction | Signature | Effect | Traps
-- | -- | -- | --
cmp.i64 | `cmp.i64 rd, rlhs, rrhs, #flag` | `x[rd] = CMP(x[rlhs], x[rrhs])` using the comparison and signedness selected by `#flag` | invalid `CMP` function
cmp.f64 | `cmp.f64 rd, rlhs, rrhs, #flag` | `x[rd] = CMP(x[rlhs], x[rrhs])` using the comparison selected by `#flag`. `NaN` comparisons yield false, except `NE` which is true if either or both operands are `NaN`; `+0` and `-0` compare equal | invalid `CMP` function

### Conversion

There are dedicated instructions for converting between all supported primitive types. Converting an integer from a lower to a higher precision is a no-op, as integer values are stored in 64-bit registers with the sign-bit extended.

Float-to-integer conversions use saturating, truncation-toward-zero semantics: `NaN` maps to `0`, values at or above the destination type's maximum (including `+INF`) saturate to that maximum, and values at or below the destination type's minimum (including `-INF`) saturate to that minimum. None of these instructions trap.

Instruction | Signature | Effect | Traps
-- | -- | -- | --
convert.i8tof32 | `convert.i8tof32 rd, rs` | `x[rd] = convert<float>(x[rs])` | -
convert.i8tof64 | `convert.i8tof64 rd, rs` | `x[rd] = convert<double>(x[rs])` | -
convert.i16toi8 | `convert.i16toi8 rd, rs` | `x[rd] = convert<int8_t>(x[rs])` | -
convert.i16tof32 | `convert.i16tof32 rd, rs` | `x[rd] = convert<float>(x[rs])` | -
convert.i16tof64 | `convert.i16tof64 rd, rs` | `x[rd] = convert<double>(x[rs])` | -
convert.i32toi8 | `convert.i32toi8 rd, rs` | `x[rd] = convert<int8_t>(x[rs])` | -
convert.i32toi16 | `convert.i32toi16 rd, rs` | `x[rd] = convert<int16_t>(x[rs])` | -
convert.i32tof32 | `convert.i32tof32 rd, rs` | `x[rd] = convert<float>(x[rs])` | -
convert.i32tof64 | `convert.i32tof64 rd, rs` | `x[rd] = convert<double>(x[rs])` | -
convert.i64toi8 | `convert.i64toi8 rd, rs` | `x[rd] = convert<int8_t>(x[rs])` | -
convert.i64toi16 | `convert.i64toi16 rd, rs` | `x[rd] = convert<int16_t>(x[rs])` | -
convert.i64toi32 | `convert.i64toi32 rd, rs` | `x[rd] = convert<int32_t>(x[rs])` | -
convert.i64tof32 | `convert.i64tof32 rd, rs` | `x[rd] = convert<float>(x[rs])` | -
convert.i64tof64 | `convert.i64tof64 rd, rs` | `x[rd] = convert<double>(x[rs])` | -
convert.f32toi8 | `convert.f32toi8 rd, rs` | `x[rd] = saturating_convert<int8_t>(x[rs])` | -
convert.f32toi16 | `convert.f32toi16 rd, rs` | `x[rd] = saturating_convert<int16_t>(x[rs])` | -
convert.f32toi32 | `convert.f32toi32 rd, rs` | `x[rd] = saturating_convert<int32_t>(x[rs])` | -
convert.f32toi64 | `convert.f32toi64 rd, rs` | `x[rd] = saturating_convert<int64_t>(x[rs])` | -
convert.f32tof64 | `convert.f32tof64 rd, rs` | `x[rd] = convert<double>(x[rs])` | -
convert.f64toi8 | `convert.f64toi8 rd, rs` | `x[rd] = saturating_convert<int8_t>(x[rs])` | -
convert.f64toi16 | `convert.f64toi16 rd, rs` | `x[rd] = saturating_convert<int16_t>(x[rs])` | -
convert.f64toi32 | `convert.f64toi32 rd, rs` | `x[rd] = saturating_convert<int32_t>(x[rs])` | -
convert.f64toi64 | `convert.f64toi64 rd, rs` | `x[rd] = saturating_convert<int64_t>(x[rs])` | -
convert.f64tof32 | `convert.f64tof32 rd, rs` | `x[rd] = convert<float>(x[rs])` | -

### Conditional assignment

Instruction | Signature | Effect | Traps
-- | -- | -- | --
select | `select rd, rcond, rtrue, rfalse` | `x[rd] = x[rcond] != 0 ? x[rtrue] : x[rfalse]` (the condition is true when `x[rcond]` is non-zero) | -

### Control Flow

Signatures follow the conventions in [Operand Notation](#operand-notation). `Effect` states the happy-path behavior; `Traps` lists the conditions under which the VM halts execution. The jump offsets (`#off`) are 64-bit signed immediates (see [Binary Representation](#binary-representation)); for the variadic `call` and `retv`, each count immediate (`#ndst`, `#nargs`, `#nret`) is followed in the encoding by that many registers.

Instruction | Signature | Effect | Traps
-- | -- | -- | --
call | `call rs, #ndst, rdst…, #nargs, rarg…` | Creates a call frame, passes the `rarg…` values to the callee's parameter registers, jumps to the function at index `x[rs]`, and on return writes the callee's results into `rdst…`. `#ndst` and `#nargs` give the lengths of the register groups that follow them | -
ret | `ret` | Returns control to the caller. At the entrypoint, the special return address halts execution | -
retv | `retv #nret, rret…` | Returns control to the caller and writes the `rret…` values into the caller's `rdst…` registers from the matching `call`. `#nret` must equal that `call`'s `#ndst` | -
jmp | `jmp #off` | `pc += off` (PC-relative) | `off == 0`; target outside the current function's body
je | `je #off, rlhs, rrhs` | `if x[rlhs] == x[rrhs]: pc += off` | `off == 0`; target outside the current function's body
jne | `jne #off, rlhs, rrhs` | `if x[rlhs] != x[rrhs]: pc += off` | `off == 0`; target outside the current function's body
assert | `assert rs, #msg` | If `x[rs] == 0`, halts and prints the string at index `#msg` in the string section; otherwise no trap is taken and no message is printed | `x[rs] == 0`

### Kernel Submission

The compute submission mechanism is modeled after the [oneAPI Level Zero](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/index.html) specification. As a result, concepts such as kernels and command lists are expressed via dedicated instructions. Details about the kernel submission concepts can be found in the [VM kernel submission](#64-kernel-submission) chapter. Each count immediate (`#nin`, `#nout`, `#nsig`, `#nwait`) is followed in the encoding by that many registers.

Instruction | Signature | Effect | Traps
-- | -- | -- | --
kernel.create | `kernel.create rd, #kidx, #kname, #nin, rin…, #nout, rout…` | Creates a kernel handle in `rd` for kernel `#kidx` (index into the kernel section) named by `#kname` (index into the string section), binding the input buffers `rin…` and output buffers `rout…` | invalid kernel index; unknown buffer handles
cmd_list.create | `cmd_list.create rd` | Creates a command list and stores its handle in `rd` | -
cmd_list.add_kernel | `cmd_list.add_kernel rcmd, rkernel, #nsig, rsig…, #nwait, rwait…` | Adds the kernel `x[rkernel]` to the command list `x[rcmd]`, with signal events `rsig…` (at most one) and wait events `rwait…` | `#nsig != 0` or `#nwait != 0` (not yet implemented)
cmd_list.close | `cmd_list.close rs` | Closes the command list `x[rs]`, making it ready to be executed by a command queue | -
cmd_list.exec | `cmd_list.exec rs, #flag` | Executes the command list `x[rs]`; `#flag` selects whether to execute with host sync objects (e.g. fence or event). Does not wait for the execution to finish | -

### Buffers

Buffers are a common primitive used for representing and manipulating large chunks of memory. A buffer's metadata is comprised of the following:
- the rank of the buffer
- the shape of the buffer, which must be static during execution (i.e. not dynamic)
- the strides of the buffer, which must be static during execution (i.e. not dynamic)
- the element type of the buffer

The buffer also contains information about the underlying data, which is not exposed to instructions:
- the memory allocation of the data (i.e. its address and size)
- whether the data is owned by the buffer or not
- the read / write permissions over the data

Buffers are identified by unique handles, which represent opaque 64-bit integers that are managed by the VM. These handles can be obtained upon the creation of a buffer, for example when using `buffer.create`. In this case, the buffer owns the underlying memory and can be deleted. Handles can also be passed by the VM to the entrypoint function, via parameter registers which correspond to the function arguments. In this case, the buffers are created internally by the VM and it references external memory which cannot be deleted; the permission of these buffers can also be limited by the VM upon their creation. More information about the way buffers are handled by the VM can be found in the [Memory Sandboxing](#62-memory-sandboxing) chapter.

The buffer instructions are described in the table below. In every buffer instruction, the element type is referenced as an immediate index into the type section, and each rank immediate (`#rank`) is followed in the encoding by the corresponding register groups (shape and strides; offsets, sizes and strides; or indices). Derived buffers (`buffer.subview`, `buffer.view`) share memory with the source buffer and have their lifetime tied to the parent, so deleting the parent invalidates the derived handle.

Instruction | Signature | Effect | Traps
-- | -- | -- | --
buffer.create | `buffer.create rd, #etype, #rank, rshape…, rstride…` | Allocates a buffer in `rd` with element type `#etype`, rank `#rank`, shape `rshape…` and strides `rstride…`. The buffer owns its memory and has read+write permission | negative rank; invalid element type; invalid shape / strides; allocation failure
buffer.get_dim | `buffer.get_dim rd, rbuf, rdim` | Extracts the dimension `x[rdim]` from the shape of buffer `x[rbuf]` | unknown handle; dimension index out of range
buffer.subview | `buffer.subview rd, rsrc, #rank, roff…, rsize…, rstride…` | Creates in `rd` a subview of buffer `x[rsrc]` described by offsets `roff…`, sizes `rsize…` and strides `rstride…` (each of length `#rank`) | invalid handle; negative / mismatched rank; subview exceeds the source buffer
buffer.view | `buffer.view rd, rsrc, roff, #etype, #rank, rshape…, rstride…` | Creates in `rd` a re-typed and re-shaped view of buffer `x[rsrc]` at byte offset `x[roff]`, with new element type `#etype`, rank `#rank`, shape `rshape…` and strides `rstride…` | invalid handle; negative rank; invalid element type; invalid shape / strides; view range exceeds the source buffer
buffer.store | `buffer.store rbuf, rval, #rank, ridx…` | Stores `x[rval]` into buffer `x[rbuf]` at indices `ridx…` (length `#rank`) | mismatched rank; dimension indices out of range; computed byte offset outside the buffer's range; memory not writable

#### Working with Buffers

##### Creation

```mlir
// Create a buffer using the `buffer.create` instruction. It receives the following operands:
// - the destination register, where the handle of the new buffer will be stored (here `rd`)
// - the element type of the buffer, passed as an immediate index into the type section (here `0`)
// - the rank of the buffer, passed as an immediate value (here `3`)
// - `rank` operands that represent the shape of the buffer (here `r1`, `r2`, `r3`)
// - `rank` operands that represent the strides of the buffer (here `r4`, `r5`, `r6`)
// A buffer will be allocated by this instruction, and its handle will be stored in the destination register (here `rd`).
buffer.create rd, 0, 3, r1, r2, r3, r4, r5, r6
```

##### SubView

```mlir
// The `buffer.subview` instruction takes the following arguments:
// - the destination register, where the handle of the new buffer will be stored (here `rd`)
// - the source register, which contains the handle of the original buffer (here `rs`)
// - the rank of the buffer, passed as an immediate value (here `3`)
// - a variable number of operands, containing the offsets, sizes and strides that describe the subview;
//   there are `3 x rank` operands expected; for this example, rank three is used with the following subview description:
//     offsets: [r1, r2, r3]
//     sizes:   [r4, r5, r6]
//     strides: [r7, r8, r9]
buffer.subview rd, rs, 3, r1, r2, r3, r4, r5, r6, r7, r8, r9
```

### Set

Instruction | Signature | Effect | Traps
-- | -- | -- | --
set | `set rd, rs` | `x[rd] = x[rs]` | -
set.imm | `set.imm rd, #imm` | `x[rd] = imm` (`#imm` is a 64-bit immediate) | -

## 5. Bytecode Dialect

To simplify the representation, implementation and the compatibility testing, a Bytecode dialect is introduced. This dialect is intended to represent a 1-to-1 mapping with the bytecode format described above. This means that every opcode has an equivalent operation inside the dialect, and that every section is represented in the IR, which makes the serialization straight-forward.

### 5.1. Operations

There are two main types of operations represented in the dialect:

1. Section operations, which are meant to be containers for other operations. For example, the constant section contains a list of constant operations, each identified by its index within the section.
2. Instruction operations, which are meant to represent the instruction set of the bytecode format. Each operation is based on the specification of the instruction and contains the unique opcode, as well as the operands. Some operands could have primitive types, such as for immediate values, while others have register types.

Every serializable operation and type contains a field which specifies the version in which it was introduced. This field is used to check whether the operation is compatible with the target version for the compilation. If it is not, the operation must either be converted to functionally-equivalent operations that are compatible, or the compilation must fail with a clear message.

### 5.2. Types

#### Register Type

A register type is introduced to represent the registers used by the instructions. This type is an alias to a 16-bit signed integer, which is used to represent the register number. It is used to represent both global and general registers.

## 6. Virtual Machine

The Virtual Machine (VM) is able to parse the bytecode format and interpret its instructions. An implementation is expected to perform the following steps during execution:

1. Parse the file header, to ensure compatibility with the file version and identify the sections using the section header table.
2. Identify the entrypoint function from the function section, parse the function type associated with it, and initialize its call frame.
3. Allocate the general and parameter registers used by the function. Create external buffers for the input and output data, and set their handles into the frame's parameter registers. The permissions for each external buffer are configured based on the function signature (e.g. read-only for inputs, read-write for outputs).
4. Add the "exit" return address to the call frame. As this is the entrypoint function, the return address that is set has a special meaning to stop execution. The VM implementation can decide what this special return address should be (e.g. the return address could be an optional which has no value).
5. Set the `pc` register to the starting address of the entrypoint function's body.
6. Begin the fetch-decode-execute cycle.

As the bytecode format stores its content in little-endian format, the VM is expected to be executed on a little-endian host. This allows data to be directly interpreted as values, without byte reordering. The VM implementation is expected to check for the byte ordering of the host machine and stop execution if it is not little-endian.

### 6.1. Registers

The VM uses 64-bit registers for all instructions. There are two types of registers used:

#### Global Registers

These registers are used across the entire execution. These are the following:

- `pc` program counter: stores the address of the instruction currently executing; this register is not exposed to instructions and cannot be manually modified; only dedicated instructions (such as `jmp`), can modify its state, via a relative offset

#### General Registers

Beside the global registers, a variable-number of general registers are used during execution. They are identified as `r[0-N]` in this document (e.g. `r0`, `r1`).

The general registers are function-specific, such that every function has its own set of registers. This was chosen as it simplifies the design due to the following:
- it removes the risk of a callee function manipulating the state of a caller function
- in case the compilation optimizes the register utilization via register allocation, such that the number of registers utilized is reduced, there is no need to save the register state when calling a function
- registers could be created and destroyed by the VM implementation, as function calls occur, thus reducing the memory utilized during execution

It is also not necessary to have registers shared across function calls, as the chosen [calling convention](#63-calling-convention) transfers data between calls via the call frames that have their own set of registers.

Every function specifies the number of general registers it utilizes.

### 6.2. Memory Sandboxing

The VM enforces memory safety through buffers. All memory accessed during execution is managed via buffer handles, which encapsulate the underlying memory allocation along with metadata such as ownership and permissions. The VM validates every memory access by checking the buffer's bounds and permissions, ensuring that no out-of-bounds or unauthorized access can occur.

#### Buffer Memory Model

Each buffer managed by the VM tracks the following internal state:

Field | Type | Description
-- | -- | --
host_ptr | void* | The host memory pointer for the buffer's data (internal to the VM, never exposed to bytecode)
size | uint64_t | The total size of the buffer's memory allocation in bytes
ownership | enum | `owned` or `unowned`
permissions | enum | `R` (read-only) or `RW` (read-write)

These fields are maintained internally by the VM and are not directly accessible to bytecode instructions. Instructions interact with buffers exclusively through handles stored in registers.

Buffers are categorized based on their origin:

Category | Ownership | Permissions | Notes
-- | -- | -- | --
Managed buffers | `owned` | `RW` | Buffers allocated via `buffer.create`. The VM allocates the underlying memory.
External buffers | `unowned` | `R` / `RW` | Buffers passed to the entrypoint function as arguments. These reference memory external to the VM (e.g. model input/output data provided by the NPU plugin). The permissions are configured by the VM upon creation based on the function signature (e.g. read-only for inputs, read-write for outputs). These buffers cannot be deleted by bytecode instructions.
Derived buffers | `unowned` | Inherited | Buffers created via `buffer.subview` or `buffer.view`. These share the underlying memory with the source buffer and inherit the source buffer's permissions. Deletion is not permitted since the memory is not owned.
Meta buffers | `unowned` | `R` | Buffers that reference data inside the bytecode file sections (e.g. large constants from the constant section). These are read-only and cannot be deleted.

#### Memory Access Validation

When a buffer-access is executed, the VM performs the following validation steps:

1. **Handle validation**: Verify that the buffer handle refers to a valid, live buffer. If the handle is invalid or references a buffer that has been deleted, the VM traps with `"invalid buffer handle"`.
2. **Bounds check**: Verify that the entire access range falls within the buffer's memory allocation. The access offset and size are computed from the buffer's shape, strides and element type. If the access would exceed the buffer's bounds, the VM traps with `"out-of-bounds buffer access"`.
3. **Permission check**: Verify that the buffer's permissions allow the requested operation. Write operations (e.g. `buffer.store`) require `RW` permission. If the permission is insufficient, the VM traps with `"buffer access permission denied"`.

#### Memory Allocation Limits

To prevent exhaustion of host memory, the VM enforces a configurable maximum total allocation size for owned buffers. When instructions such as `buffer.create` are executed, the VM checks whether the new allocation would exceed this limit. If so, the VM traps with `"memory allocation limit exceeded"`.

### 6.3. Calling Convention

The calling convention used by the bytecode format makes use of the concept of a **call frame**. Each function invocation has its own call frame, which contains the following:
- general registers (scratch), whose number (`G`) is derived by subtracting the parameter count (`P`) from the `num_general_registers` value stored in the section header
- parameter registers, determined by the number of parameters passed to the `call` instruction (`P`, which matches the referenced function type's argument count)
- the return address, which contains the address of the next instruction after the `call` instruction; it is inaccessible to instructions

When a function is called, a call frame is created. Internally, the return address is set and enough registers are allocated to store both the general and parameter registers needed by the function. These registers are zero-initialized, to prevent leaking information between call frames (e.g. if the VM implementation reuses call frame allocations). The registers are identified by unique numbers. If a function has `G` general registers and `P` parameter registers, the call frame will contain `G+P` registers (which equals `num_general_registers` from the function header), each identified by the following numbers:

```
                     general registers         parameter registers
                   |                    |  |                        |
Register numbers: [0, 1, 2, ..., G-2, G-1, G+0, G+1, G+2, ... , G+P-1]

Example:
- 10 general registers
- 5 parameter registers
                   general registers   parameter registers
                   |                |  |                 |
Register numbers: [0, 1, 2, ..., 8, 9, 10, 11, 12, ..., 14]
```

The order of the parameter registers corresponds with the order of parameters passed to the `call` instructions. In other words, the first parameter will correspond with the register number `G+0`, the last parameter will correspond with the register number `G+P-1`.

Beside parameters, the `call` instruction can also receive zero or more destination registers, in which the return values will be stored. When the `retv` instruction is called, the VM will set the return values to the specified destination registers. The order of the destination register operands for the `call` instruction corresponds with the order of operands for the `retv` instruction.

Before executing the entrypoint function, the VM performs the following steps:
1. Initialize a call frame for the entrypoint function, where the requested number of general and parameter registers are allocated.
2. Set the "exit" return address in the call frame.
3. Set the values inside the parameter registers to the entrypoint function's arguments. If these arguments are input and output buffers, the VM will create buffer objects that point to these external addresses, and place the buffer handles in the parameter registers.

The following is an example which shows the calling convention in practice, starting with the entrypoint function which internally calls another function.

```mlir
function_section: [
    // The entrypoint function's call frame is created with the following configuration:
    //   - general registers:   r0, r1 (register numbers 0, 1)
    //   - parameter registers: rp0    (register number 2)
    //   - return address: END
    0: @main, num_args=1, num_results=0, {
        // ---
        // Function body...
        // ---

        // Create a buffer to be used by the inner function.
        // `rp0` holds the element type index, and the remaining operands (that are not shown) describe the rank, shape and strides.
        buffer.create r0, rp0, ...

        // The `call` instruction creates a new call frame and jumps to the target function.
        // It receives one destination register which will store the result value (`r1`), and one parameter registers (`r0`).
        call @inner_fn, 1, r1, 1, r0

        // `r1` now contains the result value from `@inner_fn` (i.e. the value 5)

        // ---
        // Function body...
        // ---

        // The `ret` instruction jumps to the return address, returning control to the caller function.
        // As there is no caller function and the return address has a special meaning, the execution will stop.
        ret
    }

    // The inner function's call frame is created with the following configuration:
    //   - general registers:   r0  (register number 0)
    //   - parameter registers: rp0 (register numbers 1)
    //   - return address: address of next instruction after `call` from @main
    1: @inner_fn, num_args=1, num_results=1, {
        // ---
        // Function body which uses rp0...
        // ---

        set.imm r0, 5  // Prepare the value that will be returned by the function; in this example, return the value 5

        // The `retv` instruction jumps to the return address, returning control to the caller function.
        // The variadic operands specify the returned values. In this case, there is a single returned value contained inside register `r0`
        retv 1, r0
    }
]
```

#### Call Frame Limits

To prevent memory overflow in case of infinite recursions, a limit has been set to the maximum number of call frames that can exist. This limit has been set to 1000. This means that there can be at most 1000 nested function calls at a given time. Additionally, instructions use 16 bits to address register numbers, which is interpreted as a signed integer where the positive numbers correspond to general registers; this means that there is a total of 2^15 addressable registers. As a call frame can also contain parameter registers alongside the general registers, this addressable register space is shared between the two register types. As a result, the total number of registers allocated per call frame (general + parameter) is 2^15.

This results in the following theoretical maximum memory utilization for call frames:

```
max_frames     = 1000
max_regs       = 2^15
reg_size_bytes = 8
Maximum memory utilization for call frames = max_frames * max_regs * reg_size_bytes = 250MB
```

### 6.4. Kernel Submission

The compute submission mechanism is modeled after the [oneAPI Level Zero](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/index.html) specification, which makes use of concepts such as kernels, command lists, command queues, events and barriers:

- **Kernels** are units of compute that are executed on the NPU. They are composed of an ELF blob binary, as well as the input and output buffers used for the execution. The ELF blob binaries are found in the kernel section of the bytecode format.
- **Command lists** represent a sequence of commands to execute. They are populated with kernels for performing computations, or by barriers for synchronization.
- **Command queues** are used for submitting command lists for execution. They allow the programming stage (i.e. populating the command list) to be done separately from the execution stage (i.e. populating the command queue with command lists).
- **Events** are synchronization primitives which can be used by kernels to signal completion, or to delay execution until a signal is received.
- **Barriers** are synchronization primitives that are submitted to the command lists, similar to kernels. They ensure that execution waits until the previous commands in the list are completed.

The bytecode format could also create and manage kernels, events, barriers and command lists, by modeling these elements via dedicated instructions. Part of these concepts are already modeled via dedicated instructions. This gives the code flexibility in generating computation workloads, defining synchronization points between them, submitting work for execution, and waiting for execution to complete. The command queues however are not visible to the bytecode. Instead, an external component (in this case, the NPU plugin) provides the VM a single command queue that can be used for execution. When work is submitted for execution (e.g. via `cmd_list.exec`), the VM will internally make use of the command queue. The command queue must be owned and managed by an external component from the VM, in order to support multiple inferences at the same time, as this external queue could be populated by multiple sources of inferences (e.g. multiple VM instances).

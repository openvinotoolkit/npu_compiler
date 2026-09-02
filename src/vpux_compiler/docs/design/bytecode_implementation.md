# Bytecode Implementation

This document describes the implementation of the [bytecode format](./bytecode_format.md) in the project, including compilation, runtime and the testing infrastructure.

## Table of Contents

- [Compilation](#compilation)
  - [Compilation Example](#compilation-example)
- [Runtime](#runtime)
  - [API](#api)
  - [Libraries](#libraries)
- [Testing Infrastructure](#testing-infrastructure)
  - [Lit-Tests](#lit-tests)
  - [Unit Tests](#unit-tests)
  - [Functional Tests](#functional-tests)
  - [Backward Compatibility Tests](#backward-compatibility-tests)

## Compilation

Neural networks can be compiled into bytecode blobs when using the host compilation pipeline, called `HostCompile`. These blobs contain both the kernels, which are represented by ELF blobs executable on NPU, and the scheduling logic, which is represented by bytecode functions executable in a virtual machine.

The host compilation pipeline partitions the network into individual kernels and separates them from the scheduling logic by outlining them into separate functions, which then get compiled independently as ELF blobs. The partitioning takes place after tiling and vertical fusion, so that the kernels contain subgraphs of operations that fit into CMX. The scheduling logic is then lowered to the `Bytecode` dialect during the backend compilation. This dialect is intended to reflect the bytecode format, as it closely represents all of the items from the format inside the IR. There are dedicated operations for each section, as well as operations for all section entries; this includes constants, strings, kernels, types, metadata and individual instructions. This makes it easy to serialize the IR into a final blob. Registers are also represented via dedicated operations whose results have register types; these registers are not serialized, but their values are used to represent the data flow between instructions.

The lowering is done in two-stages:
1. All host-side items are lowered to the `Bytecode` dialect. This includes all items from upstream dialects that describe the control flow, shape arithmetic, and so on: `arith`, `tensor`, `scf` and other dialects. During this first stage, items that use elements from other sections are stored in-place; for example, an `assert` instruction that uses a string will be lowered to an intermediate `bytecode.ext.assert` instruction which stores the string as an attribute.
2. Convert intermediate operations to the final bytecode operations. During this stage, locally stored elements are moved to their respective sections: string / constant / type operations are created in the relevant section and become addressable via symbols. The `bytecode.ext.assert` example from the first stage gets lowered to the `bytecode.assert` instruction, which uses the associated string's symbol from the string section.

After everything has been lowered to the `Bytecode` dialect, some optimizations are applied, such as reducing the number of duplicated `set.imm` instructions that act as constants. Finally, the last step is register allocation. During lowering, all registers are created as virtual registers. It is necessary to assign a register number to all of the virtual registers, which is done in the register allocation pass.

Now the IR is ready to be serialized into a bytecode blob. The `BytecodeWriter` class contains the serialization logic, and handles the creation of the file header and section payloads. The file header contains the minimum runtime version required to execute the blob. It is deduced based on the operations present in the IR. Each serializable operation (i.e. inheriting `bytecode::SerializableOpInterface`) also contains the bytecode version where it was introduced (i.e. by inheriting `bytecode::VersionedOpInterface`); by extracting the latest version required from the operations inside the IR, the minimum runtime version can be deduced. During serialization, symbols are converted into indices, which is what the instructions expect to be able to use data from other sections. The relative offsets for the jump instructions are also computed, as they are represented as MLIR blocks in the IR.

### Compilation Example

Below is an example of an IR being lowered through the bytecode backend, up to serialization. The example omits the constant, kernel and metadata sections, for simplicity.

```mlir
// The input IR, which uses items from upstream dialects to represent arithmetic computation and control flow

func.func @main(%input: i64) -> () attributes {config.pureHostCompileFunc} {
  %c10 = arith.constant 10 : i64
  %add = arith.addi %input, %c10 : i64
  %cmp  = arith.cmpi sge, %add, %c10 : i64
  cf.assert %cmp, "Assertion failed"
  return
}
```

```mlir
// The IR after the first stage of lowering to Bytecode dialect

bytecode.func_section @func_section {
  // The function is lowered to an intermediate `bytecode.ext.func` operation, which contains
  // the function name and type in-place. All functions are stored inside the function section,
  // which is also represented as a dedicated operation
  bytecode.ext.func @main (i64) -> () {
    // Virtual registers are created at the start. In this case, the register refers
    // to the first parameter of the function. See the bytecode spec for details about
    // general and parameter registers
    %0 = bytecode.virtual_parameter_register 0
    // The `arith.constant` value is lowered to a register via a `bytecode.imm_register` operation
    // (note: `bytecode.imm_register` is a compilation-only helper that will later be lowered to a
    // new virtual general register + `bytecode.set_imm` instruction; it is used to simplify the
    // logic of deduplicating registers that are assigned constants)
    %1 = bytecode.imm_register 10
    // The `arith.addi` instruction is lowered into a new general register, which will store the
    // result, and a `bytecode.add.i64` instruction
    %2 = bytecode.virtual_general_register
    bytecode.add.i64 %2, %0, %1
    // Similarly, the `arith.cmpi` instruction is lowered to a new general register, plus the
    // `bytecode.cmp.i64` instruction. Value `259` represents a 'greater-or-equal' comparison
    // between signed integers, as per the bytecode spec
    %3 = bytecode.virtual_general_register
    bytecode.cmp.i64 %3, %2, %1, 259
    // The assert instruction is lowered into an intermediate `bytecode.ext.assert` operation which
    // stores the message as an attribute
    bytecode.ext.assert %3, "Assertion failed"
    bytecode.ret
  }
}
```

```mlir
// The IR after the second stage of lowering to Bytecode dialect

// The string and types sections are created and populated with entries from the intermediate
// operations that were previously in the IR. Each entry is identified by a unique symbol,
// which is used by the new operations
bytecode.string_section @string_section {
  bytecode.string @function_name "main"
  bytecode.string @assert_msg "Assertion failed"
}
bytecode.type_section @type_section {
  bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
  bytecode.type @function_type #bytecode.function_type<arguments = [@i64], results = []>
}

bytecode.func_section @func_section {
  // The `bytecode.ext.func` operation is converted into the `bytecode.func` operation,
  // which uses the symbols from the string and type sections to refer to the name and type
  bytecode.func @main @string_section::@function_name @type_section::@function_type {
    %0 = bytecode.virtual_parameter_register 0
    %1 = bytecode.imm_register 10
    %2 = bytecode.virtual_general_register
    bytecode.add.i64 %2, %0, %1
    %3 = bytecode.virtual_general_register
    bytecode.cmp.i64 %3, %2, %1, 259
    // The `bytecode.ext.assert` operation is lowered to the `bytecode.assert` instruction,
    // which uses the symbol of the string from the string section to refer to the message
    bytecode.assert %3, @assert_msg
    bytecode.ret
  }
}
```

```mlir
// The IR right before serialization, after registers have been allocated

// When this IR is serialized, each `bytecode.string` and `bytecode.type` entry will be addressed
// via an integer, which represents their index within the respective section
// (e.g. `@string_section::@assert_msg` will have index 1, which will be used by the
// `bytecode.assert` instruction)
bytecode.string_section @string_section {
  bytecode.string @function_name "main"
  bytecode.string @assert_msg "Assertion failed"
}
bytecode.type_section @type_section {
  bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
  bytecode.type @function_type #bytecode.function_type<arguments = [@i64], results = []>
}
bytecode.func_section @func_section {
  bytecode.func @main @string_section::@function_name @type_section::@function_type {
    // All virtual registers have been converted into general registers and have been assigned
    // register numbers. The parameter registers are assigned the last numbers, as per calling
    // convention described in the bytecode spec
    %0 = bytecode.general_register 3
    // `bytecode.imm_register` has been lowered into a new general register and a
    // `bytecode.set_imm` instruction
    %1 = bytecode.general_register 0
    bytecode.set_imm %1, 10
    %2 = bytecode.general_register 1
    bytecode.add.i64 %2, %0, %1
    %3 = bytecode.general_register 2
    bytecode.cmp.i64 %3, %2, %1, 259
    bytecode.assert %3, @assert_msg
    bytecode.ret
  }
}
```

## Runtime

The main part of the runtime component is the virtual machine (VM). It is used to parse the content of a bytecode blob and to execute its content. The general flow is the following:
1. Parse the bytecode blob into a module object.
    - Similar to the serialization logic which uses a `BytecodeWriter` class, the deserialization logic uses a `BytecodeReader` class. Its purpose is to interpret the data inside the bytecode blob, ensure it is valid (i.e. the version is supported, the section offsets and sizes are valid etc.) and extract the information into a format usable by the VM. This means higher-level objects which describe functions, metadata, kernel binaries etc. By default, no memory is copied into the module object; instead, references to the external bytecode blob are used, in order to avoid duplicating memory unnecessarily.
2. Create a VM engine instance and load a module object into it.
    - The VM engine is the main execution unit for bytecode. It contains a reference to the target module object and state. The state describes the current execution and is comprised of the following: a program counter, the buffer allocations, an execution context (used for kernel submissions), etc.
3. (Optional) Query the functions inside the module and extract information about them.
    - The module object can be queried, to find out whether a function exists and what its signature is. This can help the user check what parameters and results the function expects / returns.
4. Execute the (entrypoint) function with the desired parameters and use the result values.
    - The target function can be called with any valid input value. Unless done for testing or debugging purposes, the called function will be an entrypoint function. If the function has any results, they will be returned.
5. Reset the state of the VM engine, for subsequent executions.
    - This is a light-weight operation, which keeps the loaded module and memory allocations, but ensures the VM is in a clean state: reset the program counter, clean the memory etc.

Multiple VM engines can be created, which allows the user to run executions in parallel. The same module object can be loaded by any number of engines, which helps avoid duplicating the memory utilization.

### API

The virtual machine offers two APIs, each intended for different purposes:

1. The VM API, found in [virtual_machine.h](../../../../src/npu_interpreter_runtime/include/npu_interpreter_runtime/virtual_machine.h).
    - This API allows direct control over the VM and exposes functions for printing and parsing a bytecode blob, querying functions, creating engines, loading modules into engines, executing functions, as well as destroying any object created by the API (e.g. destroying a parsed module). It is mainly intended to be used directly only for development and debugging, as the production path uses the second API.
    - It is written in C, in order to ensure binary compatibility for the users of this API.
2. The VM Runtime API, found in [npu_vm_runtime.hpp](../../../../src/npu_interpreter_runtime/include/npu_interpreter_runtime/npu_vm_runtime.hpp).
    - This is the API used in production by the NPU plugin. It represents a wrapper over the VM API, which combines it with Level Zero structures in order to be able to run inferences. Compared to direct execution of functions with the VM API, Level Zero structures are necessary for kernel submission and memory allocation for the NPU. The NPU plugin prepares such structures when calling the VM Runtime API, which is what enables inferences to be executed.
    - The API is in large part modeled after the OpenVINO API, as follows:
        1. An `ov::CompiledModel` is created, either by calling `ov::Core::compile_model` or `ov::Core::import_model`. Internally, `npuVMRuntimeCreate` is called which parses the bytecode blob into a module object.
        2. An `ov::InferRequest` is created from the compiled model. One infer request will be associated with one instance of VM engine.
        3. An inference is called with the desired input & output tensors (e.g. via `ov::InferRequest::infer`), which does the following internally: it prepares the `MemRef` objects for the VM Runtime API via `npuVMRuntimeCreateMemRef` and `npuVMRuntimeSetMemRef`, (in case this is the first inference) it spawns a new VM engine and loads the module object into it via `npuVMRuntimeCreateExecutionContext`, it prepares the Level Zero structure and starts an inference via `npuVMRuntimeExecute`. The inference results are stored in the output tensors that were prepared before the inference, at the application level.
    - `npuVMRuntimeExecute` internally calls a bytecode function that is expected to have the name `main`. For dynamic models, the `npuVMRuntimePredictOutputShape` function is also used before starting an inference when the input shapes change. This calls another bytecode function called `output_shape`, which predicts the output shapes that the dynamic model will produce for the given input shapes. The compiler is expected to produce the two functions with the expected names, to ensure that inferences work.
    - The API is versioned. It is written in C, in order to ensure binary compatibility.

### Libraries

The runtime component consists of two libraries:

- `npu_interpreter_runtime`: A shared library that contains the implementation for the virtual machine and the API(s) described in the previous section. This library has a dependency on Level Zero and it is used by the NPU plugin. It can be found in [src/npu_interpreter_runtime](../../../../src/npu_interpreter_runtime/). The library is only built when `ENABLE_NPU_EXECUTION_ENGINE=ON`.
- `npu_bytecode_utils`: A static library that contains the logic for the serialization and deserialization of a bytecode blob, as well as some helper utilities. This library has a minimal number of dependencies and it is statically linked by the compiler and `npu_interpreter_runtime` library. It can be found in [src/npu_bytecode_utils](../../../../src/npu_bytecode_utils/).

## Testing Infrastructure

There are multiple levels of testing for the bytecode implementation:

### Lit-Tests

The compilation is tested via lit-tests at operation-level (e.g. printing / parsing functionality) and pass-level. The serialization logic is also validated; for this, the `bytecode_interpreter` tool is used in order to print the contents of a serialized bytecode blob and ensure its content is as expected.

The lit-tests can be found in [tests/lit/NPU/dialect/bytecode](../../../../tests/lit/NPU/dialect/bytecode/).

### Unit Tests

There are numerous unit tests that validate the runtime component. This includes the functionality of the VM API, version compatibility checks, memory management infrastructure, various utilities, as well as the functionality of all supported instructions.

The unit tests can be found in [tests/unit/virtual_machine](../../../../tests/unit/virtual_machine/). These tests are only built when `ENABLE_NPU_EXECUTION_ENGINE=ON`.

> Note that instructions which require Level Zero structures for submission are not currently validated via these unit tests. This is done to keep the unit test execution decoupled from an NPU device; they only need a host machine in order to be executed.

### Functional Tests

Functional tests are also used for cases where an NPU device is required for execution. Currently, these tests mainly target the VM Runtime API and the memory management infrastructure that uses Level Zero internally.

The functional tests can be found in [tests/functional/behavior/vm_runtime](../../../../tests/functional/behavior/vm_runtime/). These tests are only built when `ENABLE_NPU_EXECUTION_ENGINE=ON`.

### Backward Compatibility Tests

An infrastructure also exists for testing the backward compatibility of the virtual machine. It makes use of frozen bytecode artifacts at various versions, which are validated against the current runtime. These tests are also executed as part of the suite of unit tests. Detailed information about the artifacts and this infrastructure can be found [here](../../../../tests/artifacts/bytecode/README.md).

//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_SECTION
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_SECTION

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/attributes.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"

def Bytecode_FuncSectionOp : Bytecode_SerializableOp<"func_section", [NoTerminator, Symbol, SymbolTable], "1.0.0"> {
    let summary = "Function section operation";

    let description = [{
        This operation represents a section in the bytecode format that contains function operations.
        The section is identified by a symbol name.

        Example:
        ```mlir
        bytecode.func_section @my_func_section {
            bytecode.func @my_function {
                // Bytecode instructions for the function body
            }
            bytecode.func @another_function {
                // Bytecode instructions for the function body
            }
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let regions = (region
        SizedRegion<1>:$content
    );

    let assemblyFormat = [{
        $sym_name attr-dict-with-keyword
        $content
    }];
}

def Bytecode_FuncOp : Bytecode_SerializableOp<"func", [HasParent<"vpux::bytecode::FuncSectionOp">, IsolatedFromAbove, Symbol,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>], "1.0.0"> {
    let summary = "Function operation";

    let description = [{
        This operation represents a function in the bytecode format and it is meant to be used within a function section.
        The function is identified by a symbol name and references a function type entry in the type section.

        Example:
        ```mlir
        bytecode.func @my_function @string_section::@my_function_name @type_section::@function_type {
            // Bytecode instructions for the function body
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        SymbolRefAttr:$func_name,
        SymbolRefAttr:$function_type_ref
    );

    let regions = (region
        AnyRegion:$body
    );

    let assemblyFormat = [{
        $sym_name $func_name $function_type_ref attr-dict-with-keyword
        $body
    }];
}

def Bytecode_ConstantSectionOp : Bytecode_SerializableOp<"constant_section", [NoTerminator, Symbol, SymbolTable], "1.0.0"> {
    let summary = "Constant section operation";

    let description = [{
        This operation represents a section in the bytecode format that contains constant operations.
        The section is identified by a symbol name.

        Example:
        ```mlir
        bytecode.constant_section @my_constant_section {
            bytecode.constant @my_constant 42 : i64
            bytecode.constant @another_constant 3.14 : f32
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let regions = (region
        SizedRegion<1>:$content
    );

    let assemblyFormat = [{
        $sym_name attr-dict-with-keyword
        $content
    }];
}

def Bytecode_ConstantOp : Bytecode_SerializableOp<"constant", [HasParent<"vpux::bytecode::ConstantSectionOp">, Symbol], "1.0.0"> {
    let summary = "Constant operation";

    let description = [{
        This operation represents a constant in the bytecode format.
        The constant is identified by a symbol name and has a type and value.

        Example:
        ```mlir
        bytecode.constant @my_constant : i32 = 42
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        ElementsAttr:$value
    );

    let assemblyFormat = [{
        $sym_name $value attr-dict-with-keyword
    }];

    let hasVerifier = 1;
}

def Bytecode_StringSectionOp : Bytecode_SerializableOp<"string_section", [NoTerminator, Symbol, SymbolTable], "1.0.0"> {
    let summary = "String section operation";

    let description = [{
        This operation represents a section in the bytecode format that contains string operations.
        The section is identified by a symbol name.

        Example:
        ```mlir
        bytecode.string_section @my_string_section {
            bytecode.string @my_string "Example of a string"
            bytecode.string @another_string "Another example"
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let regions = (region
        SizedRegion<1>:$content
    );

    let assemblyFormat = [{
        $sym_name attr-dict-with-keyword
        $content
    }];
}

def Bytecode_StringOp : Bytecode_SerializableOp<"string", [HasParent<"vpux::bytecode::StringSectionOp">, Symbol], "1.0.0"> {
    let summary = "String operation";

    let description = [{
        This operation represents a string in the bytecode format.
        The string is identified by a symbol name and has a value. The value does not contain a null terminator.

        Example:
        ```mlir
        bytecode.string @my_string "Example of a string"
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        StrAttr:$value
    );

    let assemblyFormat = [{
        $sym_name $value attr-dict-with-keyword
    }];
}

def Bytecode_TypeSectionOp : Bytecode_SerializableOp<"type_section", [NoTerminator, Symbol, SymbolTable], "1.0.0"> {
    let summary = "Type section operation";

    let description = [{
        This operation represents a section in the bytecode format that contains type operations.
        The section is identified by a symbol name.

        Example:
        ```mlir
        bytecode.type_section @my_type_section {
            bytecode.type @i64_type i64
            bytecode.type @f32_type f32
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let regions = (region
        SizedRegion<1>:$content
    );

    let assemblyFormat = [{
        $sym_name attr-dict-with-keyword
        $content
    }];
}

def Bytecode_TypeOp : Bytecode_SerializableOp<"type", [HasParent<"vpux::bytecode::TypeSectionOp">, Symbol], "1.0.0"> {
    let summary = "Type operation";

    let description = [{
        This operation represents a type in the bytecode format.
        The type is identified by a symbol name and has a value.

        Example:
        ```mlir
        bytecode.type @i64_type i64
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        AnyAttrOf<[Bytecode_IntegerTypeAttr, Bytecode_FloatTypeAttr, Bytecode_OpaqueTypeAttr,
                   Bytecode_BufferTypeAttr, Bytecode_FunctionTypeAttr]>:$value
    );

    let assemblyFormat = [{
        $sym_name $value attr-dict-with-keyword
    }];
}


def Bytecode_KernelSectionOp : Bytecode_SerializableOp<"kernel_section", [NoTerminator, Symbol, SymbolTable], "1.0.0"> {
    let summary = "Kernel section operation";

    let description = [{
        This operation represents a section in the bytecode format that contains kernel binary data.
        Each kernel is a serialized ELF binary blob identified by a symbol name.

        Example:
        ```mlir
        bytecode.kernel_section @my_kernel_section {
            bytecode.kernel @first_kernel "\7FELF..."
            bytecode.kernel @second_kernel "\7FELF..."
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let regions = (region
        SizedRegion<1>:$content
    );

    let assemblyFormat = [{
        $sym_name attr-dict-with-keyword
        $content
    }];
}

def Bytecode_KernelOp : Bytecode_SerializableOp<"kernel", [HasParent<"vpux::bytecode::KernelSectionOp">, Symbol], "1.0.0"> {
    let summary = "Kernel operation";

    let description = [{
        This operation represents a kernel binary in the bytecode format.
        The kernel is identified by a symbol name and carries a serialized ELF binary blob.

        Example:
        ```mlir
        bytecode.kernel @my_kernel "\7FELF..."
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        StrAttr:$data
    );

    let assemblyFormat = [{
        $sym_name $data attr-dict-with-keyword
    }];
}

def Bytecode_MetadataSectionOp : Bytecode_SerializableOp<"metadata_section", [NoTerminator, Symbol, SymbolTable], "1.0.0"> {
    let summary = "Metadata section operation";

    let description = [{
        This operation represents a section in the bytecode IR that contains metadata operations.
        It is used as a container for network metadata and does not participate in binary serialization yet.

        Example:
        ```mlir
        bytecode.metadata_section @metadata_section {
            bytecode.network_metadata @string_section::@str_network_name 1
            bytecode.input_metadata @string_section::@str_input_dynamic @type_section::@type_f16 @constant_section::@shape_value 0 false
            bytecode.output_metadata @string_section::@str_output_dynamic @type_section::@type_f16 @constant_section::@shape_value 0 false
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let regions = (region
        SizedRegion<1>:$content
    );

    let assemblyFormat = [{
        $sym_name attr-dict-with-keyword
        $content
    }];
}

#endif

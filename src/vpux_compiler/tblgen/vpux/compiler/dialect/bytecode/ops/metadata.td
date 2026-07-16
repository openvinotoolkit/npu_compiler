//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/BuiltinAttributes.td"

class Bytecode_MetadataDescriptorOpBase<string mnemonic> :
        Bytecode_SerializableOp<mnemonic, [HasParent<"vpux::bytecode::MetadataSectionOp">], "1.0.0"> {
    let arguments = (ins
        SymbolRefAttr:$name,
        SymbolRefAttr:$precision,
        SymbolRefAttr:$shape,
        I32Attr:$index_used_by_driver,
        BoolAttr:$has_dynamic_strides
    );

    let assemblyFormat = [{
        $name
        attr-dict
        $precision
        $shape
        `index_used_by_driver` `(` $index_used_by_driver `)`
        `has_dynamic_strides` `(` $has_dynamic_strides `)`
    }];
}

def Bytecode_NetworkMetadataOp : Bytecode_SerializableOp<"network_metadata", [HasParent<"vpux::bytecode::MetadataSectionOp">], "1.0.0"> {
    let summary = "Root network metadata operation";

    let description = [{
        Root metadata entry containing network-level information.
        Descriptor operations (input/output/profiling) are stored in the same metadata section.
    }];

    let arguments = (ins
        SymbolRefAttr:$name,
        I64Attr:$num_streams,
        I64Attr:$num_cmdlists
    );

    let assemblyFormat = [{
        $name $num_streams $num_cmdlists attr-dict
    }];
}

def Bytecode_InputMetadataOp : Bytecode_MetadataDescriptorOpBase<"input_metadata"> {
    let summary = "Input metadata descriptor operation";

    let description = [{
        Metadata descriptor for one network input.
    }];
}

def Bytecode_OutputMetadataOp : Bytecode_MetadataDescriptorOpBase<"output_metadata"> {
    let summary = "Output metadata descriptor operation";

    let description = [{
        Metadata descriptor for one network output.
    }];
}

def Bytecode_ProfilingOutputMetadataOp : Bytecode_MetadataDescriptorOpBase<"profiling_output_metadata"> {
    let summary = "Profiling output metadata descriptor operation";

    let description = [{
        Metadata descriptor for one profiling output.
    }];
}

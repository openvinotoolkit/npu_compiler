//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_CONTROL_FLOW
#define VPUX_COMPILER_DIALECT_IE_OPS_CONTROL_FLOW

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/core/types.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"

def IE_IfOp :
        IE_LayerOp<
            "If",
            [
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                                ["inferReturnTypeComponents"]>,
                SingleBlockImplicitTerminator<"YieldOp">,
                RecursiveMemoryEffects
            ]
        > {
    let summary = "InferenceEngine Conditional if layer";

    let description = [{
    Evaluates a Boolean condition and then takes one of two distinct execution
    paths. This implements the semantic If-then-else structure.
    }];

    let arguments = (ins
        1DTensorOf<[Bool8, SI8]>:$cond,
        Variadic<AnyRankedTensor>:$inputs
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$output
    );

    let regions = (region
        SizedRegion<1>:$then_branch,
        SizedRegion<1>:$else_branch
    );

    let assemblyFormat = [{
        `then_branch` `:` $then_branch
        `else_branch` `:` $else_branch
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let hasVerifier = 1;
}

def IE_LoopOp :
        IE_LayerOp<
            "Loop",
            [
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                                ["inferReturnTypeComponents"]>,
                SingleBlockImplicitTerminator<"LoopTerminatorOp">,
                RecursiveMemoryEffects
            ]
        > {
    let summary = "InferenceEngine Loop layer";

    let description = [{
    Loop layer performs recurrent execution of the network, which is described in the body, iterating through the data along axis if given.
    Besides, it has current_iter and execution_cond params to determine if the loop go on or terminate.

    Inputs and Output are operands and results of Loop Op, i.e. op input and op output.
    Specially, for Loop Op, its inputs is: [trip_cpunt, condition, input1,  input2,  input3, ... ]
    trip_count: max number of iteration, usually connected to a constant node, but can be a uncertain parameter
    condition: the boolean to decide whether this loop will be executed or not in the first iteration

    The region body_module is the network which will be executed repeatedly by Loop.
    It has own input and output, i.e. body input and body output.

    Attributes:
    num_iterations is the number of iterations, which is inferred according to given trip_count(max number of iteration), execution_conditon
    (cosntant true or false will lead to a certain number of iterations) at it was built by Openvino.

    current_iter_index is the index of one special param of body input.
    It is the current iteration. If it is set as -1, that means there is no such param.

    exec_cond_index is the index of one special result of body output.
    It is the next iteration execution condition. It is required, cannot be -1 (i.e. unset).

    descs is a vector of descriptions on `op I/O index` and `body I/O index` mappings.
    For example, invariant_input_descs contains one or more IE::InvariantInputPortMapAttr attributes.
    Given one IE::InvariantInputPortMapAttr with externalPortId = 3 and internalLayerId = 5, it
    means op input[3] will be transformed to body input[5].
    All of these descs have externalPortId and internalLayerId as mapping.
    According to different processings on I/O, there are 5 kinds of descs below.

    slice_input_descs: attrs inside are IE::SliceInputPortMapAttr.
    It has axis, begin, end, stride, part_size, to help slice the op input along the axis
    from begin to end into part_size as body input while stride being +1 or -1 decides the direction.
    For example, IE.SliceInputPortMap<
    external_port_id = 2 : i64,
    internal_layer_id = 0 : i64,
    axis = 1 : i64,
    start = 0 : i64,
    stride = 1 : i64,
    part_size = 1 : i64,
    end = 2 : i64>
    Will take op input[2] and slice it along axis 1 from index 0 until index 2 into (end - start)/part_size = 2 pieces.
    And the first piece will be body input[0] in the first iteration.
    The second one will be body input[0] in the second iteration.

    concat_output_descs: attrs inside are IE::ConcatOutputPortMapAttr.
    It has axis, begin, end, stride, part_size, to help concat the body output along the axis
    into op output while stride being +1 or -1 decides the direction.

    invariant_input_descs: attrs inside are IE::InvariantInputPortMapAttr.
    It will map op input to body input in every iteration.

    invariant_output_descs: attrs inside are IE::InvariantOutputPortMapAttr.
    It will map body output to op output in the last iteration.

    feedback_input_descs: attrs inside are IE::MergedInputPortMapAttr.
    It describes how body output will be reused as body input after the first iteration.
    It has the body_input_index to mapping the body output[body_input_index] to body input[internalLayerId]
    after the first iteration.
    }];

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,
        IntAttr:$num_iterations,
        IntAttr:$current_iter_index,
        IntAttr:$exec_cond_index,
        OptionalAttr<ArrayAttr>:$slice_input_descs,
        OptionalAttr<ArrayAttr>:$invariant_input_descs,
        OptionalAttr<ArrayAttr>:$feedback_input_descs,
        OptionalAttr<ArrayAttr>:$concat_output_descs,
        OptionalAttr<ArrayAttr>:$invariant_output_descs
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$output
    );

    let regions = (region
        SizedRegion<1>:$body_module
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results) `\n`
        `(` `num_iterations` `:` $num_iterations `current_iter_index` `:` $current_iter_index `exec_cond_index` `:` $exec_cond_index  `)` `\n`
        `slice_input_descs` `:` $slice_input_descs `\n`
        `invariant_input_descs` `:` $invariant_input_descs `\n`
        `feedback_input_descs` `:` $feedback_input_descs `\n`
        `concat_output_descs` `:` $concat_output_descs `\n`
        `invariant_output_descs` `:` $invariant_output_descs `\n`
        `body_module` `:` $body_module
    }];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def IE_LoopSelectOp :
        IE_LayerOp<
            "LoopSelect"
        > {
    let summary = "Select a slice of the input according to collected execution conditions, helping implement Loop Op.";

    let arguments = (ins
        1DTensorOf<[Bool8, SI8]>:$initExecCond,
        1DTensorOf<[Bool8, SI8]>:$execConds,
        AnyRankedTensor:$input,

        BoolAttr:$do_concat,
        IntAttr:$axis,
        IntAttr:$stride
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
}

def IE_LoopTerminatorOp :
        IE_Op<
            "LoopTerminator",
            [
                ParentOneOf<["TensorIteratorOp", "LoopOp"]>,
                DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
                Pure,
                Terminator
            ]
        > {
    let summary = "Terminator for wrapping TensorIterator or Loop body module";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$operands
    );

    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

    let hasVerifier = 1;
}

def IE_TensorIteratorOp :
        IE_LayerOp<
            "TensorIterator",
            [
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                                ["inferReturnTypeComponents"]>,
                SingleBlockImplicitTerminator<"LoopTerminatorOp">,
                RecursiveMemoryEffects
            ]
        > {
    let summary = "InferenceEngine Tensor Iterator layer";

    let description = [{
    TensorIterator layer performs recurrent execution of the network, which is described in the body, iterating through the data along axis if given.
    }];

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,
        IntAttr:$num_iterations,
        OptionalAttr<ArrayAttr>:$slice_input_descs,
        OptionalAttr<ArrayAttr>:$invariant_input_descs,
        OptionalAttr<ArrayAttr>:$feedback_input_descs,
        OptionalAttr<ArrayAttr>:$concat_output_descs,
        OptionalAttr<ArrayAttr>:$invariant_output_descs
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$output
    );

    let regions = (region
        SizedRegion<1>:$body_module
    );

    let assemblyFormat = [{
        `body_module` `:` $body_module `\n`
        `num_iterations` `:` $num_iterations `\n`
        `slice_input_descs` `:` $slice_input_descs `\n`
        `invariant_input_descs` `:` $invariant_input_descs `\n`
        `feedback_input_descs` `:` $feedback_input_descs `\n`
        `concat_output_descs` `:` $concat_output_descs `\n`
        `invariant_output_descs` `:` $invariant_output_descs `\n`
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let hasVerifier = 1;
}

def IE_YieldOp :
        IE_Op<
            "Yield",
            [
                HasParent<"IfOp">,
                DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
                Pure,
                Terminator
            ]
        > {
    let summary = "Terminator for wrapping operation";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$operands
    );

    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

    let hasVerifier = 1;
}

#endif

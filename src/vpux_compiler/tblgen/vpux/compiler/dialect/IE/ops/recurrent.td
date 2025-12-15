//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_RECURRENT
#define VPUX_COMPILER_DIALECT_IE_OPS_RECURRENT

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_CTCGreedyDecoderOp :
        IE_LayerOp<
            "CTCGreedyDecoder"
        > {
    let summary = "InferenceEngine CTCGreedyDecoder layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        RankedTensorOf<[F16, F32, F64]>:$sequenceLengths,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_CTCGreedyDecoderSeqLenOp :
        IE_LayerOp<
            "CTCGreedyDecoderSeqLen"
        > {
    let summary = "InferenceEngine CTCGreedyDecoderSeqLen layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        RankedTensorOf<[SI32, SI64]>:$sequenceLength,
        Optional<RankedTensorOf<[SI32, SI64]>>:$blankIndex,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[SI32, SI64]>:$output,
        RankedTensorOf<[SI32, SI64]>:$outputLength
    );
}

def IE_GRUCellOp :
        IE_LayerOp<
            "GRUCell"
        > {
    let summary = "InferenceEngine GRUCell layer";

    let arguments = (ins
        2DTensorOf<[F16, F32, F64]>:$input_data,
        2DTensorOf<[F16, F32, F64]>:$initial_hidden_state,
        2DTensorOf<[F16, F32, F64]>:$weights,
        2DTensorOf<[F16, F32, F64]>:$recurrence_weights,
        Optional<1DTensorOf<[F16, F32, F64]>>:$biases,

        IntAttr:$hidden_size,
        UnitAttr:$should_linear_before_reset,
        F64Attr:$clip
    );

    let results = (outs
        2DTensorOf<[F16, F32, F64]>:$output_hidden_state
    );
}

def IE_GRUGatesOp :
        IE_LayerOp<
            "GRUGates"
        > {
    let summary = "Computes GRU activation functions";

    let description = [{
        This operation is intended to be run as a software stage after computing and adding GRU matrix multiplications.
        Formula is just for Gru with should_linear_before_reset true.
            ```
            -**input_data** = (inputData * weights + bias) [batchSize, 3 * hiddenSize]** or **[1, 1, batchSize, 3 * hiddenSize]**
            -**initial_hidden_state** =  [batchSize, hiddenSize]** or **[1, 1, batchSize, hiddenSize]**
            -**hidden_data** =  (initialHiddenState * recurrenceWeights) [batchSize, 3 * hiddenSize]** or **[1, 1, batchSize, 3 * hiddenSize]**
            -**biases** =  Rbh, 4't bias plan if exist (optional) [1, hiddenSize]** or **[1, 1, 1, hiddenSize]**
            * - Matrix multiplication
            + - Element-wise add
            ```
        - The meaning of other operands are identical to those in GRUCell operation.
    }];

    let arguments = (ins
        AnyTypeOf<[2DTensorOf<[F16, F32, F64]>, 4DTensorOf<[F16, F32, F64]>]>:$input_data,
        AnyTypeOf<[2DTensorOf<[F16, F32, F64]>, 4DTensorOf<[F16, F32, F64]>]>:$initial_hidden_state,
        AnyTypeOf<[2DTensorOf<[F16, F32, F64]>, 4DTensorOf<[F16, F32, F64]>]>:$hidden_data,
        Optional<AnyTypeOf<[1DTensorOf<[F16, F32, F64]>, 4DTensorOf<[F16, F32, F64]>]>>:$biases
    );

    let results = (outs
        AnyTypeOf<[2DTensorOf<[F16, F32, F64]>, 4DTensorOf<[F16, F32, F64]>]>:$output_hidden_state
    );
}

def IE_GRUSequenceOp :
        IE_LayerOp<
            "GRUSequence"
        > {
    let summary = "InferenceEngine GRUSequence layer";

    let arguments = (ins
        3DTensorOf<[F16, F32, F64]>:$input_data,
        3DTensorOf<[F16, F32, F64]>:$initial_hidden_state,
        3DTensorOf<[F16, F32, F64]>:$weights,
        3DTensorOf<[F16, F32, F64]>:$recurrence_weights,
        2DTensorOf<[F16, F32, F64]>:$biases,

        IntAttr:$hidden_size,
        IntAttr:$seq_length,
        IE_RNNSequenceDirectionAttr:$direction,
        UnitAttr:$should_linear_before_reset,
        F64Attr:$clip
    );

    let results = (outs
        4DTensorOf<[F16, F32, F64]>:$middle_hidden_state,
        3DTensorOf<[F16, F32, F64]>:$output_hidden_state
    );
}

def IE_GRUSequenceLastPartOp :
        IE_LayerOp<
            "GRUSequenceLastPart"
        > {
    let summary = "GRUSequence can be split into two parts, GRUSequenceFirstPart and GRUSequenceLastPart.";

    let description = [{
        GRUSequenceOp has five inputs and two outputs, and it can be split into GRUSequenceFirstPart and
        GRUSequenceLastPart. The calculation method of GRUSequenceFirstPart is similar to matrix multiply,
        and GRUSequenceLastPart will use output of GRUSequenceFirstPart and other inputs of GRUSequence to
        get outputs of GRUSequence.

        The split solution can be used for performance improvement since GRUSequenceFirstPart can be converted
        to HW convolution to utilize hardware feature of NPU, and can be used to meet CMX requirement when
        feasible tiling strategy can't be generated.
    }];

    let arguments = (ins
        4DTensorOf<[F16, F32, F64]>:$first_part_output,
        3DTensorOf<[F16, F32, F64]>:$initial_hidden_state,
        3DTensorOf<[F16, F32, F64]>:$recurrence_weights,
        2DTensorOf<[F16, F32, F64]>:$biases,

        IntAttr:$hidden_size,
        IntAttr:$seq_length,
        IE_RNNSequenceDirectionAttr:$direction,
        UnitAttr:$should_linear_before_reset,
        F64Attr:$clip
    );

    let results = (outs
        4DTensorOf<[F16, F32, F64]>:$middle_hidden_state,
        3DTensorOf<[F16, F32, F64]>:$output_hidden_state
    );
}

def IE_LSTMCellOp :
        IE_LayerOp<
            "LSTMCell",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine LSTMCell layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$inputData,
        RankedTensorOf<[F16, F32, F64]>:$initialHiddenState,
        RankedTensorOf<[F16, F32, F64]>:$initialCellState,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$weights,
        RankedTensorOf<[F16, F32, F64]>:$recurrenceWeights,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$biases,

        IntAttr:$hiddenSize
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$outputHiddenState,
        RankedTensorOf<[F16, F32, F64]>:$outputCellState
    );
}

def IE_LSTMGatesOp :
        IE_LayerOp<
            "LSTMGates"
        > {
    let summary = "Computes LSTM activation functions";

    let description = [{
        This operation is intended to be run as a software stage after computing and adding LSTM matrix multiplications.

        - **gatesInput** - tensor of shape **[batchSize, 4 * hiddenSize]** or **[1, 1, batchSize, 4 * hiddenSize]**. Formula:
            ```
            gatesInput = (inputData * weights) + (initialHiddenState * recurrenceWeights) + biases
            * - Matrix multiplication
            + - Element-wise add
            ```
        - The meaning of other operands are identical to those in LSTMCell operation.
    }];

    let arguments = (ins
        AnyTypeOf<[2DTensorOf<[F16, F32, F64]>, 4DTensorOf<[F16, F32, F64]>]>:$gatesInput,
        AnyTypeOf<[2DTensorOf<[F16, F32, F64]>, 4DTensorOf<[F16, F32, F64]>]>:$initialCellState
    );

    let results = (outs
        AnyTypeOf<[2DTensorOf<[F16, F32, F64]>, 4DTensorOf<[F16, F32, F64]>]>:$outputHiddenState,
        AnyTypeOf<[2DTensorOf<[F16, F32, F64]>, 4DTensorOf<[F16, F32, F64]>]>:$outputCellState
    );

    let hasVerifier = 1;
}

def IE_LSTMSequenceOp :
        IE_LayerOp<
            "LSTMSequence",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "InferenceEngine LSTMSequence layer";

    let description = [{
        This operation is for InferenceEngine LSTMSequence layer:

          * The sequenceLength attribute is not mandatory, which allows us to support dynamic
            shape cases where the sequenceLength is initially unknown. In such cases, we will
            derive the necessary information from the inputData and keep the sequenceLength
            attribute empty.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$inputData,
        RankedTensorOf<[F16, F32, F64]>:$initialHiddenState,
        RankedTensorOf<[F16, F32, F64]>:$initialCellState,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$weights,
        RankedTensorOf<[F16, F32, F64]>:$reccurenceWeights,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$biases,

        OptionalAttr<IntAttr>:$sequenceLength,
        IE_RNNSequenceDirectionAttr:$direction
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$outputHiddenValues,
        RankedTensorOf<[F16, F32, F64]>:$outputHiddenState,
        RankedTensorOf<[F16, F32, F64]>:$outputCellState
    );
}


#endif

//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_RECURRENT
#define VPUX_COMPILER_DIALECT_VPU_OPS_RECURRENT

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"



def VPU_CTCGreedyDecoderOp :
        VPU_LayerOp<
            "CTCGreedyDecoder"
        > {
    let summary = "CTCGreedyDecoder VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$sequenceLengths,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_CTCGreedyDecoderSeqLenOp :
        VPU_LayerOp<
            "CTCGreedyDecoderSeqLen"
        > {
    let summary = "CTCGreedyDecoderSeqLen VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[SI32]>:$sequenceLength,
        Optional<RankedTensorOf<[SI32]>>:$blankIndex,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[SI32]>:$output,
        RankedTensorOf<[SI32]>:$outputLength
    );
}


def VPU_GRUGatesOp :
        VPU_LayerOp<
            "GRUGates",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
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
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input_data,
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$initial_hidden_state,
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$hidden_data,
        Optional<AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$biases,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$outputHiddenState
    );


    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input_data,
            "::mlir::Value":$initial_hidden_state,
            "::mlir::Value":$hidden_data,
            "::mlir::Value":$biases
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_GRUSequenceOp :
        VPU_LayerOp<
            "GRUSequence",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "GRUSequence VPU layer";

    let arguments = (ins
        3DTensorOf<[F16, F32]>:$input_data,
        3DTensorOf<[F16, F32]>:$initial_hidden_state,
        3DTensorOf<[F16, F32]>:$weights,
        3DTensorOf<[F16, F32]>:$recurrence_weights,
        2DTensorOf<[F16, F32]>:$biases,

        IntAttr:$hidden_size,
        IntAttr:$seq_length,
        IE_RNNSequenceDirectionAttr:$direction,
        UnitAttr:$should_linear_before_reset,
        F64Attr:$clip
    );

    let results = (outs
        4DTensorOf<[F16, F32]>:$middle_hidden_state,
        3DTensorOf<[F16, F32]>:$output_hidden_state
    );

    let extraClassDeclaration = [{
        OutputTiling getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger log);
    }] # baseExtraClassDeclaration;

    let checkInferredMemSpace = 0;
}


def VPU_GRUSequenceFirstPartOp :
        VPU_LayerOp<
            "GRUSequenceFirstPart"
        > {
    let summary = "GRUSequenceFirstPart VPU layer";

    let arguments = (ins
        3DTensorOf<[F16, F32]>:$input_data,
        3DTensorOf<[F16, F32]>:$weights,

        IntAttr:$hidden_size,
        IntAttr:$seq_length,
        F64Attr:$clip
    );

    let results = (outs
        4DTensorOf<[F16, F32]>:$output
    );
}


def VPU_GRUSequenceLastPartOp :
        VPU_LayerOp<
            "GRUSequenceLastPart",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "GRUSequenceLastPart VPU layer";

    let arguments = (ins
        4DTensorOf<[F16, F32]>:$first_part_output,
        3DTensorOf<[F16, F32]>:$initial_hidden_state,
        3DTensorOf<[F16, F32]>:$recurrence_weights,
        2DTensorOf<[F16, F32]>:$biases,

        IntAttr:$hidden_size,
        IntAttr:$seq_length,
        IE_RNNSequenceDirectionAttr:$direction,
        UnitAttr:$should_linear_before_reset,
        F64Attr:$clip
    );

    let results = (outs
        4DTensorOf<[F16, F32]>:$middle_hidden_state,
        3DTensorOf<[F16, F32]>:$output_hidden_state
    );

    let extraClassDeclaration = [{
        OutputTiling getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger log);
    }] # baseExtraClassDeclaration;

    let checkInferredMemSpace = 0;
}


def VPU_LSTMCellOp :
        VPU_LayerOp<
            "LSTMCell"
        > {
    let summary = "LSTMCell VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$inputData,
        RankedTensorOf<[F16, F32]>:$initialHiddenState,
        RankedTensorOf<[F16, F32]>:$initialCellState,
        RankedTensorOf<[F16, F32]>:$weights,
        RankedTensorOf<[F16, F32]>:$recurrenceWeights,
        RankedTensorOf<[F16, F32]>:$biases,

        IntAttr:$hiddenSize
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$outputHiddenState,
        RankedTensorOf<[F16, F32]>:$outputCellState
    );

    let extraClassDeclaration = [{
        static bool isSupported(vpux::IE::LSTMCellOp op);
    }] # baseExtraClassDeclaration;


}


def VPU_LSTMGatesOp :
        VPU_LayerOp<
            "LSTMGates",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
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
        AnyTypeOf<[2DTensorOf<[F16, F32]>, 4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$gatesInput,
        AnyTypeOf<[2DTensorOf<[F16, F32]>, 4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$initialCellState,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[2DTensorOf<[F16, F32]>, 4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$outputHiddenState,
        AnyTypeOf<[2DTensorOf<[F16, F32]>, 4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$outputCellState
    );

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$gatesInput,
            "::mlir::Value":$initialCellState
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);

        OutputTiling getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger log);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_LSTMSequenceOp :
        VPU_LayerOp<
            "LSTMSequence",
        [
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverKernelCompatible",
                                                                     "isOperationSplitOverBatchCompatible"]>
        ]
        > {
    let summary = "LSTMSequence VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$inputData,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$initialHiddenState,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$initialCellState,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$reccurenceWeights,
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$biases,
        AnyTypeOf<[RankedTensorOf<[SI32]>, VPU_DistributedTensor]>:$syncBuffer,

        OptionalAttr<IntAttr>:$sequenceLength,
        IE_RNNSequenceDirectionAttr:$direction,
        OptionalAttr<BoolAttr>:$useDpu,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$outputHiddenValues,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$outputHiddenState,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$outputCellState
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$inputData,
            "::mlir::Value":$initialHiddenState,
            "::mlir::Value":$initialCellState,
            "::mlir::Value":$reccurenceWeights,
            "::mlir::Value":$biases,
            "::mlir::IntegerAttr":$sequenceLength,
            "vpux::IE::RNNSequenceDirectionAttr":$direction,
            "::mlir::BoolAttr":$useDpu,
            "vpux::VPU::MultiClusterStrategyAttr":$multiClusterStrategy
        )>,
        OpBuilder<(ins
            "::mlir::Value":$inputData,
            "::mlir::Value":$initialHiddenState,
            "::mlir::Value":$initialCellState,
            "::mlir::Value":$reccurenceWeights,
            "::mlir::Value":$biases,
            "::mlir::IntegerAttr":$sequenceLength,
            "vpux::IE::RNNSequenceDirectionAttr":$direction,
            "vpux::VPU::MultiClusterStrategyAttr":$multiClusterStrategy
        )>
    ];

    let extraClassDeclaration = [{
        static bool isSupported(vpux::IE::LSTMSequenceOp origOp, bool useDpu=false);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

#endif

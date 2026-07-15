//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_ELTWISE
#define VPUX_COMPILER_DIALECT_IE_OPS_ELTWISE

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_AddOp :
        IE_LayerOp<
            "Add",
            [
                Commutative,
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface, ["shouldJITCompile"]>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
                DeclareOpInterfaceMethods<StaticShapeOpInterface>
            ]
        > {
    let summary = "InferenceEngine Add layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$input2,
        // Placeholder for future scales-as-input support
        Optional<AnyRankedTensor>:$scales,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOpAttrInterface>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$output
    );

    // Backward-compatible builders: omit the reserved `scales` operand (passed as null).
    // Use the full generated builder to supply a scales tensor when support is added.
    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            CArg<"vpux::IE::PostOpAttr", "{}">:$post_op,
            CArg<"mlir::DictionaryAttr", "{}">:$clamp,
            CArg<"mlir::ArrayAttr", "{}">:$output_padding,
            CArg<"mlir::ArrayAttr", "{}">:$input_padding
        ), [{
            build($_builder, $_state, input1, input2, /*scales=*/::mlir::Value{},
                  auto_broadcast, post_op, clamp, output_padding, input_padding);
        }]>,
        OpBuilder<(ins
            "::mlir::Type":$result_type,
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            CArg<"vpux::IE::PostOpAttr", "{}">:$post_op,
            CArg<"mlir::DictionaryAttr", "{}">:$clamp,
            CArg<"mlir::ArrayAttr", "{}">:$output_padding,
            CArg<"mlir::ArrayAttr", "{}">:$input_padding
        ), [{
            build($_builder, $_state, result_type, input1, input2, /*scales=*/::mlir::Value{},
                  auto_broadcast, post_op, clamp, output_padding, input_padding);
        }]>,
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastType":$auto_broadcast,
            CArg<"vpux::IE::PostOpAttr", "{}">:$post_op,
            CArg<"mlir::DictionaryAttr", "{}">:$clamp,
            CArg<"mlir::ArrayAttr", "{}">:$output_padding,
            CArg<"mlir::ArrayAttr", "{}">:$input_padding
        ), [{
            build($_builder, $_state, input1, input2, /*scales=*/::mlir::Value{},
                  auto_broadcast, post_op, clamp, output_padding, input_padding);
        }]>,
        OpBuilder<(ins
            "::mlir::Type":$result_type,
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastType":$auto_broadcast,
            CArg<"vpux::IE::PostOpAttr", "{}">:$post_op,
            CArg<"mlir::DictionaryAttr", "{}">:$clamp,
            CArg<"mlir::ArrayAttr", "{}">:$output_padding,
            CArg<"mlir::ArrayAttr", "{}">:$input_padding
        ), [{
            build($_builder, $_state, result_type, input1, input2, /*scales=*/::mlir::Value{},
                  auto_broadcast, post_op, clamp, output_padding, input_padding);
        }]>
    ];

    let hasVerifier = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION];
}

def IE_DivideOp :
        IE_LayerOp<
            "Divide",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "InferenceEngine Divide layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64, UI8, UI16, UI32, UI64]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, SI64, UI8, UI16, UI32, UI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, SI32, SI64, UI8, UI16, UI32, UI64]>:$output
    );
}

def IE_FloorModOp :
        IE_LayerOp<
            "FloorMod",
            [
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "InferenceEngine FloorMod layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_MaximumOp :
        IE_LayerOp<
            "Maximum",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "InferenceEngine Maximum layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_MinimumOp :
        IE_LayerOp<
            "Minimum",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "InferenceEngine Minimum layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_ModOp :
        IE_LayerOp<
            "Mod",
            [
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "InferenceEngine Mod layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_MultiplyOp :
        IE_LayerOp<
            "Multiply",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
                DeclareOpInterfaceMethods<StaticShapeOpInterface, ["requiresStaticShape"]>,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Multiply layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$input2,
        // Placeholder for future scales-as-input support
        Optional<AnyRankedTensor>:$scales,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOpAttrInterface>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$output
    );

    // Backward-compatible builders: omit the reserved `scales` operand (passed as null).
    // Use the full generated builder to supply a scales tensor when support is added.
    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            CArg<"vpux::IE::PostOpAttr", "{}">:$post_op,
            CArg<"mlir::DictionaryAttr", "{}">:$clamp,
            CArg<"mlir::ArrayAttr", "{}">:$output_padding,
            CArg<"mlir::ArrayAttr", "{}">:$input_padding
        ), [{
            build($_builder, $_state, input1, input2, /*scales=*/::mlir::Value{},
                  auto_broadcast, post_op, clamp, output_padding, input_padding);
        }]>,
        OpBuilder<(ins
            "::mlir::Type":$result_type,
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            CArg<"vpux::IE::PostOpAttr", "{}">:$post_op,
            CArg<"mlir::DictionaryAttr", "{}">:$clamp,
            CArg<"mlir::ArrayAttr", "{}">:$output_padding,
            CArg<"mlir::ArrayAttr", "{}">:$input_padding
        ), [{
            build($_builder, $_state, result_type, input1, input2, /*scales=*/::mlir::Value{},
                  auto_broadcast, post_op, clamp, output_padding, input_padding);
        }]>,
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastType":$auto_broadcast,
            CArg<"vpux::IE::PostOpAttr", "{}">:$post_op,
            CArg<"mlir::DictionaryAttr", "{}">:$clamp,
            CArg<"mlir::ArrayAttr", "{}">:$output_padding,
            CArg<"mlir::ArrayAttr", "{}">:$input_padding
        ), [{
            build($_builder, $_state, input1, input2, /*scales=*/::mlir::Value{},
                  auto_broadcast, post_op, clamp, output_padding, input_padding);
        }]>,
        OpBuilder<(ins
            "::mlir::Type":$result_type,
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastType":$auto_broadcast,
            CArg<"vpux::IE::PostOpAttr", "{}">:$post_op,
            CArg<"mlir::DictionaryAttr", "{}">:$clamp,
            CArg<"mlir::ArrayAttr", "{}">:$output_padding,
            CArg<"mlir::ArrayAttr", "{}">:$input_padding
        ), [{
            build($_builder, $_state, result_type, input1, input2, /*scales=*/::mlir::Value{},
                  auto_broadcast, post_op, clamp, output_padding, input_padding);
        }]>
    ];

    let hasVerifier = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION];
}

def IE_PowerOp :
        IE_LayerOp<
            "Power",
            [
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "InferenceEngine Power layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64, UI8, UI16, UI32, UI64]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, SI64, UI8, UI16, UI32, UI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, SI32, SI64, UI8, UI16, UI32, UI64]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

def IE_ScaleShiftOp :
        IE_LayerOp<
            "ScaleShift",
            [
                AttrSizedOperandSegments,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine ScaleShift layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$weights,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$biases
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasCanonicalizer = 1;
}

def IE_SquaredDifferenceOp :
        IE_LayerOp<
            "SquaredDiff",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine SquaredDiff layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$output
    );
}

def IE_SubtractOp :
        IE_LayerOp<
            "Subtract",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "InferenceEngine Subtract layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI8, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI8, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$input2,
        // Placeholder for future scales-as-input support
        Optional<AnyRankedTensor>:$scales,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOpAttrInterface>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, SI8, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$output
    );

    // Backward-compatible builders: omit the reserved `scales` operand (passed as null).
    // Use the full generated builder to supply a scales tensor when support is added.
    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            CArg<"vpux::IE::PostOpAttr", "{}">:$post_op,
            CArg<"mlir::DictionaryAttr", "{}">:$clamp,
            CArg<"mlir::ArrayAttr", "{}">:$output_padding,
            CArg<"mlir::ArrayAttr", "{}">:$input_padding
        ), [{
            build($_builder, $_state, input1, input2, /*scales=*/::mlir::Value{},
                  auto_broadcast, post_op, clamp, output_padding, input_padding);
        }]>,
        OpBuilder<(ins
            "::mlir::Type":$result_type,
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            CArg<"vpux::IE::PostOpAttr", "{}">:$post_op,
            CArg<"mlir::DictionaryAttr", "{}">:$clamp,
            CArg<"mlir::ArrayAttr", "{}">:$output_padding,
            CArg<"mlir::ArrayAttr", "{}">:$input_padding
        ), [{
            build($_builder, $_state, result_type, input1, input2, /*scales=*/::mlir::Value{},
                  auto_broadcast, post_op, clamp, output_padding, input_padding);
        }]>,
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastType":$auto_broadcast,
            CArg<"vpux::IE::PostOpAttr", "{}">:$post_op,
            CArg<"mlir::DictionaryAttr", "{}">:$clamp,
            CArg<"mlir::ArrayAttr", "{}">:$output_padding,
            CArg<"mlir::ArrayAttr", "{}">:$input_padding
        ), [{
            build($_builder, $_state, input1, input2, /*scales=*/::mlir::Value{},
                  auto_broadcast, post_op, clamp, output_padding, input_padding);
        }]>,
        OpBuilder<(ins
            "::mlir::Type":$result_type,
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastType":$auto_broadcast,
            CArg<"vpux::IE::PostOpAttr", "{}">:$post_op,
            CArg<"mlir::DictionaryAttr", "{}">:$clamp,
            CArg<"mlir::ArrayAttr", "{}">:$output_padding,
            CArg<"mlir::ArrayAttr", "{}">:$input_padding
        ), [{
            build($_builder, $_state, result_type, input1, input2, /*scales=*/::mlir::Value{},
                  auto_broadcast, post_op, clamp, output_padding, input_padding);
        }]>
    ];

    let hasVerifier = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION];
}

#endif

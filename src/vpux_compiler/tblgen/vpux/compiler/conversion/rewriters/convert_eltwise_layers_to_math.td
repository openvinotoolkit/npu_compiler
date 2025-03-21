//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_ELTWISE_LAYERS_TO_MATH
#define VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_ELTWISE_LAYERS_TO_MATH

include "vpux/compiler/dialect/IE/ops.td"
include "mlir/Dialect/Math/IR/MathOps.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

//
// IE.Cos -> math.Cos
//

def createCosOp :
        NativeCodeCall<[{
            $_builder.create<mlir::math::CosOp>($_loc, $0)
        }]>;

def RewriteCos :
        Pat<
            (IE_CosOp $input),
            (createCosOp $input)
        >;

#endif

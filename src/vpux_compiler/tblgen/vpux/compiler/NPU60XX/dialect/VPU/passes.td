//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_ARCH_60XX_PASSES
#define VPUX_COMPILER_DIALECT_VPU_ARCH_60XX_PASSES

include "mlir/Pass/PassBase.td"

//
// ExtractReduceMaxFromExpStabilized
//

def ExtractReduceMaxFromExpStabilized : PassBase<"extract-reduce-max-from-exp-stabilized", "vpux::FunctionPass"> {
	let summary = "Extract ReduceMax from a stabilized SoftMax and fuse it into the parent NCE.Convolution";

	let description = [{
		This pass extracts `ReduceMax` from a stabilized `SoftMax` and fuses it into the parent `VPU.NCE.Convolution`.
		When the SoftMax carries a non-zero `pad_size`, rebuilds the convolution's weight-table scale/bias
		so that the trailing padded output channels contribute `fp32min` (scale = 1.0, bias = fp32min),
		which neutralizes those channels in the fused DPU MAX computation so the ReduceMax result ignores them.
	}];

	let constructor = "vpux::VPU::arch60xx::createExtractReduceMaxFromExpStabilizedPass()";

	let dependentDialects = [
		"vpux::VPU::VPUDialect"
	];
}

#endif

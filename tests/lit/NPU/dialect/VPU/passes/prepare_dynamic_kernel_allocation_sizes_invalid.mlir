//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile_Interpreter" --prepare-dynamic-kernel-allocation-sizes --verify-diagnostics %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// expected-error@+1 {{Unexpected op between VPU.Empty and its consumer}}
module @PrepareDynamicKernelAllocationSizesInvalid {
    func.func @PrepareDynamicKernelAllocationSizesInvalid(%input: tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>, %scales: tensor<2xf16>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}> {
        %aux = VPU.Empty : tensor<1x1x1x1324212xui8>
        %input_cast = VPU.Convert(%input) {dstElemType = ui8} : tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>

        %0 = VPU.InterpolateDMA(%input_cast, %scales, %aux) {
            attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [2, 3]
        } : tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>, tensor<2xf16>, tensor<1x1x1x1324212xui8> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}>
        return %0 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}>
    }
}

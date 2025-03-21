//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_OPS
#define VPUX_COMPILER_DIALECT_VPUIPDPU_OPS

include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIPDPU/attributes.td"
include "vpux/compiler/dialect/VPUIPDPU/dialect.td"
include "vpux/compiler/dialect/VPUIPDPU/ops_interfaces.td"

//
// Base classes
//

class VPUIPDPU_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPUIPDPU_Dialect,
            mnemonic,
            !listconcat(
                traits,
                [
                    DefinedInArch<"vpux::VPU::ArchKind::NPU37XX">
                ]
            )
        >;

//
// IDUInActivations
//

def VPUIPDPU_IDUInActivationsOp :
        VPUIPDPU_Op<"IDUInActivations",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">
            ]
        > {
    let summary = "IDU Input Activations configuration";

    let arguments =(ins
        AnyMemRef:$in_activations,
        UnitAttr:$in_sparse
        );

    let assemblyFormat = [{
        `in_activations`    `(` $in_activations `:` type($in_activations) `)`
        attr-dict
    }];
}

//
// IDUWeights37XX
//

def VPUIPDPU_IDUWeights37XXOp :
        VPUIPDPU_Op<"IDUWeights37XX",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">,
                LimitedToArch<["vpux::VPU::ArchKind::NPU37XX"]>
            ]
        > {
    let summary = "IDU Weight offset NPU37XX";

    let description = [{
wt_offset - This is added to the weight sparsity and weight data address
 read by the IDU and allows an offest to be specified from the addresses
 in the weight control pointer.

pool_wt_data - A custom constant weight that is not read from CMX.
This is required when doing Average pool, where the data is expected to be set to 1.
Setting this value will trigger pool_wt_rd_dis and disable reading of weights from CMX.
         }];

    let arguments =(ins
        TypeAttr:$wmode,
        OptionalAttr<IntAttr>:$pool_wt_data,
        UnitAttr:$wt_sparse
        );

    let assemblyFormat = [{
        `wmode`         `(` $wmode `)`
        (`pool_wt_data` `(` $pool_wt_data^ `)`)?
        attr-dict
    }];
}

//
// IDUInputLayerCfg
//

def VPUIPDPU_IDUInputLayerCfgOp :
        VPUIPDPU_Op<"IDUInputLayerCfg",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">
            ]
        > {
    let summary = "IDU input layer configuration";

    let description = [{
 A custom input layer sparsity pattern - this is used when KERNAL_PAD_CFG.LAYER1_WT_SP_INS=1
          }];

    let arguments =(ins
        IntAttr:$sparsity_pattern,
        UnitAttr:$input_compressed
        );

    let assemblyFormat = [{
        `sparsity_pattern`  `(` $sparsity_pattern `)`
        attr-dict
    }];
}

//
// IDUStorageElement
//

def VPUIPDPU_IDUStorageElementOp :
        VPUIPDPU_Op<"IDUStorageElement",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">
            ]
        > {
    let summary = "IDU storage elements config";

    let description = [{
        When se_size is a power of 2 the storage element size
        is configured via Z_CONIFG.SE_Z_SPLIT.
        Otherwise, storage element size can be any multiple
        of 16 activations and the storage element size is
        configured via TENSOR_SIZE1.SE_SIZE.
        }];

    let arguments =(ins
        IntAttr:$se_size,
        OptionalAttr<IntAttr>:$num_ses_in_z_dir
        );

    let assemblyFormat = [{
        `se_size`           `(` $se_size `)`
        (`num_ses_in_z_dir` `(` $num_ses_in_z_dir^ `)`)?
        attr-dict
    }];
}

//
// IDUKernel
//

def VPUIPDPU_IDUKernelOp :
        VPUIPDPU_Op<"IDUKernel",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">
            ]
        > {
    let summary = "IDU Kernel setting";

    let arguments =(ins
        IntAttr:$kernel_x,
        IntAttr:$kernel_y
        );

    let assemblyFormat = [{
        `kernel_x` `(` $kernel_x `)`
        `kernel_y` `(` $kernel_y `)`
        attr-dict
    }];
}

//
// IDUStride
//

def VPUIPDPU_IDUStrideOp :
        VPUIPDPU_Op<"IDUStride",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">
            ]
        > {
    let summary = "IDU Stride setting";

    let arguments =(ins
        IntAttr:$stride_x,
        IntAttr:$stride_y
        );

    let assemblyFormat = [{
        `stride_x` `(` $stride_x `)`
        `stride_y` `(` $stride_y `)`
        attr-dict
    }];
}


//
// IDUWorkloadCfg
//

def VPUIPDPU_IDUWorkloadCfgOp :
        VPUIPDPU_Op<"IDUWorkloadCfg",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">
            ]
        > {
    let summary = "IDU worload configuration";

    let arguments =(ins
        VPUIPDPU_IDUWorkloadType:$workload_type
        );

    let assemblyFormat = [{
        `workload_type`       `(` $workload_type `)`
        attr-dict
    }];
}

//
// IDUDepthWiseCfg
//

def VPUIPDPU_IDUDepthWiseCfgOp :
        VPUIPDPU_Op<"IDUDepthWiseCfg",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">
            ]
        > {
    let summary = "IDU depthwise configuration";

    let description = [{
dw_3x3s1_opt_dis - Set to disable the DW optimisation for kernel width 3, stride=1 workloads. Must be set to 1 (disabled) for Max Pool operations.
For WORKLOAD_SIZE_Z=64 using 8 bit activations and WORKLOAD_SIZE_Z=32 using 16 bit activations the following constraints must be adhered to:
(TENSOR_SIZE_Z%32)          == 0;
(WORKLOAD_START_Z%32)       == 0;
(TENSOR_START%32)           == 0;
If this is not possible, then this disable bit must be set and the legacy DW mode used instead.

dw_opt_offset - For DW workloads where KERNEL_SIZE = TENSOR_SIZE = WORKLOAD_SIZE and padding is disabled, a DPU optimisation can be enabled.  The optimisation requires that the weight data for each weight set is at a fixed offset from the previous weight set in CMX.  The offset between the start of weight data for each weight set is configured via DW_OPT_OFFSET.  Each unit of DW_OPT_OFFSET corresponds to 16B in CMX.
         }];

    let arguments =(ins
        BoolAttr:$dw_3x3s1_opt_dis,
        OptionalAttr<IntAttr>:$dw_opt_offset
        );

    let assemblyFormat = [{
        `dw_3x3s1_opt_dis`      `(` $dw_3x3s1_opt_dis `)`
        (`dw_opt_offset`   `(` $dw_opt_offset^ `)`)?
        attr-dict
    }];
}

//
// IDUEltWiseCfg
//

def VPUIPDPU_IDUEltWiseCfgOp :
        VPUIPDPU_Op<"IDUEltWiseCfg",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">
            ]
        > {
    let summary = "IDU eltwise configuration";

    let arguments =(ins
        UnitAttr:$bf16_flow_on,
        AnyAttrOf<[IntAttr, F32Attr]>:$elop_scale_a,
        AnyAttrOf<[IntAttr, F32Attr]>:$elop_scale_b
        );

    let hasVerifier = 1;

    let assemblyFormat = [{
        `elop_scale_a`  `(` $elop_scale_a `)`
        `elop_scale_b`  `(` $elop_scale_b `)`
        attr-dict
    }];
}

//
// IDUWorkloadSet
//

def VPUIPDPU_IDUWorkloadSetOp :
        VPUIPDPU_Op<"IDUWorkloadSet",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU workload settings";

    let arguments =(ins
        IntAttr:$start_x,
        IntAttr:$start_y,
        IntAttr:$start_z,
        IntAttr:$size_x,
        IntAttr:$size_y,
        IntAttr:$size_z
        );

    let assemblyFormat = [{
        `start_x`           `(` $start_x `)`
        `start_y`          `(` $start_y `)`
        `start_z`          `(` $start_z `)`
        `size_x`           `(` $size_x `)`
        `size_y`           `(` $size_y `)`
        `size_z`           `(` $size_z `)`

        attr-dict
    }];
}

//
// IDUWeightSet
//

def VPUIPDPU_IDUWeightSetOp :
        VPUIPDPU_Op<"IDUWeightSet",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU Weight settings";

    let description = [{
    From the memRef weights, following data should be extracted:
weight_start  - Address of list of weight sparsity and weight data pointers.  Each weight set has a weight sparsity and weight data pointer.
    The list of pointers is contiguous in memory.  The number of entries in the list is specified by WEIGHT_NUM.
weight_num    - Number of weight sets (i.e. Z depth of output tensor). Must be a multiple of 16.
weight_size   - Weight set size (Input Z * kernel X * kernel Y)

         }];

    let arguments =(ins
        IntAttr:$weight_start,
        IntAttr:$weight_num,
        IntAttr:$weight_size
        );

    let assemblyFormat = [{
        `weight_start`        `(` $weight_start `)`
        `weight_num`          `(` $weight_num `)`
        `weight_size`         `(` $weight_size `)`
        attr-dict
    }];
}

//
// IDUPadding
//

def VPUIPDPU_IDUPaddingOp :
        VPUIPDPU_Op<"IDUPadding",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU workload padding settings";

    let description = [{
         }];

    let arguments =(ins
        VPU_PaddingAttr:$pad_count
        );

    let assemblyFormat = [{
        `pad_count`    `(` $pad_count `)`
        attr-dict
    }];
}

//
// IDUActSwizzle
//

def VPUIPDPU_IDUActSwizzleOp :
        VPUIPDPU_Op<"IDUActSwizzle",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU Activations swizzle settings";

    let description = [{
swizzle_key - to be used for  only for activations (default swizzle key
    for weights is set to 0. Call IDUWeightSwizzleOp to set weight swizzle key)
         }];


    let arguments = (ins
        VPUIPDPU_DPUSwizzleKey:$swizzle_key
    );

    let assemblyFormat = [{
        `swizzle_key`          `(` $swizzle_key `)`
        attr-dict
    }];
}

//
// IDUWeightSwizzle
//

def VPUIPDPU_IDUWeightSwizzleOp :
        VPUIPDPU_Op<"IDUWeightSwizzle",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU Weight swizzle settings";

    let description = [{
        Weight swizzle is off for weights unless this op is called.
wt_swizzle_key - weights can use sepparate swizzle key from activations.

         }];


    let arguments = (ins
        VPUIPDPU_DPUSwizzleKey:$wt_swizzle_key
    );

    let assemblyFormat = [{
        `wt_swizzle_key`          `(` $wt_swizzle_key `)`
        attr-dict
    }];
}



//
// IDUNthwNtk
//

def VPUIPDPU_IDUNthwNtkOp :
        VPUIPDPU_Op<"IDUNthwNtk",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU NTHW/NTK reuse optimization settings";

    let description = [{
Reuse of activations and weights to reduce CMX read bandwidth. There are 3 modes of activation/weight processing
 based on how much activations and weights are to be re-cycled by the SCL.
 The level of reuse drops depending on the tensor size and the number of weight sets.
              |  Activations reuse   |  Weights reuse   |
--------------|----------------------|------------------|
NTHW_NTK_16_4 |         4            |         16       |
NTHW_NTK_8_8  |         8            |         8        |
NTHW_NTK_4_16 |         16           |         4        |
         }];


    let arguments = (ins
        VPUIPDPU_IDUNthwNtk:$nthw_ntk
    );

    let assemblyFormat = [{
        `nthw_ntk`         `(` $nthw_ntk `)`
        attr-dict
    }];
}

//
// IDUSEDense
//

def VPUIPDPU_IDUSEDenseOp :
        VPUIPDPU_Op<"IDUSEDense",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU set Storage Element table in dense mode";

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// IDUConvContinue
//

def VPUIPDPU_IDUConvContinueOp :
        VPUIPDPU_Op<"IDUConvContinue",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU set convolution continue mode";

    let description = [{
For partial convolutions, where all required activations/weights won't fit in CMX the IDU supports
 a convolution continue mode.
 In this mode the SCL does not push the result to the PPE at the end of the convolution,
 but instead stores the result as the starting point for the next convolution.
 In this mode workload size cannot be bigger than the grid size
 (i.e. 168 for NTHW/NTK = 8/8, 1616 for NTHW/NTK = 16/4, 8*4 for NTHW/NTK = 4/16)
         }];

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// IDUBinaryConfig
//

def VPUIPDPU_IDUBinaryConfigOp :
        VPUIPDPU_Op<"IDUBinaryConfig",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "IDU set Binary config";

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// IDUSESegment
//

def VPUIPDPU_IDUSESegmentOp :
        VPUIPDPU_Op<"IDUSESegment",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">,
                LimitedToArch<["vpux::VPU::ArchKind::NPU37XX"]>

            ]
        > {
    let summary = "IDU storage element segmentation config";

    let description = [{
        In sparse mode, se_seg_size_0/1 contains the size of each
        segment of the storage element table.Each unit corresponds to 1 storage entry (i.e. 4 bytes in CMX).
        Each segment of the SE table must contain a multiple of 4 pointers.
        Therefore SE_SEG_SIZE must be configured to be a multiple of 4.

        In dense mode contains the size of segments of the dense tensor.
        Each unit corresponds to four Z columns of input tensor activations.

        In dense_se mode (i.e. when KERNAL_PAD_CFG.DENSE_SE = 1 and KERNAL_PAD_CFG.ACT_DENSE = 0) contains the size of segments of the tensor.
        Each unit corresponds to four SE's of input tensor activations.

        In depth wise mode contains the size of segments of the dense tensor.
        Each unit corresponds to one Z column of input tensor activations (i.e. 16 Activation bytes).
        }];

    let arguments =(ins
        IntAttr:$se_seg_size_0,
        IntAttr:$se_seg_size_1,
        IntAttr:$se_seg_size_2
        );

    let assemblyFormat = [{
        `se_seg_size_0`   `(` $se_seg_size_0 `)`
        `se_seg_size_1`   `(` $se_seg_size_1 `)`
        `se_seg_size_2`   `(` $se_seg_size_2 `)`
        attr-dict
    }];
}

//
// IDUSPSegment
//

def VPUIPDPU_IDUSPSegmentOp :
        VPUIPDPU_Op<"IDUSPSegment",
            [
                HasParent<"vpux::VPUIPDPU::IDUCfgOp">,
                LimitedToArch<["vpux::VPU::ArchKind::NPU37XX"]>
            ]
        > {
    let summary = "IDU sparsity segmentation config";

    let description = [{
        Size of segment 0 and 1 of the sparsity table.
        Each unit corresponds to 128 bits of sparsity (i.e.16 bytes in CMX).
        }];

    let arguments =(ins
        IntAttr:$sp_seg_size_0,
        IntAttr:$sp_seg_size_1,
        IntAttr:$sp_seg_size_2
        );

    let assemblyFormat = [{
        `sp_seg_size_0`   `(` $sp_seg_size_0 `)`
        `sp_seg_size_1`   `(` $sp_seg_size_1 `)`
        `sp_seg_size_2`   `(` $sp_seg_size_2 `)`
        attr-dict
    }];
}

//
// MPEDenormalOperandsFTZ
//

def VPUIPDPU_MPEDenormalOperandsFTZOp :
        VPUIPDPU_Op<"MPEDenormalOperandsFTZ",
            [
                HasParent<"vpux::VPUIPDPU::MPECfgOp">
            ]
        > {
    let summary = "MPE force denormal operands to zero op";

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// MPEActivationBias
//

def VPUIPDPU_MPEActivationBiasOp :
        VPUIPDPU_Op<"MPEActivationBias",
            [
                HasParent<"vpux::VPUIPDPU::MPECfgOp">
            ]
        > {
    let summary = "MPE set activation bias for U8 or I8 input op";

    let description = [{
        Used for asymetric integer quantization.
        The bias value (zero point) is treated as U8 if input is U8 or I8 if input is I8.
        Thus the op will restrict the bias value as such.
        For all other data types it is disregarded.
        }];

    let arguments = (ins
        IntAttr:$act_bias
    );

    let assemblyFormat = [{
        `act_bias` `(` $act_bias `)`
        attr-dict
    }];

    // TODO: E#59642 Add  verifyer type constraint to U8/I8 once IDU ops are ready

}

//
// MPEWeightsBias
//

def VPUIPDPU_MPEWeightsBiasOp :
        VPUIPDPU_Op<"MPEWeightsBias",
            [
                HasParent<"vpux::VPUIPDPU::MPECfgOp">
            ]
        > {
    let summary = "MPE set weights bias for U8 or I8 input op";

    let description = [{
        Used for asymetric integer quantization.
        The bias value (zero point) is treated as U8 if input is U8 or I8 if input is I8.
        Thus the op will restrict the bias value as such.
        For all other data types it is disregarded.
        }];

    let arguments = (ins
        IntAttr:$weights_bias
    );

    let assemblyFormat = [{
        `weights_bias` `(` $weights_bias `)`
        attr-dict
    }];
    // TODO: E#59642 Add  verifyer type constraint to U8/I8 once IDU ops are ready
}

//
// PPEFpBiasAddOp
//

def VPUIPDPU_PPEFpBiasAddOp :
        VPUIPDPU_Op<"PPEFpBiasAdd",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE floating point pipeline bias addition";

    let arguments = (ins
        Optional<AnyMemRef>:$scale_table,
        OptionalAttr<F32Attr>:$bias_static
    );

    let assemblyFormat = [{
        ($scale_table^ `:` type($scale_table))?
        (`bias_static` `(` $bias_static^ `)`)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// PPEFpScalePreluMultOp
//

def VPUIPDPU_PPEFpScalePreluMultOp :
        VPUIPDPU_Op<"PPEFpScalePreluMult",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE floating point pipeline scale multiplication";

    let arguments = (ins
        Optional<AnyMemRef>:$scale_table,
        OptionalAttr<F32Attr>:$scale_static,
        OptionalAttr<F32Attr>:$prelu_alpha
    );

    let assemblyFormat = [{
        ($scale_table^ `:` type($scale_table))?
        (`scale_static` `(` $scale_static^ `)`)?
        (`prelu_alpha` `(` $prelu_alpha^ `)`)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// PPEFpAddMultBypassOp
//

def VPUIPDPU_PPEFpAddMultBypassOp :
        VPUIPDPU_Op<"PPEFpAddMultBypass",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE floating point pipeline bypass add and mult steps";

    let arguments = (ins
        VPUIPDPU_PPEBypassMode:$bypass_mode
    );

    let assemblyFormat = [{
        `bypass_mode` `(` $bypass_mode `)`
        attr-dict
    }];
}

//
// PPEFpConvertOp
//

def VPUIPDPU_PPEFpConvertOp :
        VPUIPDPU_Op<"PPEFpConvert",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE floating point pipeline conversion";

    let description = [{
Conversion modes and attributes <-> reg. fields relation:
| Convert fp32  to | Round mode     | Reg. for round | Clamp mode                  | Ftz mode                              | Reg. for clamp/ftz            |
|:-----------------|:---------------|:---------------|:----------------------------|:--------------------------------------|:------------------------------|
| none             | N/A            | N/A            | N/A                         | N/A                                   | N/A                           |
| fp16             | rne (implicit) | N/A            | no_clamp / clamp_to_min_max | support_denoms / force_denoms_to_zero | ppe_fp16_clamp / ppe_fp16_ftz |
| bf16             | trunc / rne    | ppe_bf16_round | N/A                         | N/A                                   | N/A                           |
| fp8              | rne (implicit) | N/A            | no_clamp / clamp_to_min_max | support_denoms / force_denoms_to_zero | ppe_fp16_clamp / ppe_fp16_ftz |
| i32              | rne (implicit) | N/A            | N/A                         | N/A                                   | N/A                           |
        }];

    let arguments = (ins
        VPUIPDPU_PPEFpConvertMode:$convert_mode,
        OptionalAttr<VPUIPDPU_PPEFpConvClampMode>:$clamp_mode,
        OptionalAttr<VPUIPDPU_PPEFpConvFTZMode>:$ftz_mode,
        OptionalAttr<VPUIPDPU_PPEFpConvBf16RoundMode>:$bf16_round_mode
    );

    let assemblyFormat = [{
        `convert_mode` `(` $convert_mode `)`
        (`clamp_mode` `(` $clamp_mode^ `)`)?
        (`ftz_mode` `(` $ftz_mode^ `)`)?
        (`bf16_round_mode` `(` $bf16_round_mode^ `)`)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// PPEIntBiasAddOp
//

def VPUIPDPU_PPEIntBiasAddOp :
        VPUIPDPU_Op<"PPEIntBiasAdd",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline bias addition";

    let arguments = (ins
        Optional<AnyMemRef>:$scale_table,
        OptionalAttr<IntAttr>:$bias_static
    );

    let assemblyFormat = [{
        ($scale_table^ `:` type($scale_table))?
        (`bias_static` `(` $bias_static^ `)`)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// PPEIntScaleMultOp
//

def VPUIPDPU_PPEIntScaleMultOp :
        VPUIPDPU_Op<"PPEIntScaleMult",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline scale multiplication";

    let arguments = (ins
        Optional<AnyMemRef>:$scale_table,
        OptionalAttr<IntAttr>:$scale_static
    );

    let assemblyFormat = [{
        ($scale_table^ `:` type($scale_table))?
        (`scale_static` `(` $scale_static^ `)`)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// PPEIntPreluMultOp
//

def VPUIPDPU_PPEIntPreluMultOp :
        VPUIPDPU_Op<"PPEIntPreluMult",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline prelu multiplicator";

    let arguments = (ins
        IntAttr:$prelu_mult_static
    );

    let assemblyFormat = [{
        `prelu_mult_static` `(` $prelu_mult_static `)`
        attr-dict
    }];
}

//
// PPEIntScaleShiftOp
//

def VPUIPDPU_PPEIntScaleShiftOp :
        VPUIPDPU_Op<"PPEIntScaleShift",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline scale shift";

    let arguments = (ins
        Optional<AnyMemRef>:$scale_table,
        OptionalAttr<IntAttr>:$shift_static
    );

    let assemblyFormat = [{
        ($scale_table^ `:` type($scale_table))?
        (`shift_static` `(` $shift_static^ `)`)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// PPEIntPreluShiftOp
//

def VPUIPDPU_PPEIntPreluShiftOp :
        VPUIPDPU_Op<"PPEIntPreluShift",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline prelu shift";

    let arguments = (ins
        IntAttr:$prelu_shift_static
    );

    let assemblyFormat = [{
        `prelu_shift_static` `(` $prelu_shift_static `)`
        attr-dict
    }];
}

//
// PPEIntRoundOp
//

def VPUIPDPU_PPEIntRoundOp :
        VPUIPDPU_Op<"PPEIntRound",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline rounding mode";

    let arguments = (ins
        VPUIPDPU_PPEIntRoundMode:$round_mode
    );

    let assemblyFormat = [{
        `round_mode` `(` $round_mode `)`
        attr-dict
    }];
}

//
// PPEIntZeroPointOffsetOp
//

def VPUIPDPU_PPEIntZeroPointOffsetOp :
        VPUIPDPU_Op<"PPEIntZeroPointOffset",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline zero point";

    let arguments = (ins
        IntAttr:$zero_point_static
    );

    let assemblyFormat = [{
        `zero_point_static` `(` $zero_point_static `)`
        attr-dict
    }];
}

//
// PPEIntClampOp
//

def VPUIPDPU_PPEIntClampOp :
        VPUIPDPU_Op<"PPEIntClamp",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline clamping";

    let description = [{
        In case PPE output is FP, only clamp_high is used by HW.
        Therefore the clamp_low param is optional, to be used only on the integer output data path.
    }];

    let arguments = (ins
        OptionalAttr<IntAttr>:$clamp_low,
        IntAttr:$clamp_high
    );

    let assemblyFormat = [{
        (`clamp_low` `(` $clamp_low^ `)`)?
        `clamp_high` `(` $clamp_high `)`
        attr-dict
    }];
}

//
// PPEIntConvertOp
//

def VPUIPDPU_PPEIntConvertOp :
        VPUIPDPU_Op<"PPEIntConvert",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline conversion";

    let arguments = (ins
        VPUIPDPU_PPEIntConvertMode:$convert_mode
    );

    let assemblyFormat = [{
        `convert_mode` `(` $convert_mode `)`
        attr-dict
    }];
}

//
// ODUCastOp
//

def VPUIPDPU_ODUCastOp :
        VPUIPDPU_Op<"ODUCast",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">,
                LimitedToArch<["vpux::VPU::ArchKind::NPU37XX"]>
            ]
        > {
    let summary = "ODU cast config.";

    let arguments = (ins
        AnyMemRef:$cast_output
    );

    let assemblyFormat = [{
        `cast_output` `(` $cast_output `:` type($cast_output) `)`
        attr-dict
    }];
}

//
// ODUOutTensorSizeOp
//

def VPUIPDPU_ODUOutTensorSizeOp :
        VPUIPDPU_Op<"ODUOutTensorSize",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU output tensor dimensions";

    let arguments = (ins
        IntAttr:$dim_x,
        IntAttr:$dim_y,
        IntAttr:$dim_z
    );

    let assemblyFormat = [{
        `dim_x` `(` $dim_x `)`
        `dim_y` `(` $dim_y `)`
        `dim_z` `(` $dim_z `)`
        attr-dict
    }];
}

//
// ODUDataReuseOp
//

def VPUIPDPU_ODUDataReuseOp :
        VPUIPDPU_Op<"ODUDataReuse",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU data type";

    let arguments = (ins

        VPUIPDPU_ODUActivationReuseMode:$activation_reuse
    );

    let assemblyFormat = [{
        `activation_reuse` `(` $activation_reuse `)`
        attr-dict
    }];
}

//
// ODUPermuteDataOp
//

def VPUIPDPU_ODUPermuteDataOp :
        VPUIPDPU_Op<"ODUPermuteData",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU data permutation mode";

    let arguments = (ins
        VPUIPDPU_ODUPermuteDataMode:$permute_mode
    );

    let assemblyFormat = [{
        `permute_mode` `(` $permute_mode `)`
        attr-dict
    }];
}

//
// ODUSparsityOp
//

def VPUIPDPU_ODUSparsityOp :
        VPUIPDPU_Op<"ODUSparsity",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU data sparsity config.";

    let description = [{
The default behaviour regarding sparsity (in case this op is not specified in the IR):
write_sp = 0, sp_out_en = 0 → dense activation written to ac_base; no sparsity output

Valid combinations of params:
| sparsity_map      | compression_enabled | sparse_value      | Behaviour                                                                                                    |
|:------------------|:--------------------|:------------------|:-------------------------------------------------------------------------------------------------------------|
| param present     | param not present   | param optional    | write_sp = 1, sp_out_en = 1 → sparse activation written to ac_base; sparsity map written to sp_base          |
| param present     | eq FALSE            | param not present | write_sp = 1, sp_out_en = 0 → dense activation written to ac_base; "dummy" sparsity map written to sp_base |
| param present     | eq TRUE             | param optional    | write_sp = 1, sp_out_en = 1 → sparse activation written to ac_base; sparsity map written to sp_base          |
| param not present | eq TRUE             | param optional    | write_sp = 0, sp_out_en = 1 → sparse activation written to ac_base; no sparsity output                       |

where sparse_value is used by the ODU to determine sparsity.

The table above could be transformed into the following argument verification rules where at least
one of the statements has to be satisfied:
* compression is enabled (eq TRUE)
* sparsity map is specified (param present) and compression is not set (param not present)
* sparsity map is specified (param present) and sparse value is not set (param not present)
        }];

    let arguments = (ins
        Optional<AnyMemRef>:$sparsity_map,
        OptionalAttr<BoolAttr>:$compression_enabled,
        OptionalAttr<IntAttr>:$sparse_value
    );

    let assemblyFormat = [{
        ($sparsity_map^ `:` type($sparsity_map))?
        (`compression_enabled` `(` $compression_enabled^ `)`)?
        (`sparse_value` `(` $sparse_value^ `)`)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// ODUSwizzleDataOp
//

def VPUIPDPU_ODUSwizzleDataOp :
        VPUIPDPU_Op<"ODUSwizzleData",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU swizzle write out data config.";

    let arguments = (ins
        VPUIPDPU_DPUSwizzleKey:$swizzle_key
    );

    let assemblyFormat = [{
        `swizzle_key` `(` $swizzle_key `)`
        attr-dict
    }];
}

//
// ODUOutActivationsOp
//

def VPUIPDPU_ODUOutActivationsOp :
        VPUIPDPU_Op<"ODUOutActivations",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU activations output config.";

    let description = [{
Data type specification:
| attr name  | value range                                | vpu arch |
|:-----------|:-------------------------------------------|:---------|
| data_type  | fp16, u8f, g8, i8, i32, i4, i2, log, bin   | NPU37XX  |
| data_width | 1-bit, 2-bit, 4-bit, 8-bit, 16-bit, 32-bit | NPU40XX  |
If data_type/data_width is not specified then ODU output data type should be inferred from %out_activations.
        }];

    let arguments = (ins
        AnyMemRef:$out_activations,
        OptionalAttr<VPUIPDPU_ODUDataTypes>:$data_type,
        OptionalAttr<VPUIPDPU_ODUDataBitWidth>:$data_width
    );

    let assemblyFormat = [{
        `out_activations` `(` $out_activations `:` type($out_activations) `)`
        (`data_type` `(` $data_type^ `)`)?
        (`data_width` `(` $data_width^ `)`)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// ODUMemoryModeOp
//

def VPUIPDPU_ODUMemoryModeOp :
        VPUIPDPU_Op<"ODUMemoryMode",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU memory mode config.";

    let arguments = (ins
        VPUIPDPU_ODUMemoryMode:$mem_mode
    );

    let assemblyFormat = [{
        `mem_mode` `(` $mem_mode `)`
        attr-dict
    }];
}

//
// ODUOutSubtensorOp
//

def VPUIPDPU_ODUOutSubtensorOp :
        VPUIPDPU_Op<"ODUOutSubtensor",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">
            ]
        > {
    let summary = "ODU output subtensor descriptor - begin and end coordinates";

    let arguments = (ins
        IntAttr:$begin_coord_x,
        IntAttr:$begin_coord_y,
        IntAttr:$begin_coord_z,
        IntAttr:$end_coord_x,
        IntAttr:$end_coord_y,
        IntAttr:$end_coord_z
    );

    let assemblyFormat = [{
        `begin_coord_x` `(` $begin_coord_x `)`
        `begin_coord_y` `(` $begin_coord_y `)`
        `begin_coord_z` `(` $begin_coord_z `)`
        `end_coord_x` `(` $end_coord_x `)`
        `end_coord_y` `(` $end_coord_y `)`
        `end_coord_z` `(` $end_coord_z `)`
        attr-dict
    }];
}

#endif

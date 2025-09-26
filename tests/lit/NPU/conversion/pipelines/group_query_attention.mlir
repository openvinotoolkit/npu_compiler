//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --import-IE ./group_query_attention.xml | FileCheck %s

// CHECK-LABEL: module @gqaTest
// CHECK: net.NetworkInfo entryPoint : @main inputsInfo : {
// CHECK:   DataInfo "Q" : tensor<1x32x12x96xf32>
// CHECK:   DataInfo "K" : tensor<1x32x12x96xf32>
// CHECK:   DataInfo "V" : tensor<1x32x12x96xf32>
// CHECK:   DataInfo "past_key_values.0.key" : tensor<1x32x1024x96xf32>
// CHECK:   DataInfo "past_key_values.0.value" : tensor<1x32x1024x96xf32>
// CHECK:   DataInfo "sequence" : tensor<1x1xsi32>
// CHECK:   DataInfo "cos_cache" : tensor<4096x48xf32>
// CHECK:   DataInfo "sin_cache" : tensor<4096x48xf32>
// CHECK: } outputsInfo : {
// CHECK:   DataInfo "/model/layers.0/attn/GroupQueryAttention.0" friendlyName = "gqa_result" tensorNames = ["/model/layers.0/attn/GroupQueryAttention/output_0"] : tensor<1x12x3072xf32>
// CHECK:   DataInfo "/model/layers.0/attn/GroupQueryAttention.1" friendlyName = "present.0.key" tensorNames = ["/model/layers.0/attn/GroupQueryAttention_cast_to_present.0.key"] : tensor<1x32x1024x96xf32>
// CHECK:   DataInfo "/model/layers.0/attn/GroupQueryAttention.2" friendlyName = "present.0.value" tensorNames = ["/model/layers.0/attn/GroupQueryAttention_cast_to_present.0.value"] : tensor<1x32x1024x96xf32>
// CHECK: }

// CHECK: func.func @main(
// CHECK-SAME: [[QIN:%[^:]+]]: tensor<1x32x12x96xf32>,
// CHECK-SAME: [[KIN:%[^:]+]]: tensor<1x32x12x96xf32>,
// CHECK-SAME: [[VIN:%[^:]+]]: tensor<1x32x12x96xf32>,
// CHECK-SAME: [[PK:%[^:]+]]: tensor<1x32x1024x96xf32>,
// CHECK-SAME: [[PV:%[^:]+]]: tensor<1x32x1024x96xf32>,
// CHECK-SAME: [[SEQ:%[^:]+]]: tensor<1x1xsi32>,
// CHECK-SAME: [[COS:%[^:]+]]: tensor<4096x48xf32>,
// CHECK-SAME: [[SIN:%[^:]+]]: tensor<4096x48xf32>
// CHECK-SAME: ) -> (tensor<1x12x3072xf32>, tensor<1x32x1024x96xf32>, tensor<1x32x1024x96xf32>)

// CHECK: [[CST:%.*]] = const.Declare tensor<3xsi64> = dense<[0, 0, 12]> : tensor<3xsi64>
// CHECK: [[CST0:%.*]] = const.Declare tensor<3xsi64> = dense<[0, 0, 1024]> : tensor<3xsi64>
// CHECK: [[CST1:%.*]] = const.Declare tensor<3xsi64> = dense<1> : tensor<3xsi64>
// CHECK: [[SLICE_V:%.*]] = IE.StridedSlice([[PV]], [[CST]], [[CST0]], [[CST1]])
// CHECK: [[PRESENT_VALUE:%.*]] = IE.Concat([[SLICE_V]], [[VIN]])
// CHECK: [[CST2:%.+]] = const.Declare tensor<3xsi64> = dense<[0, 0, 12]> : tensor<3xsi64>
// CHECK: [[CST3:%.+]] = const.Declare tensor<3xsi64> = dense<[0, 0, 1024]> : tensor<3xsi64>
// CHECK: [[CST4:%.+]] = const.Declare tensor<3xsi64> = dense<1> : tensor<3xsi64>
// CHECK: [[SLICE_K:%.+]] = IE.StridedSlice([[PK]], [[CST2]], [[CST3]], [[CST4]])
// CHECK: [[CST5:%.+]] = const.Declare tensor<si64> = dense<-1> : tensor<si64>
// CHECK: [[SPLITK0:%.+]]:2 = IE.Split([[KIN]], [[CST5]])
// CHECK: [[CST6:%.+]] = const.Declare tensor<12xsi64> = dense<{{\[}}0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11{{\]}}> : tensor<12xsi64>
// CHECK: [[SEQ_I64:%.+]] = IE.Convert([[SEQ]]) {dstElemType = si64}
// CHECK: [[CST7:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
// CHECK: [[SEQ_REAL:%.+]] = IE.Add([[SEQ_I64]], [[CST7]])
// CHECK: [[SEQ_RESHAPE:%.+]] = IE.Reshape([[SEQ_REAL]], [[CST7]])
// CHECK: [[CST8:%.+]] = const.Declare tensor<1xsi64> = dense<12> : tensor<1xsi64>
// CHECK: [[SEQSUB:%.+]] = IE.Subtract([[SEQ_RESHAPE]], [[CST8]])
// CHECK: [[IDX:%.+]] = IE.Add([[CST6]], [[SEQSUB]])
// CHECK: [[CST9:%.+]] = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
// CHECK: [[GATHER_COS:%.+]] = IE.Gather([[COS]], [[IDX]], [[CST9]])
// CHECK: [[MULK0:%.+]] = IE.Multiply([[SPLITK0]]#0, [[GATHER_COS]])
// CHECK: [[GATHER_SIN:%.+]] = IE.Gather([[SIN]], [[IDX]], [[CST9]])
// CHECK: [[MULK1:%.+]] = IE.Multiply([[SPLITK0]]#1, [[GATHER_SIN]])
// CHECK: [[SUBK:%.+]] = IE.Subtract([[MULK0]], [[MULK1]])
// CHECK: [[MULK0S:%.+]] = IE.Multiply([[SPLITK0]]#0, [[GATHER_SIN]])
// CHECK: [[MULK1C:%.+]] = IE.Multiply([[SPLITK0]]#1, [[GATHER_COS]])
// CHECK: [[ADDK:%.+]] = IE.Add([[MULK0S]], [[MULK1C]])
// CHECK: [[CONCATK:%.+]] = IE.Concat([[SUBK]], [[ADDK]])
// CHECK: [[PRESENT_KEY:%.+]] = IE.Concat([[SLICE_K]], [[CONCATK]])
// CHECK: [[SPLITQ:%.+]]:2 = IE.Split([[QIN]], {{%.+}})
// CHECK: [[MULQ0:%.+]] = IE.Multiply([[SPLITQ]]#0, [[GATHER_COS]])
// CHECK: [[MULQ1:%.+]] = IE.Multiply([[SPLITQ]]#1, [[GATHER_SIN]])
// CHECK: [[SUBQ:%.+]] = IE.Subtract([[MULQ0]], [[MULQ1]])
// CHECK: [[MULQ0S:%.+]] = IE.Multiply([[SPLITQ]]#0, [[GATHER_SIN]])
// CHECK: [[MULQ1C:%.+]] = IE.Multiply([[SPLITQ]]#1, [[GATHER_COS]])
// CHECK: [[ADDQ:%.+]] = IE.Add([[MULQ0S]], [[MULQ1C]])
// CHECK: [[CONCATQ:%.+]] = IE.Concat([[SUBQ]], [[ADDQ]])
// CHECK: [[CST11:%.+]] = const.Declare tensor<f32> = dense<0.102062076> : tensor<f32>
// CHECK: [[MULQ:%.+]] = IE.Multiply([[CONCATQ]], [[CST11]])
// CHECK: [[CST12:%.+]] = const.Declare tensor<4xsi32> = dense<{{\[}}0, 1, 3, 2{{\]}}> : tensor<4xsi32>
// CHECK: [[TRANSPOSEK:%.+]] = IE.Transpose([[PRESENT_KEY]], [[CST12]])
// CHECK: [[MATMUL:%.+]] = IE.MatMul([[MULQ]], [[TRANSPOSEK]])
// CHECK: [[CST13:%.+]] = const.Declare tensor<1x1024xsi64>
// CHECK: [[CST14:%.+]] = const.Declare tensor<1xsi64> = dense<1024> : tensor<1xsi64>
// CHECK: [[SUBSEQ:%.+]] = IE.Subtract([[CST14]], [[SEQ_RESHAPE]])
// CHECK: [[CST15:%.+]] = const.Declare tensor<2xsi64> = dense<{{\[}}12, 1{{\]}}> : tensor<2xsi64>
// CHECK: [[BROADCAST:%.+]] = IE.Broadcast([[SUBSEQ]], [[CST15]])
// CHECK: [[GE:%.+]] = IE.GreaterEqual([[CST13]], [[BROADCAST]])
// CHECK: [[CST16:%.+]] = const.Declare tensor<12x1024xf32>
// CHECK: [[CST17:%.+]] = const.Declare tensor<f32> = dense<0xFF800000> : tensor<f32>
// CHECK: [[SELECT:%.+]] = IE.Select([[GE]], [[CST16]], [[CST17]])
// CHECK: [[ADDATTN:%.+]] = IE.Add([[MATMUL]], [[SELECT]])
// CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[ADDATTN]])
// CHECK: [[MATMUL2:%.+]] = IE.MatMul([[SOFTMAX]], [[PRESENT_VALUE]])
// CHECK: [[CST18:%.+]] = const.Declare tensor<4xsi64> = dense<{{\[}}0, 2, 1, 3{{\]}}> : tensor<4xsi64>
// CHECK: [[TRANSPOSEO:%.+]] = IE.Transpose([[MATMUL2]], [[CST18]])
// CHECK: [[CST19:%.+]] = const.Declare tensor<3xsi32> = dense<{{\[}}0, 0, -1{{\]}}> : tensor<3xsi32>
// CHECK: [[RESULT:%.+]] = IE.Reshape([[TRANSPOSEO]], [[CST19]]) {special_zero}
// CHECK: return [[RESULT]], [[PRESENT_KEY]], [[PRESENT_VALUE]]

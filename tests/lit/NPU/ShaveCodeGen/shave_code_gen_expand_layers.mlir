//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
// RUN:  vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --expand-layers %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// IE.Sin

module @SingleSinF16Layer {
  module @VPU.SW {
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    func.func @generated_0(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 1000 {
              %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
              %1 = arith.extf %0 : f16 to f32
              %2 = math.sin %1 : f32
              %3 = arith.truncf %2 : f32 to f16
              affine.store %3, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
            }
          }
        }
      }
      return %arg1 : memref<1x1x1x1000xf16>
    }
  }
}

// CHECK-LABEL: SingleSinF16Layer 
// CHECK: func.func @generated_0([[ARG0:%.+]]: memref<1x1x1x1000xf16>, [[ARG1:%.+]]: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
// CHECK-NEXT:  [[C1:%.+]] = arith.constant 1 : i32
// CHECK-NEXT:  [[C3:%.+]] = arith.constant 3 : i32
// CHECK-NEXT:  [[CST:%.+]] = arith.constant -2.59630184E-7 : f32
// CHECK-NEXT:  [[CST0:%.+]] = arith.constant 2.47562348E-5 : f32
// CHECK-NEXT:  [[CST1:%.+]] = arith.constant -0.00138883304 : f32
// CHECK-NEXT:  [[CST2:%.+]] = arith.constant 0.0416666418 : f32
// CHECK-NEXT:  [[CST3:%.+]] = arith.constant -5.000000e-01 : f32
// CHECK-NEXT:  [[CST4:%.+]] = arith.constant -2.50293279E-8 : f32
// CHECK-NEXT:  [[CST5:%.+]] = arith.constant 2.76001265E-6 : f32
// CHECK-NEXT:  [[CST6:%.+]] = arith.constant -1.98426045E-4 : f32
// CHECK-NEXT:  [[CST7:%.+]] = arith.constant 0.00833334774 : f32
// CHECK-NEXT:  [[CST8:%.+]] = arith.constant -0.166666672 : f32
// CHECK-NEXT:  [[CST9:%.+]] = arith.constant -1.000000e+00 : f32
// CHECK-NEXT:  [[CST10:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:  [[CST11:%.+]] = arith.constant 1.57079637 : f32
// CHECK-NEXT:  [[CST12:%.+]] = arith.constant 0.636619746 : f32
// CHECK-NEXT:  affine.for [[D0:%.+]] = 0 to 1 {
// CHECK-NEXT:    affine.for [[D1:%.+]] = 0 to 1 {
// CHECK-NEXT:      affine.for [[D2:%.+]] = 0 to 1 {
// CHECK-NEXT:        affine.for [[D3:%.+]] = 0 to 1000 {
// CHECK-NEXT:          [[IN:%.+]] = affine.load [[ARG0]][[[D0]], [[D1]], [[D2]], [[D3]]] : memref<1x1x1x1000xf16>
// CHECK-NEXT:          [[EX1:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:          [[M1:%.+]] = arith.mulf [[EX1]], [[CST12]] : f32
// CHECK-NEXT:          [[F1:%.+]] = math.floor [[M1]] : f32
// CHECK-NEXT:          [[M2:%.+]] = arith.mulf [[F1]], [[CST11]] : f32
// CHECK-NEXT:          [[S1:%.+]] = arith.subf [[EX1]], [[M2]] : f32
// CHECK-NEXT:          [[FPT1:%.+]] = arith.fptosi [[F1]] : f32 to i32
// CHECK-NEXT:          [[AND1:%.+]] = arith.andi [[FPT1]], [[C3]] : i32
// CHECK-NEXT:          [[CMP1:%.+]] = arith.cmpi eq, [[AND1]], [[C1]] : i32
// CHECK-NEXT:          [[CMP2:%.+]] = arith.cmpi eq, [[AND1]], [[C3]] : i32
// CHECK-NEXT:          [[O1:%.+]] = arith.ori [[CMP1]], [[CMP2]] : i1
// CHECK-NEXT:          [[CMP3:%.+]] = arith.cmpi sgt, [[AND1]], [[C1]] : i32
// CHECK-NEXT:          [[M3:%.+]] = arith.mulf [[S1]], [[S1]] : f32
// CHECK-NEXT:          [[S1:%.+]] = arith.select [[O1]], [[CST10]], %5 : f32
// CHECK-NEXT:          [[S2:%.+]] = arith.select [[O1]], [[CST3]], [[CST8]] : f32
// CHECK-NEXT:          [[S3:%.+]] = arith.select [[O1]], [[CST2]], [[CST7]] : f32
// CHECK-NEXT:          [[S4:%.+]] = arith.select [[O1]], [[CST1]], [[CST6]] : f32
// CHECK-NEXT:          [[S5:%.+]] = arith.select [[O1]], [[CST0]], [[CST5]] : f32
// CHECK-NEXT:          [[S6:%.+]] = arith.select [[O1]], [[CST]], [[CST4]] : f32
// CHECK-NEXT:          [[M4:%.+]] = arith.mulf [[M3]], [[S6]] : f32
// CHECK-NEXT:          [[A1:%.+]] = arith.addf [[M4]], [[S5]] : f32
// CHECK-NEXT:          [[M5:%.+]] = arith.mulf [[M3]], [[A1]] : f32
// CHECK-NEXT:          [[A2:%.+]] = arith.addf [[M5]], [[S4]] : f32
// CHECK-NEXT:          [[M6:%.+]] = arith.mulf [[M3]], [[A2]] : f32
// CHECK-NEXT:          [[A3:%.+]] = arith.addf [[M6]], [[S3]] : f32
// CHECK-NEXT:          [[M7:%.+]] = arith.mulf [[M3]], [[A3]] : f32
// CHECK-NEXT:          [[A4:%.+]] = arith.addf [[M7]], [[S2]] : f32
// CHECK-NEXT:          [[M8:%.+]] = arith.mulf [[M3]], [[A4]] : f32
// CHECK-NEXT:          [[A5:%.+]] = arith.addf [[M8]], [[CST10]] : f32
// CHECK-NEXT:          [[M9:%.+]] = arith.mulf [[S1]], [[A5]] : f32
// CHECK-NEXT:          [[M10:%.+]] = arith.mulf [[M9]], [[CST9]] : f32
// CHECK-NEXT:          [[S6:%.+]] = arith.select [[CMP3]], [[M10]], [[M9]] : f32
// CHECK-NEXT:          [[FINAL:%.+]] = arith.truncf [[S6]] : f32 to f16
// CHECK-NEXT:          affine.store [[FINAL]], [[ARG1]][[[D0]], [[D1]], [[D2]], [[D3]]] : memref<1x1x1x1000xf16>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return [[ARG1]] : memref<1x1x1x1000xf16>
// CHECK-NEXT:}

// -----
// IE.Cos

module @SingleCosF16Layer {
  module @VPU.SW {
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    func.func @generated_0(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 1000 {
              %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
              %1 = arith.extf %0 : f16 to f32
              %2 = math.cos %1 : f32
              %3 = arith.truncf %2 : f32 to f16
              affine.store %3, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
            }
          }
        }
      }
      return %arg1 : memref<1x1x1x1000xf16>
    }
  }
}

// CHECK-LABEL: SingleCosF16Layer 
// CHECK: func.func @generated_0([[ARG0:%.+]]: memref<1x1x1x1000xf16>, [[ARG1:%.+]]: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
// CHECK-NEXT:  [[C2:%.+]] = arith.constant 2 : i32
// CHECK-NEXT:  [[C1:%.+]] = arith.constant 1 : i32
// CHECK-NEXT:  [[C0:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:  [[C3:%.+]] = arith.constant 3 : i32
// CHECK-NEXT:  [[CST:%.+]] = arith.constant -2.59630184E-7 : f32
// CHECK-NEXT:  [[CST0:%.+]] = arith.constant 2.47562348E-5 : f32
// CHECK-NEXT:  [[CST1:%.+]] = arith.constant -0.00138883304 : f32
// CHECK-NEXT:  [[CST2:%.+]] = arith.constant 0.0416666418 : f32
// CHECK-NEXT:  [[CST3:%.+]] = arith.constant -5.000000e-01 : f32
// CHECK-NEXT:  [[CST4:%.+]] = arith.constant -2.50293279E-8 : f32
// CHECK-NEXT:  [[CST5:%.+]] = arith.constant 2.76001265E-6 : f32
// CHECK-NEXT:  [[CST6:%.+]] = arith.constant -1.98426045E-4 : f32
// CHECK-NEXT:  [[CST7:%.+]] = arith.constant 0.00833334774 : f32
// CHECK-NEXT:  [[CST8:%.+]] = arith.constant -0.166666672 : f32
// CHECK-NEXT:  [[CST9:%.+]] = arith.constant -1.000000e+00 : f32
// CHECK-NEXT:  [[CST10:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:  [[CST11:%.+]] = arith.constant 1.57079637 : f32
// CHECK-NEXT:  [[CST12:%.+]] = arith.constant 0.636619746 : f32
// CHECK-NEXT:  affine.for [[D0:%.+]] = 0 to 1 {
// CHECK-NEXT:    affine.for [[D1:%.+]] = 0 to 1 {
// CHECK-NEXT:      affine.for [[D2:%.+]] = 0 to 1 {
// CHECK-NEXT:        affine.for [[D3:%.+]] = 0 to 1000 {
// CHECK-NEXT:          [[IN:%.+]] = affine.load [[ARG0]][[[D0]], [[D1]], [[D2]], [[D3]]] : memref<1x1x1x1000xf16>
// CHECK-NEXT:          [[EX1:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:          [[M1:%.+]] = arith.mulf [[EX1]], [[CST12]] : f32
// CHECK-NEXT:          [[F1:%.+]] = math.floor [[M1]] : f32
// CHECK-NEXT:          [[M2:%.+]] = arith.mulf [[F1]], [[CST11]] : f32
// CHECK-NEXT:          [[S1:%.+]] = arith.subf [[EX1]], [[M2]] : f32
// CHECK-NEXT:          [[FPT1:%.+]] = arith.fptosi [[F1]] : f32 to i32
// CHECK-NEXT:          [[AND1:%.+]] = arith.andi [[FPT1]], [[C3]] : i32
// CHECK-NEXT:          [[CMP1:%.+]] = arith.cmpi eq, [[AND1]], [[C0]] : i32
// CHECK-NEXT:          [[CMP2:%.+]] = arith.cmpi eq, [[AND1]], [[C1]] : i32
// CHECK-NEXT:          [[CMP3:%.+]] = arith.cmpi eq, [[AND1]], [[C2]] : i32
// CHECK-NEXT:          [[O1:%.+]] = arith.ori [[CMP1]], [[CMP3]] : i1
// CHECK-NEXT:          [[O2:%.+]] = arith.ori [[CMP2]], [[CMP3]] : i1
// CHECK-NEXT:          [[M3:%.+]] = arith.mulf [[S1]], [[S1]] : f32
// CHECK-NEXT:          [[S1:%.+]] = arith.select [[O1]], [[CST10]], %5 : f32
// CHECK-NEXT:          [[S2:%.+]] = arith.select [[O1]], [[CST3]], [[CST8]] : f32
// CHECK-NEXT:          [[S3:%.+]] = arith.select [[O1]], [[CST2]], [[CST7]] : f32
// CHECK-NEXT:          [[S4:%.+]] = arith.select [[O1]], [[CST1]], [[CST6]] : f32
// CHECK-NEXT:          [[S5:%.+]] = arith.select [[O1]], [[CST0]], [[CST5]] : f32
// CHECK-NEXT:          [[S6:%.+]] = arith.select [[O1]], [[CST]], [[CST4]] : f32
// CHECK-NEXT:          [[M4:%.+]] = arith.mulf [[M3]], [[S6]] : f32
// CHECK-NEXT:          [[A1:%.+]] = arith.addf [[M4]], [[S5]] : f32
// CHECK-NEXT:          [[M5:%.+]] = arith.mulf [[M3]], [[A1]] : f32
// CHECK-NEXT:          [[A2:%.+]] = arith.addf [[M5]], [[S4]] : f32
// CHECK-NEXT:          [[M6:%.+]] = arith.mulf [[M3]], [[A2]] : f32
// CHECK-NEXT:          [[A3:%.+]] = arith.addf [[M6]], [[S3]] : f32
// CHECK-NEXT:          [[M7:%.+]] = arith.mulf [[M3]], [[A3]] : f32
// CHECK-NEXT:          [[A4:%.+]] = arith.addf [[M7]], [[S2]] : f32
// CHECK-NEXT:          [[M8:%.+]] = arith.mulf [[M3]], [[A4]] : f32
// CHECK-NEXT:          [[A5:%.+]] = arith.addf [[M8]], [[CST10]] : f32
// CHECK-NEXT:          [[M9:%.+]] = arith.mulf [[S1]], [[A5]] : f32
// CHECK-NEXT:          [[M10:%.+]] = arith.mulf [[M9]], [[CST9]] : f32
// CHECK-NEXT:          [[S6:%.+]] = arith.select [[O2]], [[M10]], [[M9]] : f32
// CHECK-NEXT:          [[FINAL:%.+]] = arith.truncf [[S6]] : f32 to f16
// CHECK-NEXT:          affine.store [[FINAL]], [[ARG1]][[[D0]], [[D1]], [[D2]], [[D3]]] : memref<1x1x1x1000xf16>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return [[ARG1]] : memref<1x1x1x1000xf16>
// CHECK-NEXT:}

// -----
// IE.RoundEven

module @RoundEvenFP16Layer {
  module @VPU.SW {
    func.func @generated_0(%arg0: f16) -> f16 {
      %0 = math.roundeven %arg0 : f16
      return %0 : f16
    }
  }
}

// CHECK-LABEL: RoundEvenFP16Layer 
// CHECK: func.func @generated_0([[ARG0:%.+]]: f16) -> f16
// CHECK-DAG:  [[C327:%.+]] = arith.constant 32767 : i16
// CHECK-DAG:  [[CM1:%.+]] = arith.constant -1 : i16
// CHECK-DAG:  [[C255_I16:%.+]] = arith.constant 25599 : i16
// CHECK-DAG:  [[C15:%.+]] = arith.constant 15 : i16
// CHECK-DAG:  [[CST:%.+]] = arith.constant 1.024000e+03 : f16
// CHECK-DAG:  [[C328:%.+]] = arith.constant -32768 : i16
// CHECK:       [[BCAST:%.+]] = arith.bitcast [[ARG0]] : f16 to i16
// CHECK-NEXT:  [[XABS:%.+]] = arith.andi [[BCAST]], [[C327]] : i16
// CHECK-NEXT:  [[XSIGN:%.+]] = arith.andi [[BCAST]], [[C328]] : i16
// CHECK-NEXT:  [[DIFFG:%.+]] = arith.subi [[C255_I16]], [[XABS]] : i16
// CHECK-NEXT:  [[ISGREAT:%.+]] = arith.shrsi [[DIFFG]], [[C15]] : i16
// CHECK-NEXT:  [[FXABS:%.+]] = arith.bitcast [[XABS]] : i16 to f16
// CHECK-NEXT:  [[SUM:%.+]] = arith.addf [[FXABS]], [[CST]] : f16
// CHECK-NEXT:  [[VROUND:%.+]] = arith.subf [[SUM]], [[CST]] : f16
// CHECK-NEXT:  [[IVROUND:%.+]] = arith.bitcast [[VROUND]] : f16 to i16
// CHECK-NEXT:  [[ISNOTG:%.+]] = arith.xori [[ISGREAT]], [[CM1]] : i16
// CHECK-NEXT:  [[ONE:%.+]] = arith.andi [[ISNOTG]], [[IVROUND]] : i16
// CHECK-NEXT:  [[TWO:%.+]] = arith.andi [[ISGREAT]], [[BCAST]] : i16
// CHECK-NEXT:  [[COMBINED1:%.+]] = arith.ori [[ONE]], [[TWO]] : i16
// CHECK-NEXT:  [[COMBINED:%.+]] = arith.ori [[COMBINED1]], [[XSIGN]] : i16
// CHECK-NEXT:  [[RES:%.+]] = arith.bitcast [[COMBINED]] : i16 to f16


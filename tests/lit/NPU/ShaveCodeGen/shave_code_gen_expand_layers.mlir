//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN:  vpux-opt --split-input-file --init-compiler="platform=%platform%" --expand-layers %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// IE.Sin

module @SingleSinF16Layer {
  module @VPU.SW {
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
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
// CHECK-NEXT:          [[SUBF1:%.+]] = arith.subf [[EX1]], [[M2]] : f32
// CHECK-NEXT:          [[FPT1:%.+]] = arith.fptosi [[F1]] : f32 to i32
// CHECK-NEXT:          [[AND1:%.+]] = arith.andi [[FPT1]], [[C3]] : i32
// CHECK-NEXT:          [[CMP1:%.+]] = arith.cmpi eq, [[AND1]], [[C1]] : i32
// CHECK-NEXT:          [[CMP2:%.+]] = arith.cmpi eq, [[AND1]], [[C3]] : i32
// CHECK-NEXT:          [[O1:%.+]] = arith.ori [[CMP1]], [[CMP2]] : i1
// CHECK-NEXT:          [[CMP3:%.+]] = arith.cmpi sgt, [[AND1]], [[C1]] : i32
// CHECK-NEXT:          [[M3:%.+]] = arith.mulf [[SUBF1]], [[SUBF1]] : f32
// CHECK-NEXT:          [[S1:%.+]] = arith.select [[O1]], [[CST10]], [[SUBF1]] : f32
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
// IE.Asin

module @SingleAsinF16Layer {
  module @VPU.SW {
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    func.func @generated_0(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
      affine.for %d0 = 0 to 1 {
        affine.for %d1 = 0 to 1 {
          affine.for %d2 = 0 to 1 {
            affine.for %d3 = 0 to 1000 {
              %in = affine.load %arg0[%d0, %d1, %d2, %d3] : memref<1x1x1x1000xf16>
              %res = math.asin %in fastmath<afn> : f16
              affine.store %res, %arg1[%d0, %d1, %d2, %d3] : memref<1x1x1x1000xf16>
            }
          }
        }
      }
      return %arg1 : memref<1x1x1x1000xf16>
    }
  }
}
// CHECK-LABEL: SingleAsinF16Layer
// CHECK: func.func @generated_0([[ARG0:%.+]]: memref<1x1x1x1000xf16>, [[ARG1:%.+]]: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
// CHECK-NEXT:  [[CST:%.+]] = arith.constant 1.570310e+00 : f16
// CHECK-NEXT:  [[CST0:%.+]] = arith.constant 1.666260e-01 : f16
// CHECK-NEXT:  [[CST1:%.+]] = arith.constant 7.501220e-02 : f16
// CHECK-NEXT:  [[CST2:%.+]] = arith.constant 4.464720e-02 : f16
// CHECK-NEXT:  [[CST3:%.+]] = arith.constant 3.03802{{[^:]*}} : f16
// CHECK-NEXT:  [[CST4:%.+]] = arith.constant 2.236940e-02 : f16
// CHECK-NEXT:  [[CST5:%.+]] = arith.constant 1.733400e-02 : f16
// CHECK-NEXT:  [[CST6:%.+]] = arith.constant 1.410680e-02 : f16
// CHECK-NEXT:  [[CST7:%.+]] = arith.constant 1.049040e-02 : f16
// CHECK-NEXT:  [[CST8:%.+]] = arith.constant 1.526640e-02 : f16
// CHECK-NEXT:  [[CST9:%.+]] = arith.constant -1.132970e-02 : f16
// CHECK-NEXT:  [[CST10:%.+]] = arith.constant 5.422970e-02 : f16
// CHECK-NEXT:  [[CST11:%.+]] = arith.constant -6.204220e-02 : f16
// CHECK-NEXT:  [[CST12:%.+]] = arith.constant 5.557250e-02 : f16
// CHECK-NEXT:  [[CST13:%.+]] = arith.constant 5.000000e-01 : f16
// CHECK-NEXT:  [[CST14:%.+]] = arith.constant 1.000000e+00 : f16
// CHECK-NEXT:  affine.for [[D0:%.+]] = 0 to 1 {
// CHECK-NEXT:    affine.for [[D1:%.+]] = 0 to 1 {
// CHECK-NEXT:      affine.for [[D2:%.+]] = 0 to 1 {
// CHECK-NEXT:        affine.for [[D3:%.+]] = 0 to 1000 {
// CHECK-NEXT:          [[IN:%.+]] = affine.load [[ARG0]][[[D0]], [[D1]], [[D2]], [[D3]]] : memref<1x1x1x1000xf16>
// CHECK-NEXT:          [[ABS1:%.+]] = math.absf [[IN]] : f16
// CHECK-NEXT:          [[MUL1:%.+]] = arith.mulf [[IN]], [[IN]] : f16
// CHECK-NEXT:          [[SUB1:%.+]] = arith.subf [[CST14]], [[MUL1]] : f16
// CHECK-NEXT:          [[SQR1:%.+]] = math.sqrt [[SUB1]] : f16
// CHECK-NEXT:          [[CMP1:%.+]] = arith.cmpf ogt, [[MUL1]], [[CST13]] : f16
// CHECK-NEXT:          [[SEL1:%.+]] = arith.select [[CMP1]], [[SQR1]], [[ABS1]] : f16
// CHECK-NEXT:          [[MUL2:%.+]] = arith.mulf [[SEL1]], [[SEL1]] : f16
// CHECK-NEXT:          [[MUL3:%.+]] = arith.mulf [[MUL2]], [[MUL2]] : f16
// CHECK-NEXT:          [[MUL4:%.+]] = arith.mulf [[MUL3]], [[CST12]] : f16
// CHECK-NEXT:          [[ADD1:%.+]] = arith.addf [[MUL4]], [[CST10]] : f16
// CHECK-NEXT:          [[MUL5:%.+]] = arith.mulf [[MUL3]], [[CST11]] : f16
// CHECK-NEXT:          [[ADD2:%.+]] = arith.addf [[MUL5]], [[CST9]] : f16
// CHECK-NEXT:          [[MUL6:%.+]] = arith.mulf [[ADD1]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD3:%.+]] = arith.addf [[MUL6]], [[CST8]] : f16
// CHECK-NEXT:          [[MUL7:%.+]] = arith.mulf [[ADD2]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD4:%.+]] = arith.addf [[MUL7]], [[CST7]] : f16
// CHECK-NEXT:          [[MUL8:%.+]] = arith.mulf [[ADD3]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD5:%.+]] = arith.addf [[MUL8]], [[CST6]] : f16
// CHECK-NEXT:          [[MUL9:%.+]] = arith.mulf [[ADD4]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD6:%.+]] = arith.addf [[MUL9]], [[CST5]] : f16
// CHECK-NEXT:          [[MUL10:%.+]] = arith.mulf [[ADD5]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD7:%.+]] = arith.addf [[MUL10]], [[CST4]] : f16
// CHECK-NEXT:          [[MUL11:%.+]] = arith.mulf [[ADD6]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD8:%.+]] = arith.addf [[MUL11]], [[CST3]] : f16
// CHECK-NEXT:          [[MUL12:%.+]] = arith.mulf [[ADD7]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD9:%.+]] = arith.addf [[MUL12]], [[CST2]] : f16
// CHECK-NEXT:          [[MUL13:%.+]] = arith.mulf [[ADD8]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD10:%.+]] = arith.addf [[MUL13]], [[CST1]] : f16
// CHECK-NEXT:          [[MUL14:%.+]] = arith.mulf [[ADD9]], [[MUL2]] : f16
// CHECK-NEXT:          [[ADD11:%.+]] = arith.addf [[MUL14]], [[ADD10]] : f16
// CHECK-NEXT:          [[MUL15:%.+]] = arith.mulf [[ADD11]], [[MUL2]] : f16
// CHECK-NEXT:          [[ADD12:%.+]] = arith.addf [[MUL15]], [[CST0]] : f16
// CHECK-NEXT:          [[MUL16:%.+]] = arith.mulf [[SEL1]], [[MUL2]] : f16
// CHECK-NEXT:          [[MUL17:%.+]] = arith.mulf [[ADD12]], [[MUL16]] : f16
// CHECK-NEXT:          [[ADD13:%.+]] = arith.addf [[MUL17]], [[SEL1]] : f16
// CHECK-NEXT:          [[SUB2:%.+]] = arith.subf [[CST]], [[ADD13]] : f16
// CHECK-NEXT:          [[SEL2:%.+]] = arith.select [[CMP1]], [[SUB2]], [[ADD13]] : f16
// CHECK-NEXT:          [[FINAL:%.+]] = math.copysign [[SEL2]], [[IN]] : f16
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
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
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
// CHECK-NEXT:          [[SUBF1:%.+]] = arith.subf [[EX1]], [[M2]] : f32
// CHECK-NEXT:          [[FPT1:%.+]] = arith.fptosi [[F1]] : f32 to i32
// CHECK-NEXT:          [[AND1:%.+]] = arith.andi [[FPT1]], [[C3]] : i32
// CHECK-NEXT:          [[CMP1:%.+]] = arith.cmpi eq, [[AND1]], [[C0]] : i32
// CHECK-NEXT:          [[CMP2:%.+]] = arith.cmpi eq, [[AND1]], [[C1]] : i32
// CHECK-NEXT:          [[CMP3:%.+]] = arith.cmpi eq, [[AND1]], [[C2]] : i32
// CHECK-NEXT:          [[O1:%.+]] = arith.ori [[CMP1]], [[CMP3]] : i1
// CHECK-NEXT:          [[O2:%.+]] = arith.ori [[CMP2]], [[CMP3]] : i1
// CHECK-NEXT:          [[M3:%.+]] = arith.mulf [[SUBF1]], [[SUBF1]] : f32
// CHECK-NEXT:          [[S1:%.+]] = arith.select [[O1]], [[CST10]], [[SUBF1]] : f32
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
// IE.Acos

module @SingleAcosF16Layer {
  module @VPU.SW {
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    func.func @generated_0(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
      affine.for %d0 = 0 to 1 {
        affine.for %d1 = 0 to 1 {
          affine.for %d2 = 0 to 1 {
            affine.for %d3 = 0 to 1000 {
              %in = affine.load %arg0[%d0, %d1, %d2, %d3] : memref<1x1x1x1000xf16>
              %res = math.acos %in fastmath<afn> : f16
              affine.store %res, %arg1[%d0, %d1, %d2, %d3] : memref<1x1x1x1000xf16>
            }
          }
        }
      }
      return %arg1 : memref<1x1x1x1000xf16>
    }
  }
}
// CHECK-LABEL: SingleAcosF16Layer
// CHECK: func.func @generated_0([[ARG0:%.+]]: memref<1x1x1x1000xf16>, [[ARG1:%.+]]: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
// CHECK-NEXT:  [[CST:%.+]] = arith.constant 3.140630e+00 : f16
// CHECK-NEXT:  [[CST0:%.+]] = arith.constant 2.000000e+00 : f16
// CHECK-NEXT:  [[CST1:%.+]] = arith.constant 1.570310e+00 : f16
// CHECK-NEXT:  [[CST2:%.+]] = arith.constant 1.666260e-01 : f16
// CHECK-NEXT:  [[CST3:%.+]] = arith.constant 7.501220e-02 : f16
// CHECK-NEXT:  [[CST4:%.+]] = arith.constant 4.464720e-02 : f16
// CHECK-NEXT:  [[CST5:%.+]] = arith.constant 3.03802{{[^:]*}} : f16
// CHECK-NEXT:  [[CST6:%.+]] = arith.constant 2.236940e-02 : f16
// CHECK-NEXT:  [[CST7:%.+]] = arith.constant 1.733400e-02 : f16
// CHECK-NEXT:  [[CST8:%.+]] = arith.constant 1.410680e-02 : f16
// CHECK-NEXT:  [[CST9:%.+]] = arith.constant 1.049040e-02 : f16
// CHECK-NEXT:  [[CST10:%.+]] = arith.constant 1.526640e-02 : f16
// CHECK-NEXT:  [[CST11:%.+]] = arith.constant -1.132970e-02 : f16
// CHECK-NEXT:  [[CST12:%.+]] = arith.constant 5.422970e-02 : f16
// CHECK-NEXT:  [[CST13:%.+]] = arith.constant -6.204220e-02 : f16
// CHECK-NEXT:  [[CST14:%.+]] = arith.constant 5.557250e-02 : f16
// CHECK-NEXT:  [[CST15:%.+]] = arith.constant 1.000000e+00 : f16
// CHECK-NEXT:  [[CST16:%.+]] = arith.constant -5.625000e-01 : f16
// CHECK-NEXT:  [[CST17:%.+]] = arith.constant -1.000000e+00 : f16
// CHECK-NEXT:  [[CST18:%.+]] = arith.constant 5.000000e-01 : f16
// CHECK-NEXT:  [[CST19:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:  affine.for [[D0:%.+]] = 0 to 1 {
// CHECK-NEXT:    affine.for [[D1:%.+]] = 0 to 1 {
// CHECK-NEXT:      affine.for [[D2:%.+]] = 0 to 1 {
// CHECK-NEXT:        affine.for [[D3:%.+]] = 0 to 1000 {
// CHECK-NEXT:          [[IN:%.+]] = affine.load [[ARG0]][[[D0]], [[D1]], [[D2]], [[D3]]] : memref<1x1x1x1000xf16>
// CHECK-NEXT:          [[NEG1:%.+]] = arith.subf [[CST19]], [[IN]] : f16
// CHECK-NEXT:          [[CMP1:%.+]] = arith.cmpf ogt, [[IN]], [[CST19]] : f16
// CHECK-NEXT:          [[SEL1:%.+]] = arith.select [[CMP1]], [[NEG1]], [[IN]] : f16
// CHECK-NEXT:          [[CMP2:%.+]] = arith.cmpf ogt, [[SEL1]], [[CST16]] : f16
// CHECK-NEXT:          [[ABS1:%.+]] = math.absf [[SEL1]] : f16
// CHECK-NEXT:          [[MUL1:%.+]] = arith.mulf [[SEL1]], [[SEL1]] : f16
// CHECK-NEXT:          [[SUB1:%.+]] = arith.subf [[CST15]], [[MUL1]] : f16
// CHECK-NEXT:          [[SQR1:%.+]] = math.sqrt [[SUB1]] : f16
// CHECK-NEXT:          [[CMP3:%.+]] = arith.cmpf ogt, [[MUL1]], [[CST18]] : f16
// CHECK-NEXT:          [[SEL2:%.+]] = arith.select [[CMP3]], [[SQR1]], [[ABS1]] : f16
// CHECK-NEXT:          [[MUL2:%.+]] = arith.mulf [[SEL2]], [[SEL2]] : f16
// CHECK-NEXT:          [[MUL3:%.+]] = arith.mulf [[MUL2]], [[MUL2]] : f16
// CHECK-NEXT:          [[MUL4:%.+]] = arith.mulf [[MUL3]], [[CST14]] : f16
// CHECK-NEXT:          [[ADD1:%.+]] = arith.addf [[MUL4]], [[CST12]] : f16
// CHECK-NEXT:          [[MUL5:%.+]] = arith.mulf [[MUL3]], [[CST13]] : f16
// CHECK-NEXT:          [[ADD2:%.+]] = arith.addf [[MUL5]], [[CST11]] : f16
// CHECK-NEXT:          [[MUL6:%.+]] = arith.mulf [[ADD1]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD3:%.+]] = arith.addf [[MUL6]], [[CST10]] : f16
// CHECK-NEXT:          [[MUL7:%.+]] = arith.mulf [[ADD2]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD4:%.+]] = arith.addf [[MUL7]], [[CST9]] : f16
// CHECK-NEXT:          [[MUL8:%.+]] = arith.mulf [[ADD3]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD5:%.+]] = arith.addf [[MUL8]], [[CST8]] : f16
// CHECK-NEXT:          [[MUL9:%.+]] = arith.mulf [[ADD4]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD6:%.+]] = arith.addf [[MUL9]], [[CST7]] : f16
// CHECK-NEXT:          [[MUL10:%.+]] = arith.mulf [[ADD5]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD7:%.+]] = arith.addf [[MUL10]], [[CST6]] : f16
// CHECK-NEXT:          [[MUL11:%.+]] = arith.mulf [[ADD6]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD8:%.+]] = arith.addf [[MUL11]], [[CST5]] : f16
// CHECK-NEXT:          [[MUL12:%.+]] = arith.mulf [[ADD7]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD9:%.+]] = arith.addf [[MUL12]], [[CST4]] : f16
// CHECK-NEXT:          [[MUL13:%.+]] = arith.mulf [[ADD8]], [[MUL3]] : f16
// CHECK-NEXT:          [[ADD10:%.+]] = arith.addf [[MUL13]], [[CST3]] : f16
// CHECK-NEXT:          [[MUL14:%.+]] = arith.mulf [[ADD9]], [[MUL2]] : f16
// CHECK-NEXT:          [[ADD11:%.+]] = arith.addf [[MUL14]], [[ADD10]] : f16
// CHECK-NEXT:          [[MUL15:%.+]] = arith.mulf [[ADD11]], [[MUL2]] : f16
// CHECK-NEXT:          [[ADD12:%.+]] = arith.addf [[MUL15]], [[CST2]] : f16
// CHECK-NEXT:          [[MUL16:%.+]] = arith.mulf [[SEL2]], [[MUL2]] : f16
// CHECK-NEXT:          [[MUL17:%.+]] = arith.mulf [[ADD12]], [[MUL16]] : f16
// CHECK-NEXT:          [[ADD13:%.+]] = arith.addf [[MUL17]], [[SEL2]] : f16
// CHECK-NEXT:          [[SUB2:%.+]] = arith.subf [[CST1]], [[ADD13]] : f16
// CHECK-NEXT:          [[SEL3:%.+]] = arith.select [[CMP3]], [[SUB2]], [[ADD13]] : f16
// CHECK-NEXT:          [[CPY1:%.+]] = math.copysign [[SEL3]], [[SEL1]] : f16
// CHECK-NEXT:          [[ADD14:%.+]] = arith.addf [[CPY1]], [[CST1]] : f16
// CHECK-NEXT:          [[MUL18:%.+]] = arith.mulf [[SEL1]], [[CST18]] : f16
// CHECK-NEXT:          [[ADD15:%.+]] = arith.addf [[MUL18]], [[CST18]] : f16
// CHECK-NEXT:          [[SQR2:%.+]] = math.sqrt [[ADD15]] : f16
// CHECK-NEXT:          [[ABS2:%.+]] = math.absf [[SQR2]] : f16
// CHECK-NEXT:          [[MUL19:%.+]] = arith.mulf [[SQR2]], [[SQR2]] : f16
// CHECK-NEXT:          [[SUB3:%.+]] = arith.subf [[CST15]], [[MUL19]] : f16
// CHECK-NEXT:          [[SQR3:%.+]] = math.sqrt [[SUB3]] : f16
// CHECK-NEXT:          [[CMP4:%.+]] = arith.cmpf ogt, [[MUL19]], [[CST18]] : f16
// CHECK-NEXT:          [[SEL4:%.+]] = arith.select [[CMP4]], [[SQR3]], [[ABS2]] : f16
// CHECK-NEXT:          [[MUL20:%.+]] = arith.mulf [[SEL4]], [[SEL4]] : f16
// CHECK-NEXT:          [[MUL21:%.+]] = arith.mulf [[MUL20]], [[MUL20]] : f16
// CHECK-NEXT:          [[MUL22:%.+]] = arith.mulf [[MUL21]], [[CST14]] : f16
// CHECK-NEXT:          [[ADD16:%.+]] = arith.addf [[MUL22]], [[CST12]] : f16
// CHECK-NEXT:          [[MUL23:%.+]] = arith.mulf [[MUL21]], [[CST13]] : f16
// CHECK-NEXT:          [[ADD17:%.+]] = arith.addf [[MUL23]], [[CST11]] : f16
// CHECK-NEXT:          [[MUL24:%.+]] = arith.mulf [[ADD16]], [[MUL21]] : f16
// CHECK-NEXT:          [[ADD18:%.+]] = arith.addf [[MUL24]], [[CST10]] : f16
// CHECK-NEXT:          [[MUL25:%.+]] = arith.mulf [[ADD17]], [[MUL21]] : f16
// CHECK-NEXT:          [[ADD19:%.+]] = arith.addf [[MUL25]], [[CST9]] : f16
// CHECK-NEXT:          [[MUL26:%.+]] = arith.mulf [[ADD18]], [[MUL21]] : f16
// CHECK-NEXT:          [[ADD20:%.+]] = arith.addf [[MUL26]], [[CST8]] : f16
// CHECK-NEXT:          [[MUL27:%.+]] = arith.mulf [[ADD19]], [[MUL21]] : f16
// CHECK-NEXT:          [[ADD21:%.+]] = arith.addf [[MUL27]], [[CST7]] : f16
// CHECK-NEXT:          [[MUL28:%.+]] = arith.mulf [[ADD20]], [[MUL21]] : f16
// CHECK-NEXT:          [[ADD22:%.+]] = arith.addf [[MUL28]], [[CST6]] : f16
// CHECK-NEXT:          [[MUL29:%.+]] = arith.mulf [[ADD21]], [[MUL21]] : f16
// CHECK-NEXT:          [[ADD23:%.+]] = arith.addf [[MUL29]], [[CST5]] : f16
// CHECK-NEXT:          [[MUL30:%.+]] = arith.mulf [[ADD22]], [[MUL21]] : f16
// CHECK-NEXT:          [[ADD24:%.+]] = arith.addf [[MUL30]], [[CST4]] : f16
// CHECK-NEXT:          [[MUL31:%.+]] = arith.mulf [[ADD23]], [[MUL21]] : f16
// CHECK-NEXT:          [[ADD25:%.+]] = arith.addf [[MUL31]], [[CST3]] : f16
// CHECK-NEXT:          [[MUL32:%.+]] = arith.mulf [[ADD24]], [[MUL20]] : f16
// CHECK-NEXT:          [[ADD26:%.+]] = arith.addf [[MUL32]], [[ADD25]] : f16
// CHECK-NEXT:          [[MUL33:%.+]] = arith.mulf [[ADD26]], [[MUL20]] : f16
// CHECK-NEXT:          [[ADD27:%.+]] = arith.addf [[MUL33]], [[CST2]] : f16
// CHECK-NEXT:          [[MUL34:%.+]] = arith.mulf [[SEL4]], [[MUL20]] : f16
// CHECK-NEXT:          [[MUL35:%.+]] = arith.mulf [[ADD27]], [[MUL34]] : f16
// CHECK-NEXT:          [[ADD28:%.+]] = arith.addf [[MUL35]], [[SEL4]] : f16
// CHECK-NEXT:          [[SUB4:%.+]] = arith.subf [[CST1]], [[ADD28]] : f16
// CHECK-NEXT:          [[SEL5:%.+]] = arith.select [[CMP4]], [[SUB4]], [[ADD28]] : f16
// CHECK-NEXT:          [[CPY2:%.+]] = math.copysign [[SEL5]], [[SQR2]] : f16
// CHECK-NEXT:          [[MUL36:%.+]] = arith.mulf [[CPY2]], [[CST0]] : f16
// CHECK-NEXT:          [[SEL6:%.+]] = arith.select [[CMP2]], [[ADD14]], [[MUL36]] : f16
// CHECK-NEXT:          [[CMP5:%.+]] = arith.cmpf oge, [[IN]], [[CST17]] : f16
// CHECK-NEXT:          [[CMP6:%.+]] = arith.cmpf olt, [[IN]], [[CST19]] : f16
// CHECK-NEXT:          [[AND1:%.+]] = arith.andi [[CMP5]], [[CMP6]] : i1
// CHECK-NEXT:          [[NEG2:%.+]] = arith.subf [[CST19]], [[SEL6]] : f16
// CHECK-NEXT:          [[ADD29:%.+]] = arith.addf [[NEG2]], [[CST]] : f16
// CHECK-NEXT:          [[FINAL:%.+]] = arith.select [[AND1]], [[ADD29]], [[SEL6]] : f16
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

// -----
// arith.NegFOp

module @NegFFP16Layer {
  module @VPU.SW {
    func.func @generated_0(%arg0: f16) -> f16 {
      %0 = arith.negf %arg0 : f16
      return %0 : f16
    }
  }
}

// CHECK-LABEL: NegFFP16Layer
// CHECK: func.func @generated_0([[ARG0:%.+]]: f16) -> f16 {
// CHECK-NEXT:  [[CST:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:  [[RES:%.+]] = arith.subf [[CST]], [[ARG0]] : f16
// CHECK-NEXT:  return [[RES]] : f16
// CHECK-NEXT:}

// -----
// math.Acosh

module @AcoshF16Layer {
  module @VPU.SW {
    func.func @generated_0(%arg0: f16) -> f16 {
      %0 = math.acosh %arg0 : f16
      return %0 : f16
    }
  }
}

    // CHECK-NOT:     math.acosh
    // CHECK-LABEL: AcoshF16Layer
    // CHECK: func.func @generated_0([[ARG0:%.+]]: f16) -> f16 {
    // CHECK-NEXT:      [[BIG:%.+]] = arith.constant 2.557500e+02 : f16
    // CHECK-NEXT:      [[LN2:%.+]] = arith.constant 6.933590e-01 : f16
    // CHECK-NEXT:      [[ONE:%.+]] = arith.constant 1.000000e+00 : f16
    // CHECK-NEXT:      [[MONE:%.+]] = arith.constant -1.000000e+00 : f16
    // CHECK-NEXT:      [[NAN:%.+]] = arith.constant 0x7E00 : f16
    // CHECK-NEXT:      [[C12:%.+]] = arith.constant 1.200200e+00 : f16
    // CHECK-NEXT:      [[CP1:%.+]] = arith.constant -8.331300e-02 : f16
    // CHECK-NEXT:      [[CP2:%.+]] = arith.constant 1.875310e-02 : f16
    // CHECK-NEXT:      [[CP3:%.+]] = arith.constant -5.580900e-03 : f16
    // CHECK-NEXT:      [[TWO:%.+]] = arith.constant 2.000000e+00 : f16

    // CHECK-NEXT:      [[SMALL:%.+]] = arith.cmpf olt, [[ARG0]], [[ONE]] : f16
    // CHECK-NEXT:      [[XMIN1:%.+]] = arith.addf [[ARG0]], [[MONE]] : f16
    // CHECK-NEXT:      [[P0:%.+]] = arith.mulf [[XMIN1]], [[CP3]] : f16
    // CHECK-NEXT:      [[P1:%.+]] = arith.addf [[P0]], [[CP2]] : f16
    // CHECK-NEXT:      [[P2:%.+]] = arith.mulf [[P1]], [[XMIN1]] : f16
    // CHECK-NEXT:      [[P3:%.+]] = arith.addf [[P2]], [[CP1]] : f16
    // CHECK-NEXT:      [[P4:%.+]] = arith.mulf  [[P3]], [[XMIN1]] : f16
    // CHECK-NEXT:      [[PU:%.+]] = arith.addf [[P4]], [[ONE]] : f16
    // CHECK-NEXT:      [[V2U:%.+]] = arith.mulf [[XMIN1]], [[TWO]] : f16
    // CHECK-NEXT:      [[S2U:%.+]] = math.sqrt [[V2U]] fastmath<afn> : f16
    // CHECK-NEXT:      [[VLT2:%.+]] = arith.mulf [[S2U]], [[PU]] : f16
    // CHECK-NEXT:      [[ISLT2:%.+]] = arith.cmpf olt, [[ARG0]], [[C12]] : f16
    // CHECK-NEXT:      [[LOGF:%.+]] = math.log [[ARG0]] fastmath<afn> : f16
    // CHECK-NEXT:      [[BIGV:%.+]] = arith.addf [[LOGF]], [[LN2]] : f16
    // CHECK-NEXT:      [[ISOVER:%.+]] = arith.cmpf ogt, [[ARG0]], [[BIG]] : f16
    // CHECK-NEXT:      [[XPLUS1:%.+]] = arith.addf [[ARG0]], [[ONE]] : f16
    // CHECK-NEXT:      [[MSQR:%.+]] = arith.mulf [[XMIN1]], [[XPLUS1]] : f16
    // CHECK-NEXT:      [[SQRV:%.+]] = math.sqrt [[MSQR]] fastmath<afn> : f16
    // CHECK-NEXT:      [[ADDS:%.+]] = arith.addf [[ARG0]], [[SQRV]] : f16
    // CHECK-NEXT:      [[REST:%.+]] = math.log [[ADDS]] : f16
    // CHECK-NEXT:      [[VSMALL:%.+]] = arith.select [[ISLT2]], [[VLT2]], [[REST]] : f16
    // CHECK-NEXT:      [[OVER:%.+]] = arith.select [[ISOVER]], [[BIGV]], [[VSMALL]] : f16
    // CHECK-NEXT:      [[RES:%.+]] = arith.select [[SMALL]], [[NAN]], [[OVER]] : f16
    // CHECK-NEXT:  return [[RES]] : f16
    // CHECK-NEXT:}


// -----
// arith.acosh
module @AcoshF32Layer {
  module @VPU.SW {
    func.func @generated_0(%arg0: f32) -> f32 {
      %0 = math.acosh %arg0 : f32
      return %0 : f32
    }
  }
}

    // CHECK-NOT:     math.acosh
    // CHECK-LABEL: AcoshF32Layer
    // CHECK: func.func @generated_0([[ARG0:%.+]]: f32) -> f32 {
    // CHECK-NEXT:  [[RES:%.+]] = math.acosh [[ARG0]] : f32
    // CHECK-NEXT:  return [[RES]] : f32
    // CHECK-NEXT:}


// -----
// math.asinh

module @AsinhF16Layer {
  module @VPU.SW {
    func.func @generated_0(%arg0: f16) -> f16 {
      %0 = math.asinh %arg0 : f16
      return %0 : f16
    }
  }
}

    // CHECK-NOT:     math.acosh
    // CHECK-LABEL: AsinhF16Layer
    // CHECK: func.func @generated_0([[ARG0:%.+]]: f16) -> f16 {

    // CHECK-NEXT:      [[HALFONE:%.+]] = arith.constant 5.000000e-01 : f32
    // CHECK-NEXT:      [[ONE:%.+]] = arith.constant 1.000000e+00 : f32
    // CHECK-NEXT:      [[VLN:%.+]] = arith.constant 0.693147182 : f32
    // CHECK-NEXT:      [[BIG:%.+]] = arith.constant 1.84467441E+19 : f32
    // CHECK-NEXT:      [[CQ6:%.+]] = arith.constant 0.481211841 : f32
    // CHECK-NEXT:      [[CQ5:%.+]] = arith.constant 0.894425511 : f32
    // CHECK-NEXT:      [[CQ4:%.+]] = arith.constant -0.178837836 : f32
    // CHECK-NEXT:      [[CQ3:%.+]] = arith.constant -0.0482780561 : f32
    // CHECK-NEXT:      [[CQ2:%.+]] = arith.constant 0.0751035884 : f32
    // CHECK-NEXT:      [[CQ1:%.+]] = arith.constant -0.0349466801 : f32
    // CHECK-NEXT:      [[CQ0:%.+]] = arith.constant 0.0058438424 : f32
    // CHECK-NEXT:      [[CP3:%.+]] = arith.constant -0.166662768 : f32
    // CHECK-NEXT:      [[CP2:%.+]] = arith.constant 0.0748451725 : f32
    // CHECK-NEXT:      [[CP1:%.+]] = arith.constant -0.0426840074 : f32
    // CHECK-NEXT:      [[CP0:%.+]] = arith.constant 0.0200918131 : f32

    // CHECK-NEXT:      [[X32:%.+]] = arith.extf [[ARG0]] : f16 to f32
    // CHECK-NEXT:      [[ABS:%.+]] = math.absf [[X32]] : f32
    // CHECK-NEXT:      [[X2:%.+]] = arith.mulf [[X32]], [[X32]] : f32
    // CHECK-NEXT:      [[P0:%.+]] = arith.mulf [[X2]], [[CP0]] : f32
    // CHECK-NEXT:      [[P1:%.+]] = arith.addf [[P0]], [[CP1]] : f32
    // CHECK-NEXT:      [[P2:%.+]] = arith.mulf [[X2]], [[P1]] : f32
    // CHECK-NEXT:      [[P3:%.+]] = arith.addf [[P2]], [[CP2]] : f32
    // CHECK-NEXT:      [[P4:%.+]] = arith.mulf [[X2]], [[P3]] : f32
    // CHECK-NEXT:      [[P5:%.+]] = arith.addf [[P4]], [[CP3]] : f32
    // CHECK-NEXT:      [[P6:%.+]] = arith.mulf [[X2]], [[P5]] : f32
    // CHECK-NEXT:      [[P7:%.+]] = arith.addf [[P6]], [[ONE]] : f32
    // CHECK-NEXT:      [[PU:%.+]] = arith.mulf [[X32]], [[P7]] : f32
    // CHECK-NEXT:      [[IS_SMALL:%.+]] = arith.cmpf olt, [[ABS]], [[HALFONE]] : f32

    // CHECK-NEXT:      [[HALF:%.+]] = arith.subf [[ABS]], [[HALFONE]] : f32
    // CHECK-NEXT:      [[Q0:%.+]] = arith.mulf [[HALF]], [[CQ0]] : f32
    // CHECK-NEXT:      [[Q1:%.+]] = arith.addf [[Q0]], [[CQ1]] : f32
    // CHECK-NEXT:      [[Q2:%.+]] = arith.mulf [[HALF]], [[Q1]] : f32
    // CHECK-NEXT:      [[Q3:%.+]] = arith.addf [[Q2]], [[CQ2]] : f32
    // CHECK-NEXT:      [[Q4:%.+]] = arith.mulf [[HALF]], [[Q3]] : f32
    // CHECK-NEXT:      [[Q5:%.+]] = arith.addf [[Q4]], [[CQ3]] : f32
    // CHECK-NEXT:      [[Q6:%.+]] = arith.mulf [[HALF]], [[Q5]] : f32
    // CHECK-NEXT:      [[Q7:%.+]] = arith.addf [[Q6]], [[CQ4]] : f32
    // CHECK-NEXT:      [[Q8:%.+]] = arith.mulf [[HALF]], [[Q7]] : f32
    // CHECK-NEXT:      [[Q9:%.+]] = arith.addf [[Q8]], [[CQ5]] : f32
    // CHECK-NEXT:      [[Q10:%.+]] = arith.mulf [[HALF]], [[Q9]] : f32
    // CHECK-NEXT:      [[Q11:%.+]] = arith.addf [[Q10]], [[CQ6]] : f32
    // CHECK-NEXT:      [[QU:%.+]] = math.copysign [[Q11]], [[X32]] : f32
    // CHECK-NEXT:      [[IS_ONE:%.+]] = arith.cmpf olt, [[ABS]], [[ONE]] : f32

    // CHECK-NEXT:      [[LOGF:%.+]] = math.log [[ABS]] fastmath<afn> : f32
    // CHECK-NEXT:      [[TVAL:%.+]] = arith.addf [[LOGF]], [[VLN]] : f32
    // CHECK-NEXT:      [[BVAL:%.+]] = math.copysign [[TVAL]], [[X32]] : f32
    // CHECK-NEXT:      [[IS_BIG:%.+]] = arith.cmpf ogt, [[ABS]], [[BIG]] : f32

    // CHECK-NEXT:      [[XX:%.+]] = arith.mulf [[ABS]], [[ABS]] : f32
    // CHECK-NEXT:      [[FMA:%.+]] = arith.addf [[XX]], [[ONE]] : f32
    // CHECK-NEXT:      [[SQRV:%.+]] = math.sqrt [[FMA]] fastmath<afn> : f32
    // CHECK-NEXT:      [[TSUM:%.+]] = arith.addf [[SQRV]], [[ABS]] : f32
    // CHECK-NEXT:      [[TMED:%.+]] = math.log [[TSUM]] fastmath<afn> : f32
    // CHECK-NEXT:      [[MED:%.+]] = math.copysign [[TMED]], [[X32]] : f32
    // CHECK-NEXT:      [[VT1:%.+]] = arith.select [[IS_SMALL]], [[PU]], [[QU]] : f32
    // CHECK-NEXT:      [[VT2:%.+]] = arith.select [[IS_ONE]], [[VT1]], [[MED]] : f32
    // CHECK-NEXT:      [[R32:%.+]] = arith.select [[IS_BIG]], [[BVAL]], [[VT2]] : f32
    // CHECK-NEXT:      [[RES:%.+]] = arith.truncf [[R32]] : f32 to f16

    // CHECK-NEXT:  return [[RES]] : f16
    // CHECK-NEXT:}

// -----
// math.asinh

module @AsinhF32Layer {
  module @VPU.SW {
    func.func @generated_0(%arg0: f32) -> f32 {
      %0 = math.asinh %arg0 : f32
      return %0 : f32
    }
  }
}

    // CHECK-NOT:     math.asinh
    // CHECK-LABEL: AsinhF32Layer
    // CHECK: func.func @generated_0([[ARG0:%.+]]: f32) -> f32 {
    // CHECK-NEXT:  [[RES:%.+]] = math.asinh [[ARG0]] : f32
    // CHECK-NEXT:  return [[RES]] : f32
    // CHECK-NEXT:}

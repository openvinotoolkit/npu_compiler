//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-precision-across-function-calls %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<u8:f16, 0.1>

// CHECK-LABEL: @QuantizationPairInMain
module @QuantizationPairInMain {
    func.func private @function(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %quant = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add = IE.Add(%quant, %quant) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %dequant = IE.Dequantize(%add) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        return %dequant : tensor<1x48x60x60xf16>
    }
    func.func @main(%arg0: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
        %dequant = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        %call = call @function(%dequant) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        %quant = IE.Quantize(%call) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        return %quant : tensor<1x48x60x60x!qElemType>
    }

    // CHECK:  func.func private @function([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[CALL:%.+]] = call @function([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:      return [[CALL]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @ConversionPairInMain
module @ConversionPairInMain {
    func.func private @function(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %convert_f16 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf16>
        %add = IE.Add(%convert_f16, %convert_f16) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        %convert_f32 = IE.Convert(%add) {dstElemType = f32} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf32>
        return %convert_f32 : tensor<1x48x60x60xf32>
    }
    func.func @main(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %convert_f32 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf32>
        %call = call @function(%convert_f32) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        %convert_f16 = IE.Convert(%call) {dstElemType = f16} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf16>
        return %convert_f16 : tensor<1x48x60x60xf16>
    }

    // CHECK:  func.func private @function([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[CALL:%.+]] = call @function([[ARG0]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    // CHECK:      return [[CALL]] : tensor<1x48x60x60xf16>
    // CHECK:  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.1>

// CHECK-LABEL: @MultipleIdenticalQuantizeUsers
module @MultipleIdenticalQuantizeUsers {
    func.func private @function(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %quant1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %quant2 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add = IE.Add(%quant1, %quant2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %dequant = IE.Dequantize(%add) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        return %dequant : tensor<1x48x60x60xf16>
    }
    func.func @main(%arg0: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
        %dequant = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        %call = call @function(%dequant) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        %quant = IE.Quantize(%call) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        return %quant : tensor<1x48x60x60x!qElemType>
    }

    // CHECK:  func.func private @function([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[CALL:%.+]] = call @function([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:      return [[CALL]]
    // CHECK:  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.1>
!qElemType1 = !quant.uniform<u8:f16, 0.2>

// CHECK-LABEL: @DoNotOptimizeMultipleDifferentQuantizeUsers
module @DoNotOptimizeMultipleDifferentQuantizeUsers {
    func.func private @function(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60x!qElemType> {
        %quant1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %quant2 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType1>
        %add = IE.Add(%quant1, %quant2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType1> -> tensor<1x48x60x60x!qElemType>
        return %add : tensor<1x48x60x60x!qElemType>
    }
    func.func @main(%arg0: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
        %dequant = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        %call = call @function(%dequant) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60x!qElemType>
        return %call : tensor<1x48x60x60x!qElemType>
    }

    // CHECK:  func.func private @function([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[QUANT1:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType}
    // CHECK:      [[QUANT2:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1}
    // CHECK:      [[ADD:%.+]] = IE.Add([[QUANT1]], [[QUANT2]])
    // CHECK:      return [[ADD]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[DEQUANT:%.+]] = IE.Dequantize([[ARG0]])
    // CHECK:      [[CALL:%.+]] = call @function([[DEQUANT]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:      return [[CALL]]
    // CHECK:  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.1>

// CHECK-LABEL: @MultipleFunctionsSameProducer
module @MultipleFunctionsSameProducer {
    func.func private @function1(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %quant = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add = IE.Add(%quant, %quant) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %dequant = IE.Dequantize(%add) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        return %dequant : tensor<1x48x60x60xf16>
    }
    func.func private @function2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %quant = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add = IE.Add(%quant, %quant) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %dequant = IE.Dequantize(%add) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        return %dequant : tensor<1x48x60x60xf16>
    }
    func.func @main(%arg0: tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>) {
        %dequant = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        %call1 = call @function1(%dequant) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        %call2 = call @function2(%dequant) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %call1, %call2 : tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>
    }

    // CHECK:  func.func private @function1([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      [[DEQUANT1:%.+]] = IE.Dequantize([[ADD1]])
    // CHECK:      return [[DEQUANT1]]
    // CHECK:  }
    // CHECK:  func.func private @function2([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      [[DEQUANT2:%.+]] = IE.Dequantize([[ADD2]])
    // CHECK:      return [[DEQUANT2]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>) {
    // CHECK:      [[CALL1:%.+]] = call @function1([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16>
    // CHECK:      [[CALL2:%.+]] = call @function2([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16>
    // CHECK:      return [[CALL1]], [[CALL2]]
    // CHECK:  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.1>

// CHECK-LABEL: @QuantizationPairInSeparateFunctions
module @QuantizationPairInSeparateFunctions {
    func.func private @function1(%arg0: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16> {
        %add = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %dequant = IE.Dequantize(%add) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        return %dequant : tensor<1x48x60x60xf16>
    }
    func.func private @function2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60x!qElemType> {
        %quant = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add = IE.Add(%quant, %quant) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        return %add : tensor<1x48x60x60x!qElemType>
    }
    func.func @main(%arg0: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
        %call1 = call @function1(%arg0) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16>
        %call2 = call @function2(%call1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60x!qElemType>
        return %call2 : tensor<1x48x60x60x!qElemType>
    }

    // CHECK:  func.func private @function1([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD1]]
    // CHECK:  }
    // CHECK:  func.func private @function2([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD2]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[CALL1:%.+]] = call @function1([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:      [[CALL2:%.+]] = call @function2([[CALL1]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:      return [[CALL2]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @ConversionPairInSeparateFunctions
module @ConversionPairInSeparateFunctions {
    func.func private @function1(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf32> {
        %add = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        %convert = IE.Convert(%add) {dstElemType = f32} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf32>
        return %convert : tensor<1x48x60x60xf32>
    }
    func.func private @function2(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16> {
        %convert = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf16>
        %add = IE.Add(%convert, %convert) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        return %add : tensor<1x48x60x60xf16>
    }
    func.func @main(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %call1 = call @function1(%arg0) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf32>
        %call2 = call @function2(%call1) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16>
        return %call2 : tensor<1x48x60x60xf16>
    }

    // CHECK:  func.func private @function1([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD1]]
    // CHECK:  }
    // CHECK:  func.func private @function2([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD2]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[CALL1:%.+]] = call @function1([[ARG0]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    // CHECK:      [[CALL2:%.+]] = call @function2([[CALL1]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    // CHECK:      return [[CALL2]]
    // CHECK:  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.1>

// CHECK-LABEL: @QuantizationPairMiddleArgPosition
module @QuantizationPairMiddleArgPosition {
    func.func private @function1(%arg0: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16> {
        %add = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %dequant = IE.Dequantize(%add) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        return %dequant : tensor<1x48x60x60xf16>
    }
    func.func private @function2(%arg0: tensor<1x48x60x60xf16>, %arg1: tensor<1x48x60x60xf16>, %arg2: tensor<1x48x60x60xf16>)
            -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60xf16>) {
        %softmax1 = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        %quant = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add = IE.Add(%quant, %quant) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %softmax2 = IE.SoftMax(%arg2) {axisInd = 1 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        return %softmax1, %add, %softmax2 : tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60xf16>
    }
    func.func @main(%arg0: tensor<1x48x60x60x!qElemType>, %arg1: tensor<1x48x60x60xf16>, %arg2: tensor<1x48x60x60xf16>)
            -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60xf16>) {
        %call1 = call @function1(%arg0) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16>
        %call2:3 = call @function2(%arg1, %call1, %arg2) : (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60xf16>)
        return %call2#0, %call2#1, %call2#2 : tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60xf16>
    }

    // CHECK:       func.func private @function1([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:           [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:           return [[ADD1]]
    // CHECK:       }
    // CHECK:       func.func private @function2([[ARG0:%.+]]: tensor<1x48x60x60xf16>, [[ARG1:%.+]]: tensor<1x48x60x60x!qElemType>, [[ARG2:%.+]]: tensor<1x48x60x60xf16>)
    // CHECK-SAME:        -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60xf16>) {
    // CHECK:           [[SOFTMAX1:%.+]] = IE.SoftMax([[ARG0]])
    // CHECK:           [[ADD2:%.+]] = IE.Add([[ARG1]], [[ARG1]])
    // CHECK:           [[SOFTMAX2:%.+]] = IE.SoftMax([[ARG2]])
    // CHECK:           return [[SOFTMAX1]], [[ADD2]], [[SOFTMAX2]]
    // CHECK:       }
    // CHECK:       func.func @main([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>, [[ARG1:%.+]]: tensor<1x48x60x60xf16>, [[ARG2:%.+]]: tensor<1x48x60x60xf16>)
    // CHECK-SAME:        -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60xf16>) {
    // CHECK:           [[CALL1:%.+]] = call @function1([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:           [[CALL2:%.+]]:3 = call @function2([[ARG1]], [[CALL1]], [[ARG2]]) : (tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60xf16>)
    // CHECK-SAME:        -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60xf16>)
    // CHECK:           return [[CALL2]]#0, [[CALL2]]#1, [[CALL2]]#2
    // CHECK:       }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.1>

// CHECK-LABEL: @MultipleQuantizationPairs
module @MultipleQuantizationPairs {
    func.func private @function1(%arg0: tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>) {
        %add1 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %dequant1 = IE.Dequantize(%add1) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        %add2 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %dequant2 = IE.Dequantize(%add2) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        return %dequant1, %dequant2 : tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>
    }
    func.func private @function2(%arg0: tensor<1x48x60x60xf16>, %arg1: tensor<1x48x60x60xf16>) -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>) {
        %quant1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add1 = IE.Add(%quant1, %quant1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %quant2 = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add2 = IE.Add(%quant2, %quant2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        return %add1, %add2 : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>
    }
    func.func @main(%arg0: tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>) {
        %call_part1:2 = call @function1(%arg0) : (tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>)
        %call_part2:2 = call @function2(%call_part1#0, %call_part1#1) : (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>) -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>)
        return %call_part2#0, %call_part2#1 : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>
    }

    // CHECK:       func.func private @function1([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>) {
    // CHECK:           [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:           [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:           return [[ADD1]], [[ADD2]]
    // CHECK:       }
    // CHECK:       func.func private @function2([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>, [[ARG1:%.+]]: tensor<1x48x60x60x!qElemType>)
    // CHECK-SAME:        -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>) {
    // CHECK:           [[ADD3:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:           [[ADD4:%.+]] = IE.Add([[ARG1]], [[ARG1]])
    // CHECK:           return [[ADD3]], [[ADD4]]
    // CHECK:       }
    // CHECK:       func.func @main([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>) {
    // CHECK:           [[CALL1:%.+]]:2 = call @function1([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>)
    // CHECK:           [[CALL2:%.+]]:2 = call @function2([[CALL1]]#0, [[CALL1]]#1) : (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>)
    // CHECK-SAME:        -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>)
    // CHECK:           return [[CALL2]]#0, [[CALL2]]#1
    // CHECK:       }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.1>

// CHECK-LABEL: @MultipleQuantUsers
module @MultipleQuantUsers {
    func.func private @function1(%arg0: tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>) {
        %add1 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %add2 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %dequant = IE.Dequantize(%add1) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        return %dequant, %add2 : tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>
    }
    func.func private @function2(%arg0: tensor<1x48x60x60xf16>, %arg1: tensor<1x48x60x60x!qElemType>)
            -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>) {
        %quant1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add1 = IE.Add(%quant1, %quant1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %quant2 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add2 = IE.Add(%quant2, %quant2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %add3 = IE.Add(%arg1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        return %add1, %add2, %add3 : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>
    }
    func.func @main(%arg0: tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>) {
        %call1:2 = call @function1(%arg0) : (tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>)
        %call2:3 = call @function2(%call1#0, %call1#1) : (tensor<1x48x60x60xf16>, tensor<1x48x60x60x!qElemType>)
            -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>)
        return %call2#0, %call2#1, %call2#2 : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>
    }

    // CHECK:       func.func private @function1([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>) {
    // CHECK:           [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:           [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:           return [[ADD1]], [[ADD2]]
    // CHECK:       }
    // CHECK:       func.func private @function2([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>, [[ARG1:%.+]]: tensor<1x48x60x60x!qElemType>)
    // CHECK-SAME:        -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>) {
    // CHECK:           [[ADD3:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:           [[ADD4:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:           [[ADD5:%.+]] = IE.Add([[ARG1]], [[ARG1]])
    // CHECK:           return [[ADD3]], [[ADD4]], [[ADD5]]
    // CHECK:       }
    // CHECK:       func.func @main([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>) {
    // CHECK:           [[CALL1:%.+]]:2 = call @function1([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>)
    // CHECK:           [[CALL2:%.+]]:3 = call @function2([[CALL1]]#0, [[CALL1]]#1) : (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>)
    // CHECK-SAME:        -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>)
    // CHECK:           return [[CALL2]]#0, [[CALL2]]#1, [[CALL2]]#2
    // CHECK:       }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.1>

// CHECK-LABEL: @DoNotOptimizeMissingQuantizationPair
module @DoNotOptimizeMissingQuantizationPair {
    func.func private @function1(%arg0: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16> {
        %add = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        // Dequantize has no Quantize pair in the second function, so it does not get optimized
        %dequant = IE.Dequantize(%add) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        return %dequant : tensor<1x48x60x60xf16>
    }
    func.func private @function2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %add = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        return %add : tensor<1x48x60x60xf16>
    }
    func.func @main(%arg0: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16> {
        %call1 = call @function1(%arg0) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16>
        %call2 = call @function2(%call1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %call2 : tensor<1x48x60x60xf16>
    }

    // CHECK:  func.func private @function1([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      [[DEQUANT:%.+]] = IE.Dequantize([[ADD1]])
    // CHECK:      return [[DEQUANT]]
    // CHECK:  }
    // CHECK:  func.func private @function2([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD2]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[CALL1:%.+]] = call @function1([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16>
    // CHECK:      [[CALL2:%.+]] = call @function2([[CALL1]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    // CHECK:      return [[CALL2]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @DoNotOptimizeMissingConversionPair
module @DoNotOptimizeMissingConversionPair {
    func.func private @function1(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf32> {
        %add = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        // Convert has no pair in the second function, so it does not get optimized
        %dequant = IE.Convert(%add) {dstElemType = f32} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf32>
        return %dequant : tensor<1x48x60x60xf32>
    }
    func.func private @function2(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %add = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %add : tensor<1x48x60x60xf32>
    }
    func.func @main(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf32> {
        %call1 = call @function1(%arg0) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf32>
        %call2 = call @function2(%call1) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        return %call2 : tensor<1x48x60x60xf32>
    }

    // CHECK:  func.func private @function1([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf32> {
    // CHECK:      [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      [[CONVERT:%.+]] = IE.Convert([[ADD1]])
    // CHECK:      return [[CONVERT]]
    // CHECK:  }
    // CHECK:  func.func private @function2([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:      [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD2]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf32> {
    // CHECK:      [[CALL1:%.+]] = call @function1([[ARG0]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf32>
    // CHECK:      [[CALL2:%.+]] = call @function2([[CALL1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:      return [[CALL2]]
    // CHECK:  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.1>

// CHECK-LABEL: @RepeatedCalls
module @RepeatedCalls {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    func.func private @function(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %quant = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add = IE.Add(%quant, %quant) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %dequant = IE.Dequantize(%add) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        return %dequant : tensor<1x48x60x60xf16>
    }

    func.func @main(%arg0: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
        %dequant = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        %call1 = call @function(%dequant) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        %call2 = call @function(%call1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        %quant = IE.Quantize(%call2) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        return %quant : tensor<1x48x60x60x!qElemType>
    }

    // CHECK:  func.func private @function([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[CALL1:%.+]] = call @function([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:      [[CALL2:%.+]] = call @function([[CALL1]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:      return [[CALL2]]
    // CHECK:  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.1>

// CHECK-LABEL: @RepeatedCallsIncompletePairs
module @RepeatedCallsIncompletePairs {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    func.func private @function(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %quant = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add = IE.Add(%quant, %quant) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %dequant = IE.Dequantize(%add) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        return %dequant : tensor<1x48x60x60xf16>
    }

    func.func @main(%arg0: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16> {
        %dequant = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        %call1 = call @function(%dequant) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        %call2 = call @function(%call1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // %call2 is not used by a (compatible) Quantize op
        return %call2 : tensor<1x48x60x60xf16>
    }

    // CHECK:  func.func private @function([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[CALL1:%.+]] = call @function([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:      [[CALL2:%.+]] = call @function([[CALL1]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:      [[DEQUANT:%.+]] = IE.Dequantize([[CALL2]])
    // CHECK:      return [[DEQUANT]]
    // CHECK:  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.1>

// CHECK-LABEL: @RepeatedCallsPairsInsideAndOusideCalls
module @RepeatedCallsPairsInsideAndOusideCalls {
    func.func private @function1(%arg0: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16> {
        %add = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        %dequant = IE.Dequantize(%add) {dstElemType = f16} : tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60xf16>
        return %dequant : tensor<1x48x60x60xf16>
    }
    func.func private @function2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60x!qElemType> {
        %quant = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>
        %add = IE.Add(%quant, %quant) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType> -> tensor<1x48x60x60x!qElemType>
        return %add : tensor<1x48x60x60x!qElemType>
    }
    func.func @main(%arg0: tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>) {
        %call1_fn1 = call @function1(%arg0) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16>
        %call2_fn1 = call @function1(%arg0) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60xf16>

        %call_fn2 = call @function2(%call1_fn1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60x!qElemType>
        %quant = IE.Quantize(%call2_fn1) {dstElemType = !qElemType} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60x!qElemType>

        return %call_fn2, %quant : tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>
    }

    // CHECK:  func.func private @function1([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD1]]
    // CHECK:  }
    // CHECK:  func.func private @function2([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType> {
    // CHECK:      [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      return [[ADD2]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x60x!qElemType>) -> (tensor<1x48x60x60x!qElemType>, tensor<1x48x60x60x!qElemType>) {
    // CHECK:      [[CALL1_FN1:%.+]] = call @function1([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:      [[CALL2_FN1:%.+]] = call @function1([[ARG0]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:      [[CALL_FN2:%.+]] = call @function2([[CALL1_FN1]]) : (tensor<1x48x60x60x!qElemType>) -> tensor<1x48x60x60x!qElemType>
    // CHECK:      return [[CALL_FN2]], [[CALL2_FN1]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @ConstReturnCall
module @ConstReturnCall {
    func.func private @function(%arg0: tensor<1x1xf16>) -> tensor<1x1xf16> {
        %cst = const.Declare tensor<1x1xf16> = dense<1.000000e+00> : tensor<1x1xf32>, [#const.CastElemType<f16>]
        return %cst : tensor<1x1xf16>
    }

    // CHECK:  func.func private @function([[ARG0:%.+]]: tensor<1x1xf16>) -> tensor<1x1xf16> {
    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x1xf16> = dense<1.000000e+00> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:      return [[CST]]
    // CHECK:  }
}

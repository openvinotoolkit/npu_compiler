//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU40XX
!ITIOutput0 = !VPUIP.ITIBuffer<1x48x50x170xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 48, 2, 170], offset = [0, 0, 48, 0], cluster_id = 0 : i64>]>
!ITIOutput1 = !VPUIP.ITIBuffer<1x48x47x170xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 48, 2, 170], offset = [0, 0, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 48, 2, 170], offset = [0, 0, 48, 0], cluster_id = 0 : i64>]>]>
// CHECK-LABEL @ITIBufferRoundTrip
func.func @ITIBufferRoundTrip(%arg0: !ITIOutput0, %arg1: !ITIOutput1) -> (!ITIOutput0, !ITIOutput1) {
  return %arg0, %arg1 : !ITIOutput0, !ITIOutput1
}

// CHECK:   return
// CHECK-SAME:    !VPUIP.ITIBuffer<
// CHECK-NEXT:      1x48x50x170xf16, {order = #NHWC}, [@CMX_NN, 0],
// CHECK-NEXT:    inwardHaloRegions = [
// CHECK-NEXT:      #VPUIP.HaloRegionAttr<shape = [1, 48, 2, 170], offset = [0, 0, 48, 0], cluster_id = 0 : i64>
// CHECK-NEXT:    ]>
// CHECK-SAME:    !VPUIP.ITIBuffer<
// CHECK-NEXT:    1x48x47x170xf16, {order = #NHWC}, [@CMX_NN, 1],
// CHECK-NEXT:    outwardHaloRegions = [
// CHECK-NEXT:      #VPUIP.OutwardHaloRegionAttr<shape = [1, 48, 2, 170], offset = [0, 0, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
// CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 48, 2, 170], offset = [0, 0, 48, 0], cluster_id = 0 : i64>
// CHECK-NEXT:      ]>
// CHECK-NEXT:    ]>

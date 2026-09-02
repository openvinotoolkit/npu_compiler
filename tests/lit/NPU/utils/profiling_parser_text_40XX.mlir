//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="platform=%platform% allow-custom-values=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --lower-VPUIP-to-ELF %data_path_npu%/profiling-40XX.mlir.txt | vpux-translate --platform=%platform% --export-ELF -o %t
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-40XX.bin -f text | FileCheck %s
// REQUIRES: platform-NPU4000

//CHECK: Task(DMA): Convolution_6?t_Convolution/reorder_in_0/PermuteQuantize/_expand_input       Time(us): 0.26          Start(us): 0.00
//CHECK: Task(DMA): Convolution_6?t_Convolution/reorder_in_0/PermuteQuantize/_expand_input/_expand_copy_3_14     Time(us): 0.26          Start(us): 0.26
//CHECK: Task(DMA): Convolution_6?t_Convolution/expand_act_channels/input-1-CMX  Time(us): 0.26          Start(us): 0.52
//CHECK: Task(DMA): Convolution_6?t_Convolution/expand_act_channels/input-2-CMX  Time(us): 0.26          Start(us): 0.78
//CHECK: Task(DPU): Convolution_6?t_Convolution/reorder_in_0/PermuteQuantize/cluster_0   Time(us): 0.06          Start(us): 7.75
//CHECK: Task(DMA): Convolution_6?t_Convolution/expand_act_channels/input-0-CMX  Time(us): 0.05          Start(us): 7.92
//CHECK: Task(DPU): Convolution_6?t_Convolution/expand_act_channels/cluster_0    Time(us): 0.08          Start(us): 7.94
//CHECK: Task(DMA): Conv_0?t_Convert/input-0-CMX                                 Time(us): 0.05          Start(us): 8.39
//CHECK: Task(SW): Conv_0?t_Convert/cluster_0                                    Time(us): 5.47          Cycles:379(5382)        Start(us): 35.83
//CHECK: Task(DMA): 495/sink_port_0?t_Result                                     Time(us): 0.05          Start(us): 44.32
//CHECK: Layer: Convolution_6                            Type: Convolution          DPU: 0.14     SW: 0.00     DMA: 1.09         Start: 0.00
//CHECK: Layer: Conv_0                                   Type: Convert              DPU: 0.00     SW: 5.47     DMA: 0.05         Start: 8.39
//CHECK: Layer: 495/sink_port_0                          Type: Result               DPU: 0.00     SW: 0.00     DMA: 0.05         Start: 44.32
//CHECK: Total time: 6.80us, Real: 44.38us

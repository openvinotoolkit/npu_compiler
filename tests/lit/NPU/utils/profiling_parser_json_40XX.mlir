//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="platform=%platform% allow-custom-values=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --lower-VPUIP-to-ELF %data_path_npu%/profiling-40XX.mlir.txt | vpux-translate --platform=%platform% --export-ELF -o %t
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-40XX.bin -f json | FileCheck %s
// REQUIRES: platform-NPU4000

//CHECK: {"traceEvents":[
//CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":0, "args": {"name" : "DMA"}},
//CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":0, "args": {"sort_index" : "0"}},
//CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":0, "tid":0, "args": {"name" : "DMA 0 DDR"}},
//CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":0, "tid":1, "args": {"name" : "DMA 0 CMX"}},
//CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":1, "args": {"name" : "Cluster (0)"}},
//CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":1, "args": {"sort_index" : "1"}},
//CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":1, "tid":2, "args": {"name" : "DPU"}},
//CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":1, "tid":3, "args": {"name" : "Shave"}},
//CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":2, "args": {"name" : "Layers"}},
//CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":2, "args": {"sort_index" : "2"}},
//CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":2, "tid":4, "args": {"name" : "Layers"}},
//CHECK-NEXT: {"name":"Convolution_6?t_Convolution/reorder_in_0/PermuteQuantize/_expand_input", "cat":"DMA", "ph":"X", "ts":0.000, "dur":0.260, "pid":0, "tid":0, "args":{"Source memory:": "DDR", "Destination memory:": "CMX", "Input tensor shape": "[32xui8]", "Output tensor shape": "[8x4xui8]", "Input tensor strides": "[32xui8]", "Output tensor strides": "[256x32xui8]", "Address": "0x8ffdf620", "Time to ready": "0ns", "Time to start": "0ns", "Transfer time": "260ns", "Time to finish": "52ns", "Link agent id": "2", "Channel id": "2", "Read stall cycles": "1", "Write stall cycles": "2", "Total bytes": "32", "Total cycles": "9"}},
//CHECK-NEXT: {"name":"Convolution_6?t_Convolution/reorder_in_0/PermuteQuantize/_expand_input/_expand_copy_3_14", "cat":"DMA", "ph":"X", "ts":0.261, "dur":0.260, "pid":0, "tid":0, "args":{"Source memory:": "DDR", "Destination memory:": "CMX", "Input tensor shape": "[224xui8]", "Output tensor shape": "[8x28xui8]", "Input tensor strides": "[224xui8]", "Output tensor strides": "[256x32xui8]", "Address": "0x8ffdf700", "Time to ready": "0ns", "Time to start": "0ns", "Transfer time": "260ns", "Time to finish": "52ns", "Link agent id": "2", "Channel id": "2", "Read stall cycles": "1", "Write stall cycles": "1", "Total bytes": "224", "Total cycles": "24"}},
//CHECK-NEXT: {"name":"Convolution_6?t_Convolution/expand_act_channels/input-1-CMX", "cat":"DMA", "ph":"X", "ts":0.521, "dur":0.260, "pid":0, "tid":0, "args":{"Source memory:": "DDR", "Destination memory:": "CMX", "Input tensor shape": "[512xui8]", "Output tensor shape": "[512xui8]", "Input tensor strides": "[512xui8]", "Output tensor strides": "[512xui8]", "Address": "0x8ffdf7e0", "Time to ready": "0ns", "Time to start": "0ns", "Transfer time": "260ns", "Time to finish": "52ns", "Link agent id": "2", "Channel id": "2", "Read stall cycles": "1", "Write stall cycles": "1", "Total bytes": "512", "Total cycles": "56"}},
//CHECK-NEXT: {"name":"Convolution_6?t_Convolution/expand_act_channels/input-2-CMX", "cat":"DMA", "ph":"X", "ts":0.781, "dur":0.260, "pid":0, "tid":0, "args":{"Source memory:": "DDR", "Destination memory:": "CMX", "Input tensor shape": "[256xui8]", "Output tensor shape": "[256xui8]", "Input tensor strides": "[256xui8]", "Output tensor strides": "[256xui8]", "Address": "0x8ffdf8c0", "Time to ready": "0ns", "Time to start": "0ns", "Transfer time": "260ns", "Time to finish": "52ns", "Link agent id": "2", "Channel id": "2", "Read stall cycles": "1", "Write stall cycles": "1", "Total bytes": "256", "Total cycles": "15"}},
//CHECK-NEXT: {"name":"Convolution_6?t_Convolution/expand_act_channels/input-0-CMX", "cat":"DMA", "ph":"X", "ts":7.917, "dur":0.052, "pid":0, "tid":1, "args":{"Source memory:": "CMX", "Destination memory:": "CMX", "Input tensor shape": "[2x16xui8]", "Output tensor shape": "[32xui8]", "Input tensor strides": "[256x128xui8]", "Output tensor strides": "[32xui8]", "Address": "0x8ffded00", "Time to ready": "8us 177ns", "Time to start": "0ns", "Transfer time": "52ns", "Time to finish": "52ns", "Link agent id": "5", "Channel id": "3", "Read stall cycles": "1", "Write stall cycles": "1", "Total bytes": "32", "Total cycles": "1"}},
//CHECK-NEXT: {"name":"Conv_0?t_Convert/input-0-CMX", "cat":"DMA", "ph":"X", "ts":8.386, "dur":0.052, "pid":0, "tid":1, "args":{"Source memory:": "CMX", "Destination memory:": "CMX", "Input tensor shape": "[8xui8]", "Output tensor shape": "[8xui8]", "Input tensor strides": "[32xui8]", "Output tensor strides": "[8xui8]", "Address": "0x8ffdeec0", "Time to ready": "0ns", "Time to start": "0ns", "Transfer time": "52ns", "Time to finish": "52ns", "Link agent id": "5", "Channel id": "3", "Read stall cycles": "0", "Write stall cycles": "3", "Total bytes": "8", "Total cycles": "1"}},
//CHECK-NEXT: {"name":"495/sink_port_0?t_Result", "cat":"DMA", "ph":"X", "ts":44.323, "dur":0.052, "pid":0, "tid":1, "args":{"Source memory:": "CMX", "Destination memory:": "DDR", "Input tensor shape": "[4xui8]", "Output tensor shape": "[4xui8]", "Input tensor strides": "[4xui8]", "Output tensor strides": "[4xui8]", "Address": "0x8ffdf080", "Time to ready": "0ns", "Time to start": "0ns", "Transfer time": "52ns", "Time to finish": "104ns", "Link agent id": "5", "Channel id": "3", "Read stall cycles": "0", "Write stall cycles": "3", "Total bytes": "4", "Total cycles": "1"}},
//CHECK-NEXT: {"name":"Convolution_6?t_Convolution/reorder_in_0/PermuteQuantize/cluster_0", "cat":"DPU", "ph":"X", "ts":7.751, "dur":0.061, "pid":1, "tid":2, "args":{"Input tensors": "[1x16x4x2xf16, 1x16x4x2xf16]", "Output tensors": "[1x16x4x2xf16]"}},
//CHECK-NEXT: {"name":"Convolution_6?t_Convolution/expand_act_channels/cluster_0", "cat":"DPU", "ph":"X", "ts":7.941, "dur":0.080, "pid":1, "tid":2, "args":{"Input tensors": "[1x16x2x2xf16, 16x16x2x2xf16]", "Output tensors": "[1x16x1x1xf16]"}},
//CHECK-NEXT: {"name":"Conv_0?t_Convert/cluster_0", "cat":"Shave", "ph":"X", "ts":35.833, "dur":5.468, "pid":1, "tid":3, "args":{"Total cycles": "5761", "Active cycles": "379", "Stall cycles": "5382", "LSU0 stalls": "372", "LSU1 stalls": "87", "Instruction stalls": "4936", "Input tensors": "[1x4x1x1xf16]", "Output tensors": "[1x4x1x1xui8]"}},
//CHECK-NEXT: {"name":"Convolution_6", "cat":"Layer", "ph":"X", "ts":0.000, "dur":8.021, "pid":2, "tid":4, "args":{"Layer type": "Convolution", "DPU time": "141ns", "DMA time": "1us 92ns"}},
//CHECK-NEXT: {"name":"Conv_0", "cat":"Layer", "ph":"X", "ts":8.386, "dur":32.915, "pid":2, "tid":4, "args":{"Layer type": "Convert", "Shave time": "5us 468ns", "DMA time": "52ns"}},
//CHECK-NEXT: {"name":"495/sink_port_0", "cat":"Layer", "ph":"X", "ts":44.323, "dur":0.052, "pid":2, "tid":4, "args":{"Layer type": "Result", "DMA time": "52ns"}}
//CHECK-NEXT: ],
//CHECK-NEXT: "taskStatistics": {
//CHECK-NEXT: "total duration":44.375,
//CHECK-NEXT: "DMA duration":1.196,
//CHECK-NEXT: "DPU duration":0.141,
//CHECK-NEXT: "SW duration":5.468,
//CHECK-NEXT: "M2I duration":0.000,
//CHECK-NEXT: "DMA-DPU overlap":0.028,
//CHECK-NEXT: "DMA-SW overlap":0.000,
//CHECK-NEXT: "SW-DPU overlap":0.000,
//CHECK-NEXT: "all tasks union":6.777,
//CHECK-NEXT: "total idle":37.598,
//CHECK-NEXT: "SW duration without DPU overlap":5.468,
//CHECK-NEXT: "DMA duration without overlaps":1.168,
//CHECK-NEXT: "Sum of DMA task durations":1.196,
//CHECK-NEXT: "Sum of DPU task durations":0.141,
//CHECK-NEXT: "Sum of SW task durations":5.468,
//CHECK-NEXT: "Sum of M2I task durations":0.000
//CHECK-NEXT: },
//CHECK-NEXT: "workpoint": { "freq": 1850.0, "status": "OK" },
//CHECK-NEXT: "displayTimeUnit": "ns"
//CHECK-NEXT: }

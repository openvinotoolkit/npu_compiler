//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="vpu-arch=%arch% allow-custom-values=true" --lower-VPUIP-to-ELF %data_path_npu%/profiling-37XX.mlir.txt | vpux-translate --vpu-arch=%arch% --export-ELF -o %t
// RUN: prof_parser -b %t -m | FileCheck %s
// REQUIRES: arch-NPU37XX

//CHECK: {
//CHECK: majorVersion: 2,
//CHECK: platform: {
//CHECK: device: 2
//CHECK: },
//CHECK: profilingBuffer: {
//CHECK: sections: [ {
//CHECK: type: 1,
//CHECK: size: 192
//CHECK: }, {
//CHECK: type: 3,
//CHECK: offset: 192,
//CHECK: size: 256
//CHECK: }, {
//CHECK: type: 4,
//CHECK: offset: 448,
//CHECK: size: 480
//CHECK: }, {
//CHECK: type: 5,
//CHECK: offset: 960,
//CHECK: size: 64
//CHECK: } ],
//CHECK: size: 1024
//CHECK: },
//CHECK: dmaTasks: [ {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 0 ]
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 0 ],
//CHECK: dataIndex: 16
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 0 ],
//CHECK: dataIndex: 1
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 0 ],
//CHECK: dataIndex: 17
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 1 ],
//CHECK: dataIndex: 2
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 1 ],
//CHECK: dataIndex: 18
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK: waitBarriers: [ 2 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 1 ],
//CHECK: dataIndex: 3
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK: waitBarriers: [ 2 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 1 ],
//CHECK: dataIndex: 19
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK: waitBarriers: [ 1 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 3 ],
//CHECK: dataIndex: 4
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK: waitBarriers: [ 1 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 3 ],
//CHECK: dataIndex: 20
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: dataIndex: 5
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: dataIndex: 21
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_fused_constant/_fused_tile",
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_fused_constant/_fused_tile",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: dataIndex: 6
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK: waitBarriers: [ 6 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: dataIndex: 7
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK: waitBarriers: [ 6 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: dataIndex: 22
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK: waitBarriers: [ 5 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 7 ],
//CHECK: dataIndex: 8
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK: waitBarriers: [ 5 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 7 ],
//CHECK: dataIndex: 23
//CHECK: }, {
//CHECK: name: "conv2/WithoutBiases?t_Convolution/_fused_constant/_fused_tile",
//CHECK: waitBarriers: [ 5 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv2/WithoutBiases?t_Convolution/_fused_constant/_fused_tile",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 8 ],
//CHECK: dataIndex: 9
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_0",
//CHECK: waitBarriers: [ 11 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 12 ],
//CHECK: dataIndex: 10
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_1",
//CHECK: waitBarriers: [ 11 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 12 ],
//CHECK: dataIndex: 24
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_0",
//CHECK: waitBarriers: [ 11 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 12 ],
//CHECK: dataIndex: 11
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_1",
//CHECK: waitBarriers: [ 11 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 12 ],
//CHECK: dataIndex: 25
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_0",
//CHECK: waitBarriers: [ 12 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 13 ],
//CHECK: dataIndex: 12
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_1",
//CHECK: waitBarriers: [ 12 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 13 ],
//CHECK: dataIndex: 26
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_0",
//CHECK: waitBarriers: [ 12 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 13 ],
//CHECK: dataIndex: 13
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_1",
//CHECK: waitBarriers: [ 12 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 13 ],
//CHECK: dataIndex: 27
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/_cluster_0",
//CHECK: waitBarriers: [ 14 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 14
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/_cluster_1",
//CHECK: waitBarriers: [ 14 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 28
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/_cluster_0",
//CHECK: waitBarriers: [ 14 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 15
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/_cluster_1",
//CHECK: waitBarriers: [ 14 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 29
//CHECK: } ],
//CHECK: dpuTasks: [ {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/cluster_0",
//CHECK: taskId: 1,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 6 ],
//CHECK: workloadIds: [ 0 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 1,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 6 ],
//CHECK: workloadIds: [ 0 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_0",
//CHECK: taskId: 2,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 7 ],
//CHECK: updateBarriers: [ 9 ],
//CHECK: workloadIds: [ 0 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 2,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 7 ],
//CHECK: updateBarriers: [ 9 ],
//CHECK: workloadIds: [ 0 ]
//CHECK: }, {
//CHECK: name: "relu1?t_Relu/cluster_0",
//CHECK: taskId: 3,
//CHECK: numVariants: 2,
//CHECK: maxVariants: 2,
//CHECK: waitBarriers: [ 9 ],
//CHECK: updateBarriers: [ 8 ],
//CHECK: workloadIds: [ 0, 1 ]
//CHECK: }, {
//CHECK: name: "relu1?t_Relu/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 3,
//CHECK: numVariants: 2,
//CHECK: maxVariants: 2,
//CHECK: waitBarriers: [ 9 ],
//CHECK: updateBarriers: [ 8 ],
//CHECK: workloadIds: [ 0, 1 ]
//CHECK: }, {
//CHECK: name: "conv2/WithoutBiases?t_Convolution/cluster_0",
//CHECK: taskId: 4,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 8 ],
//CHECK: updateBarriers: [ 10 ],
//CHECK: workloadIds: [ 0 ]
//CHECK: }, {
//CHECK: name: "conv2/WithoutBiases?t_Convolution/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 4,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 8 ],
//CHECK: updateBarriers: [ 10 ],
//CHECK: workloadIds: [ 0 ]
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/cluster_0",
//CHECK: taskId: 5,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 10 ],
//CHECK: updateBarriers: [ 11 ],
//CHECK: workloadIds: [ 0 ]
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 5,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 10 ],
//CHECK: updateBarriers: [ 11 ],
//CHECK: workloadIds: [ 0 ]
//CHECK: } ],
//CHECK: swTasks: [ {
//CHECK: name: "data?t_Parameter/converted_to_f16/tile_0/cluster_0",
//CHECK: waitBarriers: [ 0 ],
//CHECK: updateBarriers: [ 2 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 4
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/tile_0/cluster_1",
//CHECK: waitBarriers: [ 0 ],
//CHECK: updateBarriers: [ 2 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 4,
//CHECK: clusterId: 1
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/tile_1/cluster_0",
//CHECK: waitBarriers: [ 0 ],
//CHECK: updateBarriers: [ 2 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 4,
//CHECK: dataIndex: 1,
//CHECK: tileId: 1
//CHECK: }, {
//CHECK: name: "data?t_Parameter/converted_to_f16/tile_1/cluster_1",
//CHECK: waitBarriers: [ 0 ],
//CHECK: updateBarriers: [ 2 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 4,
//CHECK: dataIndex: 1,
//CHECK: tileId: 1,
//CHECK: clusterId: 1
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_0/cluster_0",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 4,
//CHECK: dataIndex: 2
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_0/cluster_1",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 4,
//CHECK: dataIndex: 2,
//CHECK: clusterId: 1
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_1/cluster_0",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 4,
//CHECK: dataIndex: 3,
//CHECK: tileId: 1
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_1/cluster_1",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 4,
//CHECK: dataIndex: 3,
//CHECK: tileId: 1,
//CHECK: clusterId: 1
//CHECK: } ]
//CHECK: }

//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="vpu-arch=%arch% allow-custom-values=true" --lower-VPUIP-to-ELF %data_path_npu%/profiling-37XX.mlir.txt | vpux-translate --vpu-arch=%arch% --export-ELF -o %t
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-37XX.bin -f text | FileCheck %s
// REQUIRES: arch-NPU37XX

//CHECK: Task(DMA): data?t_Parameter/converted_to_f16/_cluster_0                	Time(us): 0.86    	Start(us): 0.00
//CHECK: Task(DMA): data?t_Parameter/converted_to_f16/_cluster_0                	Time(us): 0.88    	Start(us): 1.09
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/_cluster_0	Time(us): 0.55    	Start(us): 2.21
//CHECK: Task(DMA): data?t_Parameter/converted_to_f16/_cluster_1                	Time(us): 0.86    	Start(us): 3.78
//CHECK: Task(DMA): data?t_Parameter/converted_to_f16/_cluster_1                	Time(us): 0.91    	Start(us): 4.87
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/_cluster_1	Time(us): 0.55    	Start(us): 6.01
//CHECK: Task(SW): data?t_Parameter/converted_to_f16/tile_0/cluster_1          	Time(us): 7.11    	Cycles:6776(1092)	Start(us): 17.40
//CHECK: Task(SW): data?t_Parameter/converted_to_f16/tile_0/cluster_0          	Time(us): 6.85    	Cycles:6765(1201)	Start(us): 17.53
//CHECK: Task(SW): data?t_Parameter/converted_to_f16/tile_1/cluster_0          	Time(us): 6.98    	Cycles:6862(1276)	Start(us): 17.66
//CHECK: Task(SW): data?t_Parameter/converted_to_f16/tile_1/cluster_1          	Time(us): 6.98    	Cycles:6860(1441)	Start(us): 17.79
//CHECK: Task(DMA): data?t_Parameter/converted_to_f16/_cluster_0                	Time(us): 0.55    	Start(us): 25.73
//CHECK: Task(DMA): data?t_Parameter/converted_to_f16/_cluster_1                	Time(us): 0.23    	Start(us): 25.89
//CHECK: Task(DMA): data?t_Parameter/converted_to_f16/_cluster_1                	Time(us): 0.42    	Start(us): 26.56
//CHECK: Task(DMA): data?t_Parameter/converted_to_f16/_cluster_0                	Time(us): 0.42    	Start(us): 26.72
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_cluster_1                	Time(us): 0.86    	Start(us): 27.37
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_cluster_0                	Time(us): 0.99    	Start(us): 27.53
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_fused_constant/_fused_tile	Time(us): 0.65    	Start(us): 28.75
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_0                 	Time(us): 0.65    	Start(us): 29.18
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_1                 	Time(us): 0.65    	Start(us): 29.19
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_cluster_1                	Time(us): 2.86    	Start(us): 30.96
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_cluster_0                	Time(us): 2.55    	Start(us): 31.12
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_cluster_0                	Time(us): 2.66    	Start(us): 34.06
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_cluster_1                	Time(us): 2.66    	Start(us): 34.22
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_1    	Time(us): 7.92    	Start(us): 36.87
//CHECK: Task(DMA): conv2/WithoutBiases?t_Convolution/_fused_constant/_fused_tile	Time(us): 2.50    	Start(us): 37.03
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_0    	Time(us): 6.79    	Start(us): 37.05
//CHECK: Task(DPU): relu1?t_Relu/cluster_0                                      	Time(us): 10.10   	Start(us): 45.68
//CHECK: Task(DPU): relu1?t_Relu/cluster_1                                      	Time(us): 6.89    	Start(us): 45.83
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_1                 	Time(us): 10.76   	Start(us): 55.90
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_0                 	Time(us): 10.19   	Start(us): 56.39
//CHECK: Task(DPU): relu2?t_Relu/cluster_0                                      	Time(us): 4.36    	Start(us): 67.20
//CHECK: Task(DPU): relu2?t_Relu/cluster_1                                      	Time(us): 3.54    	Start(us): 67.31
//CHECK: Task(DMA): relu2?t_Relu/_cluster_1                                     	Time(us): 0.96    	Start(us): 72.37
//CHECK: Task(DMA): relu2?t_Relu/_cluster_0                                     	Time(us): 0.96    	Start(us): 72.53
//CHECK: Task(DMA): relu2?t_Relu/_cluster_1                                     	Time(us): 0.70    	Start(us): 73.64
//CHECK: Task(DMA): relu2?t_Relu/_cluster_0                                     	Time(us): 0.70    	Start(us): 73.80
//CHECK: Task(DMA): relu2?t_Relu/_cluster_1                                     	Time(us): 1.07    	Start(us): 74.74
//CHECK: Task(DMA): relu2?t_Relu/_cluster_0                                     	Time(us): 1.12    	Start(us): 74.89
//CHECK: Task(DMA): relu2?t_Relu/_cluster_1                                     	Time(us): 0.91    	Start(us): 76.17
//CHECK: Task(DMA): relu2?t_Relu/_cluster_0                                     	Time(us): 1.01    	Start(us): 76.33
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_1              	Time(us): 4.92    	Cycles:4665(861)	Start(us): 77.73
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_0              	Time(us): 4.92    	Cycles:4639(973)	Start(us): 77.86
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_1              	Time(us): 4.92    	Cycles:4615(1114)	Start(us): 77.99
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_0              	Time(us): 4.40    	Cycles:4593(1344)	Start(us): 78.12
//CHECK: Task(DMA): relu2?t_Relu/converted_to_f32/_cluster_1                    	Time(us): 1.17    	Start(us): 83.59
//CHECK: Task(DMA): relu2?t_Relu/converted_to_f32/_cluster_0                    	Time(us): 0.86    	Start(us): 83.75
//CHECK: Task(DMA): relu2?t_Relu/converted_to_f32/_cluster_0                    	Time(us): 0.62    	Start(us): 84.92
//CHECK: Task(DMA): relu2?t_Relu/converted_to_f32/_cluster_1                    	Time(us): 0.62    	Start(us): 85.08
//CHECK: Layer: data                                     Type: Parameter            DPU: 0.00     SW: 27.92    DMA: 5.13    	Start: 0.00
//CHECK: Layer: conv1/WithoutBiases                      Type: Convolution          DPU: 16.02    SW: 0.00     DMA: 14.32   	Start: 2.21
//CHECK: Layer: conv2/WithoutBiases                      Type: Convolution          DPU: 20.95    SW: 0.00     DMA: 2.50    	Start: 37.03
//CHECK: Layer: relu1                                    Type: Relu                 DPU: 16.99    SW: 0.00     DMA: 0.00    	Start: 45.68
//CHECK: Layer: relu2                                    Type: Relu                 DPU: 7.89     SW: 19.16    DMA: 10.72   	Start: 67.20
//CHECK: Total time: 141.60us, Real: 85.70us

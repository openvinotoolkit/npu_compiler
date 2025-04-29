//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="vpu-arch=%arch% allow-custom-values=true" --lower-VPUIP-to-ELF %data_path_npu%/profiling-40XX.mlir.txt | vpux-translate --vpu-arch=%arch% --export-ELF -o %t
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-40XX.bin -f text | FileCheck %s
// REQUIRES: arch-NPU40XX

//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_0                 	Time(us): 0.21    	Start(us): 0.00
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_1                 	Time(us): 0.21    	Start(us): 1.10
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_2                 	Time(us): 0.20    	Start(us): 1.93
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_3                 	Time(us): 0.20    	Start(us): 2.76
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_4                 	Time(us): 0.20    	Start(us): 3.60
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_5                 	Time(us): 0.15    	Start(us): 4.43
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_3    	Time(us): 1.89    	Start(us): 7.69
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_2    	Time(us): 1.89    	Start(us): 7.69
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_4    	Time(us): 1.89    	Start(us): 7.69
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_5    	Time(us): 1.89    	Start(us): 7.69
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_0    	Time(us): 1.89    	Start(us): 7.69
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_1    	Time(us): 1.89    	Start(us): 7.69
//CHECK: Task(DPU): relu1?t_Relu/cluster_5                                      	Time(us): 1.53    	Start(us): 9.62
//CHECK: Task(DPU): relu1?t_Relu/cluster_2                                      	Time(us): 1.67    	Start(us): 9.63
//CHECK: Task(DPU): relu1?t_Relu/cluster_0                                      	Time(us): 1.61    	Start(us): 9.64
//CHECK: Task(DPU): relu1?t_Relu/cluster_1                                      	Time(us): 1.66    	Start(us): 9.64
//CHECK: Task(DPU): relu1?t_Relu/cluster_4                                      	Time(us): 1.66    	Start(us): 9.64
//CHECK: Task(DPU): relu1?t_Relu/cluster_3                                      	Time(us): 1.60    	Start(us): 9.65
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_3                 	Time(us): 3.65    	Start(us): 11.30
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_0                 	Time(us): 3.64    	Start(us): 11.30
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_1                 	Time(us): 3.64    	Start(us): 11.30
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_2                 	Time(us): 3.64    	Start(us): 11.30
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_4                 	Time(us): 1.84    	Start(us): 11.34
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_5                 	Time(us): 1.84    	Start(us): 11.34
//CHECK: Task(DPU): relu2?t_Relu/cluster_2                                      	Time(us): 0.98    	Start(us): 14.96
//CHECK: Task(DPU): relu2?t_Relu/cluster_5                                      	Time(us): 0.98    	Start(us): 14.96
//CHECK: Task(DPU): relu2?t_Relu/cluster_4                                      	Time(us): 0.97    	Start(us): 14.97
//CHECK: Task(DPU): relu2?t_Relu/cluster_0                                      	Time(us): 0.94    	Start(us): 15.00
//CHECK: Task(DPU): relu2?t_Relu/cluster_3                                      	Time(us): 0.93    	Start(us): 15.01
//CHECK: Task(DPU): relu2?t_Relu/cluster_1                                      	Time(us): 0.93    	Start(us): 15.01
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_0              	Time(us): 11.93   	Cycles:12119(11448)	Start(us): 43.75
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_1              	Time(us): 11.56   	Cycles:12105(11274)	Start(us): 43.90
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_0              	Time(us): 11.77   	Cycles:12112(11448)	Start(us): 43.90
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_1              	Time(us): 10.94   	Cycles:11106(10617)	Start(us): 44.69
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_2              	Time(us): 10.47   	Cycles:10588(9912)	Start(us): 45.21
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_2              	Time(us): 10.00   	Cycles:10440(9638)	Start(us): 45.47
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_3              	Time(us): 9.01    	Cycles:9065(8584)	Start(us): 46.61
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_3              	Time(us): 8.07    	Cycles:8069(7589)	Start(us): 47.60
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_4              	Time(us): 5.73    	Cycles:5802(5348)	Start(us): 49.48
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_4              	Time(us): 4.11    	Cycles:4051(3581)	Start(us): 51.15
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_5              	Time(us): 3.64    	Cycles:3836(3186)	Start(us): 53.18
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_5              	Time(us): 3.64    	Cycles:3593(2954)	Start(us): 54.53
//CHECK: Layer: conv1/WithoutBiases                      Type: Convolution          DPU: 12.54    SW: 0.00     DMA: 0.00    	Start: 0.00
//CHECK: Layer: relu1                                    Type: Relu                 DPU: 9.73     SW: 0.00     DMA: 0.00    	Start: 9.62
//CHECK: Layer: conv2/WithoutBiases                      Type: Convolution          DPU: 18.26    SW: 0.00     DMA: 0.00    	Start: 11.30
//CHECK: Layer: relu2                                    Type: Relu                 DPU: 5.72     SW: 100.88   DMA: 0.00    	Start: 14.96
//CHECK: Total time: 147.12us, Real: 58.17us

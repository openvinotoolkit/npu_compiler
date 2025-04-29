//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="vpu-arch=%arch% allow-custom-values=true" --lower-VPUIP-to-ELF %data_path_npu%/profiling-37XX.mlir.txt | vpux-translate --vpu-arch=%arch% --export-ELF -o %t
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-37XX.bin -f debug | FileCheck %s
// REQUIRES: arch-NPU37XX

//CHECK:   Index  Offset        Engine  Buffer ID         Cluster ID      Buffer offset    IDU dur IDU tstamp SWE ID Rvd    ODU dur ODU tstamp    Rvd  Task
//CHECK:       0       0           dpu          0                  0                  0        24c        253      0   0        352        318      0  conv1/WithoutBiases?t_Convolution/cluster_0/variant_0
//CHECK:       1      10           dpu          0                  0                 10       1e40       3541      0   0       2280       3872      0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_0/variant_0
//CHECK:       2      20           dpu          0                  0                 20       20b8       57f8      0   0       239e       5a26      0  relu1?t_Relu/cluster_0/variant_0
//CHECK:       3      30           dpu          0                  0                 30       10ba       6487      0   0        fa0       65e1      0  relu1?t_Relu/cluster_0/variant_1
//CHECK:       4      40           dpu          0                  0                 40       2e8b       8b24      0   0       33c1       8f0d      0  conv2/WithoutBiases?t_Convolution/cluster_0/variant_0
//CHECK:       5      50           dpu          0                  0                 50        e55       9c20      0   0       1620       a1f9      0  relu2?t_Relu/cluster_0/variant_0
//CHECK:       6      60           dpu          0                  1                  0        246        254      1   0        34c        319      0  conv1/WithoutBiases?t_Convolution/cluster_1/variant_0
//CHECK:       7      70           dpu          0                  1                 10       23fe       38e3      1   0       283a       3c11      0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_1/variant_0
//CHECK:       8      80           dpu          0                  1                 20       1573       5019      1   0       182f       5227      0  relu1?t_Relu/cluster_1/variant_0
//CHECK:       9      90           dpu          0                  1                 30        bf2       5911      1   0        ad1       5a46      0  relu1?t_Relu/cluster_1/variant_1
//CHECK:      10      a0           dpu          0                  1                 40       316f       8b6a      1   0       36a5       8f53      0  conv2/WithoutBiases?t_Convolution/cluster_1/variant_0
//CHECK:      11      b0           dpu          0                  1                 50        a67       999c      1   0       11f3       9f46      0  relu2?t_Relu/cluster_1/variant_0

//CHECK:   Index  Offset        Engine  Buffer ID         Cluster ID      Buffer offset              Begin   Duration      Stall   Executed      Clock     Branch  Task
//CHECK:       0      c0      actshave          0                  0                  0           27621ba2        107        4b1        93f       1a6d         39  data?t_Parameter/converted_to_f16/tile_0/cluster_0
//CHECK:       1      e0      actshave          0                  0                 20           27621ba7        10c        4fc        932       1ace         3a  data?t_Parameter/converted_to_f16/tile_1/cluster_0
//CHECK:       2     100      actshave          0                  0                 40           276224af         bd        3cd        8b5       121f         42  relu2?t_Relu/converted_to_f32/tile_0/cluster_0
//CHECK:       3     120      actshave          0                  0                 60           276224b9         a9        540        6e7       11f1         34  relu2?t_Relu/converted_to_f32/tile_1/cluster_0
//CHECK:       4     140      actshave          0                  1                  0           27621b9d        111        444        93f       1a78         39  data?t_Parameter/converted_to_f16/tile_0/cluster_1
//CHECK:       5     160      actshave          0                  1                 20           27621bac        10c        5a1        932       1acc         3a  data?t_Parameter/converted_to_f16/tile_1/cluster_1
//CHECK:       6     180      actshave          0                  1                 40           276224b4         bd        45a        8b5       1207         42  relu2?t_Relu/converted_to_f32/tile_0/cluster_1
//CHECK:       7     1a0      actshave          0                  1                 60           276224aa         bd        35d        6e7       1239         34  relu2?t_Relu/converted_to_f32/tile_1/cluster_1

//CHECK:   Index  Offset        Engine       Begin tstamp         End tstamp  Task
//CHECK:       0     1c0           dma           27621901           27621922  data?t_Parameter/converted_to_f16/_cluster_0
//CHECK:       1     1d0           dma           2762192b           2762194d  data?t_Parameter/converted_to_f16/_cluster_0
//CHECK:       2     1e0           dma           27621956           2762196b  conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/_cluster_0
//CHECK:       3     1f0           dma           27621cdd           27621cf2  data?t_Parameter/converted_to_f16/_cluster_0
//CHECK:       4     200           dma           27621d03           27621d13  data?t_Parameter/converted_to_f16/_cluster_0
//CHECK:       5     210           dma           27621d22           27621d48  conv1/WithoutBiases?t_Convolution/_cluster_0
//CHECK:       6     220           dma           27621d51           27621d6a  conv1/WithoutBiases?t_Convolution/_fused_constant/_fused_tile
//CHECK:       7     230           dma           27621dac           27621e0e  conv1/WithoutBiases?t_Convolution/_cluster_0
//CHECK:       8     240           dma           27621e1d           27621e83  conv1/WithoutBiases?t_Convolution/_cluster_0
//CHECK:       9     250           dma           27621e8f           27621eef  conv2/WithoutBiases?t_Convolution/_fused_constant/_fused_tile
//CHECK:      10     260           dma           276223e2           27622407  relu2?t_Relu/_cluster_0
//CHECK:      11     270           dma           27622413           2762242e  relu2?t_Relu/_cluster_0
//CHECK:      12     280           dma           2762243d           27622468  relu2?t_Relu/_cluster_0
//CHECK:      13     290           dma           27622474           2762249b  relu2?t_Relu/_cluster_0
//CHECK:      14     2a0           dma           27622591           276225b2  relu2?t_Relu/converted_to_f32/_cluster_0
//CHECK:      15     2b0           dma           276225be           276225d6  relu2?t_Relu/converted_to_f32/_cluster_0
//CHECK:      16     2c0           dma           27621992           276219b3  data?t_Parameter/converted_to_f16/_cluster_1
//CHECK:      17     2d0           dma           276219bc           276219df  data?t_Parameter/converted_to_f16/_cluster_1
//CHECK:      18     2e0           dma           276219e8           276219fd  conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/_cluster_1
//CHECK:      19     2f0           dma           27621ce3           27621cec  data?t_Parameter/converted_to_f16/_cluster_1
//CHECK:      20     300           dma           27621cfd           27621d0d  data?t_Parameter/converted_to_f16/_cluster_1
//CHECK:      21     310           dma           27621d1c           27621d3d  conv1/WithoutBiases?t_Convolution/_cluster_1
//CHECK:      22     320           dma           27621da6           27621e14  conv1/WithoutBiases?t_Convolution/_cluster_1
//CHECK:      23     330           dma           27621e23           27621e89  conv1/WithoutBiases?t_Convolution/_cluster_1
//CHECK:      24     340           dma           276223dc           27622401  relu2?t_Relu/_cluster_1
//CHECK:      25     350           dma           2762240d           27622428  relu2?t_Relu/_cluster_1
//CHECK:      26     360           dma           27622437           27622460  relu2?t_Relu/_cluster_1
//CHECK:      27     370           dma           2762246e           27622491  relu2?t_Relu/_cluster_1
//CHECK:      28     380           dma           2762258b           276225b8  relu2?t_Relu/converted_to_f32/_cluster_1
//CHECK:      29     390           dma           276225c4           276225dc  relu2?t_Relu/converted_to_f32/_cluster_1

//CHECK:   Index  Offset        Engine        PLL Value          CFGID
//CHECK:       0     3c0           pll               27            202
//CHECK:       1     3c4           pll               27            202

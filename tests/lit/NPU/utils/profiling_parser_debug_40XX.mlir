//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="vpu-arch=%arch% allow-custom-values=true" --lower-VPUIP-to-ELF %data_path_npu%/profiling-40XX.mlir.txt | vpux-translate --vpu-arch=%arch% --export-ELF -o %t
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-40XX.bin -f debug | FileCheck %s
// REQUIRES: arch-NPU40XX

//CHECK:    Index  Offset        Engine  Buffer ID         Cluster ID      Buffer offset    IDU dur         IDU tstamp  IDU WL ID  IDU DPU ID    ODU dur         ODU tstamp  ODU WL ID  ODU DPU ID  Task
//CHECK:        0       0           dpu          0                  0                  0         be           1882dd5b         10           0        177   1882dd5d                 10          0  conv1/WithoutBiases?t_Convolution/cluster_0/variant_0
//CHECK:        1      20           dpu          0                  0                 20        9eb           1882de07         11           0        dbf   1882de11                 11          0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_0/variant_0
//CHECK:        2      40           dpu          0                  0                 40        661           1882de23         12           0        7d6   1882de27                 12          0  relu1?t_Relu/cluster_0/variant_0
//CHECK:        3      60           dpu          0                  0                 60        317           1882de2f         13           0        403   1882de31                 13          0  relu1?t_Relu/cluster_0/variant_1
//CHECK:        4      80           dpu          0                  0                 80       16f1           1882de6f         14           0       1a1e   1882de78                 14          0  conv2/WithoutBiases?t_Convolution/cluster_0/variant_0
//CHECK:        5      a0           dpu          0                  0                 a0        306           1882de81         15           0        6ee   1882de8b                 15          0  relu2?t_Relu/cluster_0/variant_0
//CHECK:        6      c0           dpu          0                  1                  0         bd           1882dd70         10           0        178   1882dd72                 10          0  conv1/WithoutBiases?t_Convolution/cluster_1/variant_0
//CHECK:        7      e0           dpu          0                  1                 20        9eb           1882de07         11           0        dbf   1882de11                 11          0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_1/variant_0
//CHECK:        8     100           dpu          0                  1                 40        661           1882de23         12           0        7d6   1882de27                 12          0  relu1?t_Relu/cluster_1/variant_0
//CHECK:        9     120           dpu          0                  1                 60        31e           1882de2f         13           0        40a   1882de32                 13          0  relu1?t_Relu/cluster_1/variant_1
//CHECK:       10     140           dpu          0                  1                 80       16f1           1882de6f         14           0       1a1e   1882de78                 14          0  conv2/WithoutBiases?t_Convolution/cluster_1/variant_0
//CHECK:       11     160           dpu          0                  1                 a0        2f3           1882de81         15           0        6db   1882de8b                 15          0  relu2?t_Relu/cluster_1/variant_0
//CHECK:       12     180           dpu          0                  2                  0         b9           1882dd80         10           0        165   1882dd82                 10          0  conv1/WithoutBiases?t_Convolution/cluster_2/variant_0
//CHECK:       13     1a0           dpu          0                  2                 20        9ed           1882de07         11           0        dc1   1882de11                 11          0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_2/variant_0
//CHECK:       14     1c0           dpu          0                  2                 40        666           1882de23         12           0        7de   1882de27                 12          0  relu1?t_Relu/cluster_2/variant_0
//CHECK:       15     1e0           dpu          0                  2                 60        329           1882de2f         13           0        411   1882de32                 13          0  relu1?t_Relu/cluster_2/variant_1
//CHECK:       16     200           dpu          0                  2                 80       16f0           1882de6f         14           0       1a1d   1882de78                 14          0  conv2/WithoutBiases?t_Convolution/cluster_2/variant_0
//CHECK:       17     220           dpu          0                  2                 a0        2ed           1882de80         15           0        6d0   1882de8b                 15          0  relu2?t_Relu/cluster_2/variant_0
//CHECK:       18     240           dpu          0                  3                  0         ba           1882dd90         10           0        169   1882dd92                 10          0  conv1/WithoutBiases?t_Convolution/cluster_3/variant_0
//CHECK:       19     260           dpu          0                  3                 20        9ef           1882de07         11           0        dc3   1882de11                 11          0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_3/variant_0
//CHECK:       20     280           dpu          0                  3                 40        64e           1882de23         12           0        7c4   1882de27                 12          0  relu1?t_Relu/cluster_3/variant_0
//CHECK:       21     2a0           dpu          0                  3                 60        328           1882de2f         13           0        410   1882de31                 13          0  relu1?t_Relu/cluster_3/variant_1
//CHECK:       22     2c0           dpu          0                  3                 80       16f6           1882de6f         14           0       1a23   1882de78                 14          0  conv2/WithoutBiases?t_Convolution/cluster_3/variant_0
//CHECK:       23     2e0           dpu          0                  3                 a0        2f5           1882de81         15           0        6d5   1882de8b                 15          0  relu2?t_Relu/cluster_3/variant_0
//CHECK:       24     300           dpu          0                  4                  0         ba           1882dda0         10           0        169   1882dda2                 10          0  conv1/WithoutBiases?t_Convolution/cluster_4/variant_0
//CHECK:       25     320           dpu          0                  4                 20        9ed           1882de07         11           0        dc1   1882de11                 11          0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_4/variant_0
//CHECK:       26     340           dpu          0                  4                 40        654           1882de23         12           0        7cd   1882de27                 12          0  relu1?t_Relu/cluster_4/variant_0
//CHECK:       27     360           dpu          0                  4                 60        32a           1882de2f         13           0        416   1882de32                 13          0  relu1?t_Relu/cluster_4/variant_1
//CHECK:       28     380           dpu          0                  4                 80        b6d           1882de51         14           0        d6c   1882de56                 14          0  conv2/WithoutBiases?t_Convolution/cluster_4/variant_0
//CHECK:       29     3a0           dpu          0                  4                 a0        2e1           1882de80         15           0        6c9   1882de8b                 15          0  relu2?t_Relu/cluster_4/variant_0
//CHECK:       30     3c0           dpu          0                  5                  0         b9           1882ddb0         10           0        165   1882ddb1                 10          0  conv1/WithoutBiases?t_Convolution/cluster_5/variant_0
//CHECK:       31     3e0           dpu          0                  5                 20        9ed           1882de07         11           0        dc1   1882de11                 11          0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_5/variant_0
//CHECK:       32     400           dpu          0                  5                 40        5c7           1882de21         12           0        72a   1882de25                 12          0  relu1?t_Relu/cluster_5/variant_0
//CHECK:       33     420           dpu          0                  5                 60        2e1           1882de2d         13           0        3ba   1882de2f                 13          0  relu1?t_Relu/cluster_5/variant_1
//CHECK:       34     440           dpu          0                  5                 80        b6d           1882de51         14           0        d6c   1882de56                 14          0  conv2/WithoutBiases?t_Convolution/cluster_5/variant_0
//CHECK:       35     460           dpu          0                  5                 a0        2e9           1882de80         15           0        6d0   1882de8b                 15          0  relu2?t_Relu/cluster_5/variant_0

//CHECK:    Index  Offset        Engine  Buffer ID         Cluster ID      Buffer offset              Begin   Duration   Executed      Clock        LSU0 Stalls        LSU1 Stalls       Instr Stalls  Task
//CHECK:        0     480      actshave          0                  0                  0           1882e0a1         e5        378       2f57                2e1                 3a               299d  relu2?t_Relu/converted_to_f32/tile_0/cluster_0
//CHECK:        1     4a0      actshave          0                  0                 20           1882e0a4         e2        2b4       2f50                37b                 3a               2903  relu2?t_Relu/converted_to_f32/tile_1/cluster_0
//CHECK:        2     4c0      actshave          0                  1                  0           1882e0a4         de        2b4       2f49                2b8                 37               291b  relu2?t_Relu/converted_to_f32/tile_0/cluster_1
//CHECK:        3     4e0      actshave          0                  1                 20           1882e0b3         d2        2b4       2b62                34d                 37               25f5  relu2?t_Relu/converted_to_f32/tile_1/cluster_1
//CHECK:        4     500      actshave          0                  2                  0           1882e0bd         c9        378       295c                362                 39               231d  relu2?t_Relu/converted_to_f32/tile_0/cluster_2
//CHECK:        5     520      actshave          0                  2                 20           1882e0c2         c0        2b4       28c8                2ba                 37               22b5  relu2?t_Relu/converted_to_f32/tile_1/cluster_2
//CHECK:        6     540      actshave          0                  3                  0           1882e0d8         ad        2b4       2369                2a5                 35               1eae  relu2?t_Relu/converted_to_f32/tile_0/cluster_3
//CHECK:        7     560      actshave          0                  3                 20           1882e0eb         9b        2b4       1f85                2a5                 36               1aca  relu2?t_Relu/converted_to_f32/tile_1/cluster_3
//CHECK:        8     580      actshave          0                  4                  0           1882e10f         6e        2b4       16aa                2a1                 3e               1205  relu2?t_Relu/converted_to_f32/tile_0/cluster_4
//CHECK:        9     5a0      actshave          0                  4                 20           1882e12f         4f        2b4        fd3                29a                 37                b2c  relu2?t_Relu/converted_to_f32/tile_1/cluster_4
//CHECK:       10     5c0      actshave          0                  5                  0           1882e156         46        2b4        efc                288                 3a                9b0  relu2?t_Relu/converted_to_f32/tile_0/cluster_5
//CHECK:       11     5e0      actshave          0                  5                 20           1882e170         46        2b4        e09                483                 3b                6cc  relu2?t_Relu/converted_to_f32/tile_1/cluster_5

//CHECK:    Index  Offset        Engine        PLL Value          CFGID
//CHECK:        0     600           pll               4a              6
//CHECK:        1     604           pll               4a              6

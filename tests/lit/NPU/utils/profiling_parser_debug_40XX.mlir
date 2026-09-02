//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="platform=%platform% allow-custom-values=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --lower-VPUIP-to-ELF %data_path_npu%/profiling-40XX.mlir.txt | vpux-translate --platform=%platform% --export-ELF -o %t
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-40XX.bin -f debug | FileCheck %s
// REQUIRES: platform-NPU4000

//CHECK:    Index  Offset        Engine  Buffer ID         Cluster ID      Buffer offset    IDU dur         IDU tstamp  IDU WL ID  IDU DPU ID    ODU dur         ODU tstamp  ODU WL ID  ODU DPU ID  Task
//CHECK:    0       0           dpu          0                  0                  0         11            e56476c          0           0         76    e56476d                  0          0  Convolution_6?t_Convolution/reorder_in_0/PermuteQuantize/cluster_0/variant_0
//CHECK:    1      20           dpu          0                  0                 20         34            e564770          1           0         98    e564771                  1          0  Convolution_6?t_Convolution/expand_act_channels/cluster_0/variant_0

//CHECK:    Index  Offset        Engine  Buffer ID         Cluster ID      Buffer offset              Begin   Duration   Executed      Clock        LSU0 Stalls        LSU1 Stalls       Instr Stalls  Task
//CHECK:    0      40      actshave          0                  0                  0            e564987         69        17b       1681                174                 57               1348  Conv_0?t_Convert/cluster_0

//CHECK:    Index  Offset        Engine         JDESC_ADDR        JFETCH_TIME        JREADY_TIME        JSTART_TIME        JWDONE_TIME       JFINISH_TIME JLA_ID JCH_ID   RSVD  JRSTALL_CNT  JWSTALL_CNT  JTWBYTES_CNT  JCHCYCLE_CNT  Task
//CHECK:    0      c0         dmahw           8ffdf620            e5646d7            e5646d7            e5646d7            e5646dc            e5646dd      2      2      0            1            2            20             9  Convolution_6?t_Convolution/reorder_in_0/PermuteQuantize/_expand_input
//CHECK:    1     100         dmahw           8ffdf700            e5646dc            e5646dc            e5646dc            e5646e1            e5646e2      2      2      0            1            1            e0            18  Convolution_6?t_Convolution/reorder_in_0/PermuteQuantize/_expand_input/_expand_copy_3_14
//CHECK:    2     140         dmahw           8ffdf7e0            e5646e1            e5646e1            e5646e1            e5646e6            e5646e7      2      2      0            1            1           200            38  Convolution_6?t_Convolution/expand_act_channels/input-1-CMX
//CHECK:    3     180         dmahw           8ffded00            e5646d2            e56476f            e56476f            e564770            e564771      5      3      0            1            1            20             1  Convolution_6?t_Convolution/expand_act_channels/input-0-CMX
//CHECK:    4     1c0         dmahw           8ffdf8c0            e5646e6            e5646e6            e5646e6            e5646eb            e5646ec      2      2      0            1            1           100             f  Convolution_6?t_Convolution/expand_act_channels/input-2-CMX
//CHECK:    5     200         dmahw           8ffdeec0            e564778            e564778            e564778            e564779            e56477a      5      3      0            0            3             8             1  Conv_0?t_Convert/input-0-CMX
//CHECK:    6     240         dmahw           8ffdf080            e564a2a            e564a2a            e564a2a            e564a2b            e564a2d      5      3      0            0            3             4             1  495/sink_port_0?t_Result

//CHECK:    Index  Offset        Engine        PLL Value          CFGID
//CHECK:    0     280           pll               4a              6
//CHECK:    1     284           pll               4a              6

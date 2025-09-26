//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env IE_NPU_LOG_FILTER=dump-statistics-of-task-ops vpux-opt  --init-compiler="vpu-arch=%arch% allow-custom-values=true" --compress-weights-btc --dump-statistics-of-task-ops -o /dev/null %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#loc0 = loc(unknown)
#loc2 = loc("profiling_result")
module @age_gender attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096] loc(#loc0)
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"} loc(#loc0)
    func.func private @builtin_Convert(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"} loc(#loc0)
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"} loc(#loc0)
  } loc(#loc0)
  config.Resources 2 of @NCE at 1.300000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 512 bytes of @CMX_NN offset 0 loc(#loc0)
      } loc(#loc0)
    } loc(#loc0)
    config.ExecutorResource 1 of @DPU  loc(#loc0)
    config.ExecutorResource 2 of @SHAVE_ACT  loc(#loc0)
    config.ExecutorResource 1 of @SHAVE_NN  loc(#loc0)
    config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64} loc(#loc0)
  } loc(#loc0)
  config.ExecutorResource 2 of @DMA_NN  loc(#loc0)
  config.MemoryResource 524288000 bytes of @DDR {config.bandwidth = 8 : i64, config.derateFactor = 6.000000e-01 : f64} loc(#loc0)
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x3x62x62xf32> loc(#loc0)
  } outputsInfo : {
    DataInfo "age_conv3" : tensor<1x48x30x30xf32> loc(#loc0)
  } profilingOutputsInfo : {
    DataInfo "0_dpu_96_actshave_128_dma" : tensor<76xui32> loc(#loc1)
  } loc(#loc0)
  func.func @main(%arg0: memref<1x3x62x62xf32, @DDR> loc(unknown), %arg1: memref<1x48x30x30xf32, @DDR> loc(unknown), %arg2: memref<76xui32> loc("profiling_result")) -> (memref<1x48x30x30xf32, @DDR>, memref<76xui32>) {
    %cst = const.Declare memref<1x1x1x784xui8> = dense<"0x0000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F000000000000000000FF00000000803F00000000070E1C00000000000000000000000000"> : tensor<1x1x1x784xui8> loc(#loc3)
    %cst_0 = const.Declare memref<48x3x3x3xf16, #NHWC> = dense<"0x35180C19F91AC21C811E921FC91CE21EFC1F529AA69CC99DC49132926896AC18E81A311A1799319B389DAC9D949EE09E9A99D5995199E6A012183E22399EB41C51202E9D46150B95FAA1DE12651E2C076020301B4A9DED9874A40198411B2A1E761646147F9EAB1D0413DDA2621D531EE11EE4104F144310829C869C919D8E1CB11D5D1ECA899780568EF29B2A9C829DF61A171C491D50981C98E498B39B1D9CB09D0D20A49EC5942B1EA79D1C1DE51ECB9D561B741F3CA2571C411BB1A3CD20861DC0A210201B1B30A0DD1E9C97F4A27D21FE1441A1E5204399F39CADA08797249BB99FA3081E8A089C5E9CCC99BD9E839831117E993294D919581235122E1D3C1D1017B41F3120FC152C203B21EF18AA16EF99D711B180649D7219EF17AD99D9157D8F799D2A1D7819369ACA16FF87869D9F1A0310F29B401CC3151C9C3F1B8B14C49BDDA0AF9FE2903CA2BEA1599FDEA1AFA165A05922FB211B232F267D25D9243F247223F8216F9E9EA0EB9F059E68A164A2D095B89D51A1381DA01DC50B96192819319A0C9313947B9D101C101C909972195719399C2593DF8EA19DE419B41A209C4C154317919DF18FFF0E899DC912119CF5157B131D99771D3116489A11189F08B89E8B1D8E93409F65204D8D019FC81CF096719E321B4199049F021F4A94D89D781BE896079DF01C9C98A69DA21E0697979D951AEA99C19C1E1DD09B2C9D711F72995D9C5B1C5C0C178F1919A692DF8F2D1DC413840D1F1A5798F695BA964618CF1CB21D9516331D1120F89C5C9DE0A00106EE188E0C0A16C51D0B1F2398E99B12A1B8989899569E28145D1A571B612106A1041CE42100A21A9D902351A0529BD320D9A1C920DE2049A3DC9A1D23DCA07999BF91639FA423760C1DA1B61B5B1FEF9C6C1B7398302047A0F79611213B9F8499A72003A07E9D4C21CB9EFB9D2A22AD9F6D9E1522339F1FA03C21489820A06821189B529FBD21489BFBA417A4BFA343203C22BF20A315B71BE20F199721191897632568269C24519E9A9C5BA02B147C1E021A9495161910960CA244A158A264937F1CF8970698331D4A8EE190021D56890894B91C308F5599F71C00813797D61CE68E8D94D417FA98AF97F51A3294FF94DD1AB09611A4C5144B1EC30D8424862433A4CC99779C8C0CB4254A2313A6BA9E7FA6E7191B238690E21ED022139526A140A134A8F3225922608E5E8DF719951E8F90821A311FD48D0C19EB1C081CAD1D391DCE1C391D911CE9186E18260FE797529DBE9F129CF29FC8A14B9BBA9FE1A1709D461132988B9D9E12F5945E9DBF1124978A990F1ECE156A96C51F8A1DBA98011E9818A00F8619C496C3131A1C20186C0F99196D94C014428EED14709C429FD09E7E983B9D3D9E3B1D0F1EA71F3385CD8FA48C059A8F9C339DCA1C591E73208317231A2D1C7597E895AB94801B569807934D2143181A98DF20611625993C91D79D9599D61F4196A09D1121E3073F9E3D9D069CF8122412A097E698FF1C1885159CF3108A0F3C8C5C8AED8EF19234800781398F56077482E794EA91BF953098F49654996E9A80149E14EA90251146826895AA0EFA900A98E81C289BB68DEB1C3F9DB19AC41DA6999D10711D779AF410FA1F4D994693AF1EC2971214A8932D99801B2B02F8995A17D8140A95041C9F9F1BA0FB9A1F15181B0A21371E442250259106FC90E29AFEA472A59EA5251E2D209B1F9825E1241C24F292F19FB4A2A495CF9F2CA2B49D3C051B1BC49D9212D41DB29DE284CE1B629DA8150F1A219E19160E1E719EC612001DF4999B0A0E983398671867192B9CB20DFD15E10D721A9D9B1E1ACE1C529D1618561CD09CEB09581C059A53183F1D609DD4141A1D8F9C3B97221CC88F008E371D0A999B95CD1CAE994B140616B696C2091A10FC99331238163D974F16A111A09A9111A992EE9C6F16BF10C09AE519CC121B9921189A91A09CB71AA7152A993820EA9B7818AB1E239E70114F1AA3A08709FB21A39DA2990320D49FE998281C9AA04E921520AE9B589B611CD59DC39B351CB49A0294121C999D68137F188E9E5017611A899C661CB018629D358C9788EF9FEC90C418CD9CD71AEF1A5E908696740DF19BBD974018EE9774161F960E10629D7014151BCE9B7418E91D959697931616E19C07034519469D50170E1D359ACA0C9C194E9B35914317679D3A174C1DB899F09BC89F4DA0B39B5A9F1DA08E99869D269E4C137690680C68198117EE18DC1A261ACA1B13153D18E51C88183F1B411EDE19B41CB91F2F060819521EDC96AA951396919AF29C30A05818811C781E181F8E1E281D189EA0A0C5A27E97860BA119FD1E9F1FCC1F4D995C9C8F9E7015B4927298FE15A194BC99FD163A8AF995C51A6B9161997818F198EC9C621B8990DD98521C9889B3938418589A049DAC1C698BF896DF1D6CA3799793212E9E681F0D9B38A39D202B27489BDEA02626F19BA6999C0D72A4651D272301A0EE9F331E92A00110CFA0B6A290227D1C149EA49F0421D2988D9E6A1CFB9D439FB521CD99A810931EDBA07BA091217299620C1385F69D531DFD1A549C0C1D1413F09DCC1C95279920CBA6CA25251F6FA3491AD99DCA98B123521DEAA1671885959010ABA22DA1EF22049F8692FA2158A5A5A0802449A77DA1CC252A959799949A249A539D3C9EF599D49DEE9E78108F1605170094329431968699B09BD29C8C1CFD1EB2204B1CCB1D2F1F331AAE1BF01C421A511AA4A0B421F91F73A390232F203BA4AEA2D2197B1D5E9F2A20131F309725218E1E3AA50410588F3BA4281E9D20F4A3331E5C22D9199B983D99191CE3960D9CB91AEC96D69A6D1C6D9BE69AED1FAD97029B6D1D109BC79C331EC7949C16E118BB9DB09C3A1F9F92370A589CBC9CBB9E2F9514958794E11B471D5F1FCF9A649C719E541335142611BC1B501DA51ECA9B239D409F94122A1205103E196B1B771D4B9A1C9D309D2D9B789D1A9C880FD38A0319EE9CFF9E259F351627115014851CD91C3B1EF698A099249CB01C251D7C1C421DD81D821E6C259B259324A1A8FDA87BA9F624C325C725D4109C15649A681A3E93CD9FD694EF1A4314509C7B988D94EA1DF41DCA1EA6A0409C5F014189F4A3A921D21CC0A1D421381E8EA2C08F9B1C8CA6D9227C22BBA438205625C3A2FA9239198EA3BC20642218A1E40F2F25459D6C9E0A142E9AD11C04106B9B8E1CA710159CFF19130CB49D671A0994CB9EB618C603139E8A1602139E9C0A175E10479D46108218719BC811641C171FF620471C751E3820311312165E192D17261AE81A90967596A799AF9CC59D05A04C93BD963B98539BA19D749F5692CF983C9EFD15770C159C4E1BA7152A9D2C188F10E39CC6193718C09AC61CC319BD9C0C1BBA18199C9118F218A099891BEC194E9C6F196119129BAD235424C7245717C715DE1D579D0F9E259AC19D339E229EA8A55FA6F0A50098979A099CF797C89B509EDE1F211D3118152455243123D095F30CD397FF9921976E95BF8EA6167891FE9CD919F495479EEA0CA994689D6F153594F59BB31ECF14AF9C6A1D4416CE9CA31DDD16451DA41E591F6F1A031D641D32A448A421A4349CE39CA99D232535253C24AD97ED9BFC9EFFA4EBA442A4F91B2C1DE91BEC1E1820A51D"> : tensor<48x3x3x3xf16, {order = #NHWC}> loc(#loc4)
    %cst_1 = const.Declare memref<48x1x1x4xsi32> = dense<"0x800F0100FFFFFF000000803F00809A3DC00F0100FFFFFF000000803F00C0863F00100100FFFFFF000000803F00C0403D40100100FFFFFF000000803F00600DBE80100100FFFFFF000000803F00C0843DC0100100FFFFFF000000803F00E04C3F00110100FFFFFF000000803F00C0973E40110100FFFFFF000000803F00402F3F80110100FFFFFF000000803F00E0C53EC0110100FFFFFF000000803F00A08EBD00120100FFFFFF000000803F0080FEBC40120100FFFFFF000000803F006048BF80120100FFFFFF000000803F00A06F3FC0120100FFFFFF000000803F006074BE00130100FFFFFF000000803F0020C6BF40130100FFFFFF000000803F00C0383D80130100FFFFFF000000803F00E02B3EC0130100FFFFFF000000803F0040BCBF00140100FFFFFF000000803F0000113E40140100FFFFFF000000803F0040A73E80140100FFFFFF000000803F0020CF3FC0140100FFFFFF000000803F0000BFBF00150100FFFFFF000000803F0020C0BC40150100FFFFFF000000803F00E07A3D80150100FFFFFF000000803F004063BFC0150100FFFFFF000000803F00E0B03F00160100FFFFFF000000803F0060753F40160100FFFFFF000000803F0020C13F80160100FFFFFF000000803F0080933FC0160100FFFFFF000000803F00E0373E00170100FFFFFF000000803F0020953D40170100FFFFFF000000803F0040AA3F80170100FFFFFF000000803F0020A6BEC0170100FFFFFF000000803F00806F3F00180100FFFFFF000000803F00E06FBE40180100FFFFFF000000803F00E0923D80180100FFFFFF000000803F0060E9BDC0180100FFFFFF000000803F00C06E3F00190100FFFFFF000000803F00C0DD3D40190100FFFFFF000000803F00C0003E80190100FFFFFF000000803F00407ABEC0190100FFFFFF000000803F00C01ABF001A0100FFFFFF000000803F00E0A33F401A0100FFFFFF000000803F00603D3E801A0100FFFFFF000000803F00E091BDC01A0100FFFFFF000000803F0040E5BB001B0100FFFFFF000000803F00C0B73F401B0100FFFFFF000000803F00C0873E"> : tensor<48x1x1x4xsi32> loc(#loc4)
    %cst_2 = const.Declare memref<1x13x62x62xf16> = dense<0.000000e+00> : tensor<1x13x62x62xf16> loc(#loc5)
    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier loc(#loc4)
    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier loc(#loc4)
    %2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier loc(#loc4)
    %3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier loc(#loc4)
    %4 = VPURT.ConfigureBarrier<4> -> !VPURT.Barrier loc(#loc4)
    %5 = VPURT.ConfigureBarrier<5> -> !VPURT.Barrier loc(#loc4)
    %6 = VPURT.ConfigureBarrier<6> -> !VPURT.Barrier loc(#loc4)
    %7 = VPURT.ConfigureBarrier<7> -> !VPURT.Barrier loc(#loc4)
    %8 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x62x62xf32, @DDR> loc(#loc3)
    %9 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x48x30x30xf32, @DDR> loc(#loc3)
    %10 = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<6xui64, @DDR> loc(#loc6)
    %11 = VPURT.DeclareBuffer <ProfilingOutput> [0] <48> -> memref<6xui64, @DDR> loc(#loc6)
    %12 = VPURT.DeclareBuffer <ProfilingOutput> [0] <96> -> memref<8xui32> loc(#loc3)
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <72576> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc7)
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <65344> -> memref<6xui64, [@CMX_NN, 0]> loc(#loc6)
    %15 = VPURT.DeclareBuffer <CMX_NN> [1] <65344> -> memref<6xui64, [@CMX_NN, 1]> loc(#loc6)
    %16 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1x3x62x62xf32, [@CMX_NN, 0]> loc(#loc4)
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <46400> -> memref<1x3x62x62xf16, [@CMX_NN, 0]> loc(#loc4)
    %18 = VPURT.DeclareBuffer <DDR> <0> -> memref<16x1984xf16, @DDR> loc(#loc8)
    %19 = VPURT.DeclareBuffer <DDR> <3968> -> memref<16x1860xf16, @DDR> loc(#loc9)
    %20 = VPURT.DeclareBuffer <CMX_NN> <256> -> !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> loc(#loc4)
    %21 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1984x16xf16, [@CMX_NN, 0]> loc(#loc8)
    %22 = VPURT.DeclareBuffer <CMX_NN> [1] <256> -> memref<1860x16xf16, [@CMX_NN, 1]> loc(#loc9)
    %23 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1x16x32x62xf16, #NHWC, [@CMX_NN, 0]> loc(#loc10)
    %24 = VPURT.DeclareBuffer <CMX_NN> [1] <256> -> memref<1x16x30x62xf16, #NHWC, [@CMX_NN, 1]> loc(#loc11)
    %25 = VPURT.DeclareBuffer <CMX_NN> [0] <63744> -> memref<48x1x1x4xsi32, [@CMX_NN, 0]> loc(#loc12)
    %26 = VPURT.DeclareBuffer <CMX_NN> [1] <63744> -> memref<48x1x1x4xsi32, [@CMX_NN, 1]> loc(#loc13)
    %27 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <63744> -> !VPUIP.DistributedBuffer<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc4)
    %28 = VPURT.DeclareBuffer <CMX_NN> <72640> -> !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> loc(#loc4)
    %29 = VPURT.DeclareBuffer <CMX_NN> [0] <72640> -> memref<1x48x30x60xf16, #NHWC, [@CMX_NN, 0]> loc(#loc14)
    %30 = VPURT.DeclareBuffer <CMX_NN> [1] <72640> -> memref<1x48x30x60xf16, #NHWC, [@CMX_NN, 1]> loc(#loc15)
    %31 = VPURT.DeclareBuffer <CMX_NN> [0] <72640> -> memref<1x48x30x60xf16, #NHWC, [@CMX_NN, 0]> loc(#loc16)
    %32 = VPURT.DeclareBuffer <CMX_NN> [1] <72640> -> memref<1x48x30x60xf16, #NHWC, [@CMX_NN, 1]> loc(#loc17)
    %33 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <69504> -> !VPUIP.DistributedBuffer<48x3x3x3xf16, {order = #NHWC, strides = [32, 1, 9, 3]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc4)
    %34 = VPURT.DeclareBuffer <CMX_NN> <256> -> !VPUIP.DistributedBuffer<1x48x30x30xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> loc(#loc18)
    %35 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1x48x15x30xf16, [@CMX_NN, 0]> loc(#loc18)
    %36 = VPURT.DeclareBuffer <CMX_NN> [1] <256> -> memref<1x48x15x30xf16, [@CMX_NN, 1]> loc(#loc18)
    %37 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1x48x15x30xf16, [@CMX_NN, 0]> loc(#loc19)
    %38 = VPURT.DeclareBuffer <CMX_NN> [1] <256> -> memref<1x48x15x30xf16, [@CMX_NN, 1]> loc(#loc20)
    %39 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <64512> -> !VPUIP.DistributedBuffer<1x1x1x784xui8, {order = #NCHW, strides = [784, 784, 784, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc21)
    %40 = VPURT.DeclareBuffer <CMX_NN> [0] <72640> -> memref<1x48x30x30xf16, [@CMX_NN, 0]> loc(#loc22)
    %41 = VPURT.DeclareBuffer <CMX_NN> [0] <72640> -> memref<1x48x15x30xf16, {order = #NCHW, strides = [43200, 900, 30, 1]}, [@CMX_NN, 0]> loc(#loc18)
    %42 = VPURT.DeclareBuffer <CMX_NN> [0] <73540> -> memref<1x48x15x30xf16, {order = #NCHW, strides = [43200, 900, 30, 1]}, [@CMX_NN, 0]> loc(#loc18)
    %43 = VPURT.DeclareBuffer <CMX_NN> [0] <159040> -> memref<1x48x30x30xf32, [@CMX_NN, 0]> loc(#loc22)
    %44 = VPURT.DeclareBuffer <DDR> <23064> -> memref<1x13x62x62xf16, {order = #NCHW, strides = [61504, 3844, 62, 1]}, @DDR> loc(#loc23)
    %45 = VPURT.DeclareBuffer <CMX_NN> [0] <72576> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc24)
    %46 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x62x62xf16, {order = #NCHW, strides = [61504, 3844, 62, 1]}, @DDR> loc(#loc4)
    %47 = VPURT.DeclareBuffer <CMX_NN> [0] <69504> -> memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 0]> loc(#loc25)
    %48 = VPURT.DeclareBuffer <CMX_NN> [1] <69504> -> memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 1]> loc(#loc26)
    %49 = VPURT.DeclareBuffer <CMX_NN> [0] <65344> -> memref<2xui64, [@CMX_NN, 0]> loc(#loc27)
    %50 = VPURT.DeclareBuffer <CMX_NN> [1] <65344> -> memref<2xui64, [@CMX_NN, 1]> loc(#loc28)
    %51 = VPURT.DeclareBuffer <CMX_NN> [0] <64512> -> memref<48x1x1x4xsi32, [@CMX_NN, 0]> loc(#loc29)
    %52 = VPURT.DeclareBuffer <CMX_NN> [1] <64512> -> memref<48x1x1x4xsi32, [@CMX_NN, 1]> loc(#loc30)
    %55 = VPURT.DeclareBuffer <CMX_NN> [0] <65360> -> memref<4xui64, [@CMX_NN, 0]> loc(#loc33)
    %56 = VPURT.DeclareBuffer <CMX_NN> [1] <65360> -> memref<4xui64, [@CMX_NN, 1]> loc(#loc34)
    %57 = VPURT.DeclareBuffer <CMX_NN> [0] <72592> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc35)
    %58 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc36)
    %59 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc36)
    %60 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc37)
    %61 = VPURT.DeclareBuffer <CMX_NN> [0] <8> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc37)
    %62 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc38)
    %63 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc38)
    %64 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc39)
    %65 = VPURT.DeclareBuffer <CMX_NN> [0] <136> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc39)
    %66 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc40)
    %67 = VPURT.DeclareBuffer <CMX_NN> [0] <144> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc40)
    %68 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc41)
    %69 = VPURT.DeclareBuffer <CMX_NN> [0] <152> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc41)
    %70 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc36)
    %71 = VPURT.DeclareBuffer <CMX_NN> [0] <16> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc36)
    %72 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc42)
    %73 = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc42)
    %74 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc43)
    %75 = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc43)
    %76 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc44)
    %77 = VPURT.DeclareBuffer <CMX_NN> [0] <40> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc44)
    %78 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc45)
    %79 = VPURT.DeclareBuffer <CMX_NN> [0] <160> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc45)
    %80 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc46)
    %81 = VPURT.DeclareBuffer <CMX_NN> [0] <168> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc46)
    %82 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc38)
    %83 = VPURT.DeclareBuffer <CMX_NN> [0] <48> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc38)
    %84 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc47)
    %85 = VPURT.DeclareBuffer <CMX_NN> [0] <56> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc47)
    %86 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc48)
    %87 = VPURT.DeclareBuffer <CMX_NN> [0] <176> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc48)
    %88 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc49)
    %89 = VPURT.DeclareBuffer <CMX_NN> [0] <184> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc49)
    %90 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc50)
    %91 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc50)
    %92 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc51)
    %93 = VPURT.DeclareBuffer <CMX_NN> [0] <72> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc51)
    %94 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<10xui64, [@CMX_NN, 0]> loc(#loc52)
    %95 = VPURT.DeclareBuffer <ProfilingOutput> [0] <128> -> memref<10xui64> loc(#loc52)
    %96 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc53)
    %97 = VPURT.DeclareBuffer <CMX_NN> [0] <192> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc53)
    %98 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc54)
    %99 = VPURT.DeclareBuffer <CMX_NN> [0] <200> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc54)
    %100 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc55)
    %101 = VPURT.DeclareBuffer <CMX_NN> [0] <208> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc55)
    %102 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc56)
    %103 = VPURT.DeclareBuffer <CMX_NN> [0] <216> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc56)
    %104 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<12xui64, [@CMX_NN, 0]> loc(#loc57)
    %105 = VPURT.DeclareBuffer <ProfilingOutput> [0] <208> -> memref<12xui64> loc(#loc57)
    %106 = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<12xui64> loc(#loc58)
    %107 = VPURT.DeclareBuffer <ProfilingOutput> [0] <96> -> memref<8xui32> loc(#loc58)
    %108 = VPURT.DeclareBuffer <ProfilingOutput> [0] <128> -> memref<22xui64> loc(#loc58)
    VPURT.Task {
      %109 = VPUIP.NNDMA {profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%58 : memref<1xui64, @Register>) outputs(%59 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc36)
    } loc(#loc36)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.NNDMA inputs(%8 : memref<1x3x62x62xf32, @DDR>) outputs(%16 : memref<1x3x62x62xf32, [@CMX_NN, 0]>) -> memref<1x3x62x62xf32, [@CMX_NN, 0]> loc(#loc4)
    } loc(#loc4)
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %109 = VPUIP.NNDMA {profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%60 : memref<1xui64, @Register>) outputs(%61 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc37)
    } loc(#loc37)
    VPURT.Task {
      %109 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%62 : memref<1xui64, @Register>) outputs(%63 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc38)
    } loc(#loc38)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.NNDMA {port = 1 : i64} inputs(%cst_0 : memref<48x3x3x3xf16, #NHWC>) outputs(%33 : !VPUIP.DistributedBuffer<48x3x3x3xf16, {order = #NHWC, strides = [32, 1, 9, 3]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<48x3x3x3xf16, {order = #NHWC, strides = [32, 1, 9, 3]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc59)
    } loc(#loc59)
    VPURT.Task {
      %109 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%64 : memref<1xui64, @Register>) outputs(%65 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc39)
    } loc(#loc39)
    VPURT.Task {
      %109 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%66 : memref<1xui64, @Register>) outputs(%67 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc40)
    } loc(#loc40)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.NNDMA {port = 1 : i64} inputs(%cst_2 : memref<1x13x62x62xf16>) outputs(%44 : memref<1x13x62x62xf16, {order = #NCHW, strides = [61504, 3844, 62, 1]}, @DDR>) -> memref<1x13x62x62xf16, {order = #NCHW, strides = [61504, 3844, 62, 1]}, @DDR> loc(#loc60)
    } loc(#loc60)
    VPURT.Task updates(%1 : !VPURT.Barrier) {
      %109 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%68 : memref<1xui64, @Register>) outputs(%69 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc41)
    } loc(#loc41)
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Convert inputs(%16 as %arg3: memref<1x3x62x62xf32, [@CMX_NN, 0]>) outputs(%17 as %arg4: memref<1x3x62x62xf16, [@CMX_NN, 0]>) profiling_data(%45 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x3x62x62xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x3x62x62xf32, [@CMX_NN, 0]>, memref<1x3x62x62xf16, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc61)
    } loc(#loc61)
    VPURT.Task waits(%2 : !VPURT.Barrier) {
      %109 = VPUIP.NNDMA {profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%70 : memref<1xui64, @Register>) outputs(%71 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc36)
    } loc(#loc36)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.NNDMA inputs(%17 : memref<1x3x62x62xf16, [@CMX_NN, 0]>) outputs(%46 : memref<1x3x62x62xf16, {order = #NCHW, strides = [61504, 3844, 62, 1]}, @DDR>) -> memref<1x3x62x62xf16, {order = #NCHW, strides = [61504, 3844, 62, 1]}, @DDR> loc(#loc4)
    } loc(#loc4)
    VPURT.Task updates(%1 : !VPURT.Barrier) {
      %109 = VPUIP.NNDMA {profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%72 : memref<1xui64, @Register>) outputs(%73 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc42)
    } loc(#loc42)
    VPURT.Task waits(%1 : !VPURT.Barrier) {
      %109 = VPUIP.NNDMA {profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%74 : memref<1xui64, @Register>) outputs(%75 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc43)
    } loc(#loc43)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 3968 : i64, srcWidth = 3968 : i64, srcStride = 2 : i64, srcPlaneStride = 7688 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>} inputs(%18 : memref<16x1984xf16, @DDR>) outputs(%21 : memref<1984x16xf16, [@CMX_NN, 0]>) -> memref<1984x16xf16, [@CMX_NN, 0]> loc(#loc62)
    } loc(#loc62)
    VPURT.Task {
      %109 = VPUIP.NNDMA {profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%76 : memref<1xui64, @Register>) outputs(%77 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc44)
    } loc(#loc44)
    VPURT.Task waits(%1 : !VPURT.Barrier) {
      %109 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%78 : memref<1xui64, @Register>) outputs(%79 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc45)
    } loc(#loc45)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 3720 : i64, srcWidth = 3720 : i64, srcStride = 2 : i64, srcPlaneStride = 7688 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 1 : i64} inputs(%19 : memref<16x1860xf16, @DDR>) outputs(%22 : memref<1860x16xf16, [@CMX_NN, 1]>) -> memref<1860x16xf16, [@CMX_NN, 1]> loc(#loc63)
    } loc(#loc63)
    VPURT.Task updates(%3 : !VPURT.Barrier) {
      %109 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%80 : memref<1xui64, @Register>) outputs(%81 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc46)
    } loc(#loc46)
    VPURT.Task {
      %109 = VPUIP.NNDMA {profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%82 : memref<1xui64, @Register>) outputs(%83 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc38)
    } loc(#loc38)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.NNDMA inputs(%cst_1 : memref<48x1x1x4xsi32>) outputs(%27 : !VPUIP.DistributedBuffer<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc59)
    } loc(#loc59)
    VPURT.Task updates(%3 : !VPURT.Barrier) {
      %109 = VPUIP.NNDMA {profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%84 : memref<1xui64, @Register>) outputs(%85 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc47)
    } loc(#loc47)
    VPURT.Task {
      %109 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%86 : memref<1xui64, @Register>) outputs(%87 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc48)
    } loc(#loc48)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.NNDMA {port = 1 : i64} inputs(%cst : memref<1x1x1x784xui8>) outputs(%39 : !VPUIP.DistributedBuffer<1x1x1x784xui8, {order = #NCHW, strides = [784, 784, 784, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x784xui8, {order = #NCHW, strides = [784, 784, 784, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc64)
    } loc(#loc64)
    VPURT.Task updates(%4 : !VPURT.Barrier) {
      %109 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%88 : memref<1xui64, @Register>) outputs(%89 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc49)
    } loc(#loc49)
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %109:2 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>} input(%23 : memref<1x16x32x62xf16, #NHWC, [@CMX_NN, 0]>) weights(%47 : memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%25 : memref<48x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%20 : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>) parent_output(%28 : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%31 : memref<1x48x30x60xf16, #NHWC, [@CMX_NN, 0]>) profiling_data(%49 : memref<2xui64, [@CMX_NN, 0]>) -> memref<1x48x30x60xf16, #NHWC, [@CMX_NN, 0]>, memref<2xui64, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [59, 29, 47], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>} loc(#loc4)
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>} loc(#loc4)
      } loc(#loc65)
    } loc(#loc65)
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %109:2 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>} input(%24 : memref<1x16x30x62xf16, #NHWC, [@CMX_NN, 1]>) weights(%48 : memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%26 : memref<48x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%20 : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>) parent_output(%28 : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%32 : memref<1x48x30x60xf16, #NHWC, [@CMX_NN, 1]>) profiling_data(%50 : memref<2xui64, [@CMX_NN, 1]>) -> memref<1x48x30x60xf16, #NHWC, [@CMX_NN, 1]>, memref<2xui64, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [59, 59, 47], outStart = [0, 30, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>} loc(#loc4)
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>} loc(#loc4)
      } loc(#loc66)
    } loc(#loc66)
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %109:2 = VPUIP.NCEClusterTask {is_segmented, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%29 : memref<1x48x30x60xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%51 : memref<48x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%28 : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%34 : !VPUIP.DistributedBuffer<1x48x30x30xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%37 : memref<1x48x15x30xf16, [@CMX_NN, 0]>) profiling_data(%55 : memref<4xui64, [@CMX_NN, 0]>) -> memref<1x48x15x30xf16, [@CMX_NN, 0]>, memref<4xui64, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [29, 14, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>} loc(#loc18)
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [29, 14, 47], outStart = [0, 0, 32], pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>} loc(#loc18)
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>} loc(#loc18)
      } loc(#loc67)
    } loc(#loc67)
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %109:2 = VPUIP.NCEClusterTask {is_segmented, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%30 : memref<1x48x30x60xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%52 : memref<48x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%28 : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%34 : !VPUIP.DistributedBuffer<1x48x30x30xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%38 : memref<1x48x15x30xf16, [@CMX_NN, 1]>) profiling_data(%56 : memref<4xui64, [@CMX_NN, 1]>) -> memref<1x48x15x30xf16, [@CMX_NN, 1]>, memref<4xui64, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [29, 29, 31], outStart = [0, 15, 0], pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>} loc(#loc18)
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [29, 29, 47], outStart = [0, 15, 32], pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>} loc(#loc18)
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>} loc(#loc18)
      } loc(#loc68)
    } loc(#loc68)
    VPURT.Task waits(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.NNDMA inputs(%14 : memref<6xui64, [@CMX_NN, 0]>) outputs(%10 : memref<6xui64, @DDR>) -> memref<6xui64, @DDR> loc(#loc69)
    } loc(#loc69)
    VPURT.Task waits(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.NNDMA {port = 1 : i64} inputs(%15 : memref<6xui64, [@CMX_NN, 1]>) outputs(%11 : memref<6xui64, @DDR>) -> memref<6xui64, @DDR> loc(#loc70)
    } loc(#loc70)
    VPURT.Task {
      %109 = VPUIP.NNDMA {profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%90 : memref<1xui64, @Register>) outputs(%91 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc50)
    } loc(#loc50)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.NNDMA inputs(%35 : memref<1x48x15x30xf16, [@CMX_NN, 0]>) outputs(%41 : memref<1x48x15x30xf16, {order = #NCHW, strides = [43200, 900, 30, 1]}, [@CMX_NN, 0]>) -> memref<1x48x15x30xf16, {order = #NCHW, strides = [43200, 900, 30, 1]}, [@CMX_NN, 0]> loc(#loc71)
    } loc(#loc71)
    VPURT.Task {
      %109 = VPUIP.NNDMA {profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%92 : memref<1xui64, @Register>) outputs(%93 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc51)
    } loc(#loc51)
    VPURT.Task updates(%6 : !VPURT.Barrier) {
      %109 = VPUIP.NNDMA inputs(%94 : memref<10xui64, [@CMX_NN, 0]>) outputs(%95 : memref<10xui64>) -> memref<10xui64> loc(#loc52)
    } loc(#loc52)
    VPURT.Task {
      %109 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%96 : memref<1xui64, @Register>) outputs(%97 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc53)
    } loc(#loc53)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.NNDMA {port = 1 : i64} inputs(%36 : memref<1x48x15x30xf16, [@CMX_NN, 1]>) outputs(%42 : memref<1x48x15x30xf16, {order = #NCHW, strides = [43200, 900, 30, 1]}, [@CMX_NN, 0]>) -> memref<1x48x15x30xf16, {order = #NCHW, strides = [43200, 900, 30, 1]}, [@CMX_NN, 0]> loc(#loc72)
    } loc(#loc72)
    VPURT.Task updates(%6 : !VPURT.Barrier) {
      %109 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%98 : memref<1xui64, @Register>) outputs(%99 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc54)
    } loc(#loc54)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Convert inputs(%40 as %arg3: memref<1x48x30x30xf16, [@CMX_NN, 0]>) outputs(%43 as %arg4: memref<1x48x30x30xf32, [@CMX_NN, 0]>) profiling_data(%57 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x48x30x30xf32, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x48x30x30xf16, [@CMX_NN, 0]>, memref<1x48x30x30xf32, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc73)
    } loc(#loc73)
    VPURT.Task waits(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.NNDMA inputs(%13 : memref<8xui32, [@CMX_NN, 0]>) outputs(%12 : memref<8xui32>) -> memref<8xui32> loc(#loc74)
    } loc(#loc74)
    VPURT.Task waits(%7 : !VPURT.Barrier) {
      %109 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%100 : memref<1xui64, @Register>) outputs(%101 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc55)
    } loc(#loc55)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %109 = VPUIP.NNDMA {port = 1 : i64} inputs(%43 : memref<1x48x30x30xf32, [@CMX_NN, 0]>) outputs(%9 : memref<1x48x30x30xf32, @DDR>) -> memref<1x48x30x30xf32, @DDR> loc(#loc22)
    } loc(#loc22)
    VPURT.Task {
      %109 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%102 : memref<1xui64, @Register>) outputs(%103 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc56)
    } loc(#loc56)
    VPURT.Task {
      %109 = VPUIP.NNDMA {port = 1 : i64} inputs(%104 : memref<12xui64, [@CMX_NN, 0]>) outputs(%105 : memref<12xui64>) -> memref<12xui64> loc(#loc57)
    } loc(#loc57)
    return %arg1, %arg2 : memref<1x48x30x30xf32, @DDR>, memref<76xui32> loc(#loc22)
  } loc(#loc0)
} loc(#loc0)
#loc1 = loc("combinedProfilingDataOutputInfo")
#loc3 = loc(fused["pool1", "_fused_constant"])
#loc4 = loc("conv1/WithoutBiases")
#loc5 = loc(fused["conv1/WithoutBiases", "_constant_permute_1_13"])
#loc6 = loc("dpuProfilingCMX2DDR0")
#loc7 = loc("1_actProfilingSubviewBuffer_0")
#loc8 = loc(fused["conv1/WithoutBiases", "_cluster_0"])
#loc9 = loc(fused["conv1/WithoutBiases", "_cluster_1"])
#loc10 = loc(fused["conv1/WithoutBiases", "_PROF_0_0_2_1-1,1,", "_input_cluster_0"])
#loc11 = loc(fused["conv1/WithoutBiases", "_PROF_0_0_2_1-1,1,", "_input_cluster_1"])
#loc12 = loc(fused["conv1/WithoutBiases", "_PROF_0_0_2_1-1,1,", "_weightTable_cluster_0"])
#loc13 = loc(fused["conv1/WithoutBiases", "_PROF_0_0_2_1-1,1,", "_weightTable_cluster_1"])
#loc14 = loc(fused["pool1", "_PROF_1_0_2_2-2,2,", "_input_cluster_0"])
#loc15 = loc(fused["pool1", "_PROF_1_0_2_2-2,2,", "_input_cluster_1"])
#loc16 = loc(fused["conv1/WithoutBiases", "_PROF_0_0_2_1-1,1,", "_outputBuff_cluster_0"])
#loc17 = loc(fused["conv1/WithoutBiases", "_PROF_0_0_2_1-1,1,", "_outputBuff_cluster_1"])
#loc18 = loc("pool1")
#loc19 = loc(fused["pool1", "_PROF_1_0_2_2-2,2,", "_outputBuff_cluster_0"])
#loc20 = loc(fused["pool1", "_PROF_1_0_2_2-2,2,", "_outputBuff_cluster_1"])
#loc21 = loc(fused["pool1", "_fused_constant", "_fused_tile"])
#loc22 = loc("output")
#loc23 = loc(fused["conv1/WithoutBiases", "_expand_subview_1_13"])
#loc24 = loc(fused["1_actProfilingSubviewBuffer_0", "_actshaveProfilingSubview_0"])
#loc25 = loc(fused["conv1/WithoutBiases", "_PROF_0_0_2_1-1,1,", "_weights_cluster_0"])
#loc26 = loc(fused["conv1/WithoutBiases", "_PROF_0_0_2_1-1,1,", "_weights_cluster_1"])
#loc27 = loc(fused["conv1/WithoutBiases", "_PROF_0_0_2_1-1,1,", "_profilingBuff_cluster_0"])
#loc28 = loc(fused["conv1/WithoutBiases", "_PROF_0_0_2_1-1,1,", "_profilingBuff_cluster_1"])
#loc29 = loc(fused["pool1", "_PROF_1_0_2_2-2,2,", "_weightTable_cluster_0"])
#loc30 = loc(fused["pool1", "_PROF_1_0_2_2-2,2,", "_weightTable_cluster_1"])
#loc33 = loc(fused["pool1", "_PROF_1_0_2_2-2,2,", "_profilingBuff_cluster_0"])
#loc34 = loc(fused["pool1", "_PROF_1_0_2_2-2,2,", "_profilingBuff_cluster_1"])
#loc35 = loc(fused["1_actProfilingSubviewBuffer_0", "_actshaveProfilingSubview_4"])
#loc36 = loc("conv1/WithoutBiases")
#loc37 = loc("conv1/WithoutBiases")
#loc38 = loc("conv1/WithoutBiases/_broadcast_copy_to_CMX[0,1]")
#loc39 = loc("conv1/WithoutBiases/_broadcast_copy_to_CMX[0,1]")
#loc40 = loc("conv1/WithoutBiases/_expand_copy_1_13")
#loc41 = loc("conv1/WithoutBiases/_expand_copy_1_13")
#loc42 = loc("conv1/WithoutBiases")
#loc43 = loc("conv1/WithoutBiases/_cluster_0/_unrolled_permuteDMA")
#loc44 = loc("conv1/WithoutBiases/_cluster_0/_unrolled_permuteDMA")
#loc45 = loc("conv1/WithoutBiases/_cluster_1/_unrolled_permuteDMA")
#loc46 = loc("conv1/WithoutBiases/_cluster_1/_unrolled_permuteDMA")
#loc47 = loc("conv1/WithoutBiases/_broadcast_copy_to_CMX[0,1]")
#loc48 = loc("pool1/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]")
#loc49 = loc("pool1/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]")
#loc50 = loc("pool1/_cluster_0")
#loc51 = loc("pool1/_cluster_0")
#loc52 = loc("dmaProfilingCMX2DDR0")
#loc53 = loc("pool1/_cluster_1")
#loc54 = loc("pool1/_cluster_1")
#loc55 = loc("output")
#loc56 = loc("output")
#loc57 = loc("dmaProfilingCMX2DDR80")
#loc58 = loc("newProfilingBuffer")
#loc59 = loc(fused["conv1/WithoutBiases", "_broadcast_copy_to_CMX[0,1]"])
#loc60 = loc(fused["conv1/WithoutBiases", "_expand_copy_1_13"])
#loc61 = loc(fused["conv1/WithoutBiases", "_PROF_0_2_0_0"])
#loc62 = loc(fused["conv1/WithoutBiases", "_cluster_0", "_unrolled_permuteDMA"])
#loc63 = loc(fused["conv1/WithoutBiases", "_cluster_1", "_unrolled_permuteDMA"])
#loc64 = loc(fused["pool1", "_fused_constant", "_fused_tile", "_broadcast_copy_to_CMX[0,1]"])
#loc65 = loc(fused["conv1/WithoutBiases", "_PROF_0_0_2_1-1,1,", "_cluster_0"])
#loc66 = loc(fused["conv1/WithoutBiases", "_PROF_0_0_2_1-1,1,", "_cluster_1"])
#loc67 = loc(fused["pool1", "_PROF_1_0_2_2-2,2,", "_cluster_0"])
#loc68 = loc(fused["pool1", "_PROF_1_0_2_2-2,2,", "_cluster_1"])
#loc69 = loc(fused["dpuProfilingCMX2DDR0", "_cluster_0"])
#loc70 = loc(fused["dpuProfilingCMX2DDR0", "_cluster_1"])
#loc71 = loc(fused["pool1", "_cluster_0"])
#loc72 = loc(fused["pool1", "_cluster_1"])
#loc73 = loc(fused["output", "_PROF_0_2_1_0"])
#loc74 = loc("actshaveProfilingCMX2DDR0")

// CHECK:  Input size - 45.04 KB Output size - 169.04 KB
// CHECK:  DDR heap size - 120.12 KB
// CHECK:  VPUIP tasks statistics:
// CHECK:  VPUIP Tasks - 44 ops
// CHECK:    VPUIP.NNDMA - 36 ops : Size - 422.81 KB
// CHECK:      CMX2CMX - 2 ops : Size - 84.37 KB
// CHECK:      CMX2DDR - 7 ops : Size - 191.57 KB
// CHECK:        Profiling buffer management - 5 ops
// CHECK:          DMA - 2 ops
// CHECK:          DPU - 2 ops
// CHECK:          ActShave - 1 ops
// CHECK:      DDR2CMX - 4 ops : Size - 49.09 KB
// CHECK:      DDR2DDR - 1 ops : Size - 97.60 KB
// CHECK:      REG2CMX - 22 ops : Size - 176 bytes
// CHECK:        Profiling Timestamp DMA - 22 ops
// CHECK:    VPUIP.PermuteDMA - 2 ops
// CHECK:      DDR2CMX - 2 ops : Size - 120.12 KB
// CHECK:    VPUIP.SW.Kernel - 2 ops
// CHECK:      builtin_Convert - 2 ops
// CHECK:    NCETask Operations - 4 ops
// CHECK:      Dense - 4 ops
// CHECK:    Barrier statistics:
// CHECK:      VPURT.ConfigureBarrierOp - 8 ops
// CHECK:  Weights statistics
// CHECK:  Total weights - count: 4, size: 101.64 KB (no compression)
// CHECK:  Const swizzling statistics:
// CHECK:    Swizzled constants     - count: 0, size: 0 bytes
// CHECK:    Not swizzled constants - count: 4, size: 101.64 KB

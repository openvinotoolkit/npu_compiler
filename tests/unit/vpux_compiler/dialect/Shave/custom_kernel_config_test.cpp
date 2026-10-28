//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// =============================================================================
// Unit tests for the custom kernel config utilities:
//   - CustomLayerConfigParser  (XML -> CustomLayerConfigMap)
//   - CustomLayerConfigFactory (CustomLayerConfigMap -> Shave IR)
//   - config_ir_utils          (Shave IR -> CustomLayerConfig structs)
//
// Organized into three fixture classes:
//   ConfigParserTest  – pure C++ / XML file I/O, no MLIR context.
//   ConfigIrUtilsTest – builds Shave ops with OpBuilder, calls parse* helpers.
//   ConfigFactoryTest – builds a config map, runs factory, inspects emitted IR.
//
// The ConfigFactoryTest::RoundTrip case exercises the full stack:
//   config map -> factory -> IR -> ir_utils -> recovered struct == original.
// =============================================================================

#include "common/utils.hpp"
#include "vpux/compiler/dialect/Shave/IR/dialect.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/Shave/custom_layer_config.hpp"
#include "vpux/compiler/dialect/Shave/utils/config_factory.hpp"
#include "vpux/compiler/dialect/Shave/utils/config_ir_utils.hpp"
#include "vpux/compiler/dialect/Shave/utils/config_parser.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <gtest/gtest.h>
#include <llvm/Support/FileSystem.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include <fstream>
#include <string>

using namespace vpux;
using namespace vpux::Shave;

// =============================================================================
// Helpers
// =============================================================================

namespace {

/// Writes XML content to a temp file and returns the path.
std::string writeTempXml(const std::string& content) {
    llvm::SmallString<128> path;
    const auto err = llvm::sys::fs::createTemporaryFile("custom_kernel_config", "xml", path);
    VPUX_THROW_UNLESS(!err, "Failed to create temp file: {0}", err.message());
    std::ofstream f(path.str().str());
    f << content;
    return path.str().str();
}

/// RAII wrapper that deletes the temp file on destruction.
struct TempFile {
    std::string path;
    explicit TempFile(std::string p): path(std::move(p)) {
    }
    ~TempFile() {
        llvm::sys::fs::remove(path);
    }
};

}  // namespace

// =============================================================================
// ConfigParserTest  – pure C++ / XML-driven, no MLIR
// =============================================================================

struct ConfigParserTest : testing::Test {};

TEST_F(ConfigParserTest, P1_MissingFile) {
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser("/nonexistent/path/config.xml");
    EXPECT_THROW(parser.loadConfigs(configs), std::exception);
}

TEST_F(ConfigParserTest, P2_MalformedXml) {
    TempFile tmp{writeTempXml("<<< NOT XML >>>")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    EXPECT_THROW(parser.loadConfigs(configs), std::exception);
}

TEST_F(ConfigParserTest, P3_WrongRootElement) {
    TempFile tmp{writeTempXml(R"(<WrongRoot target="NPU"><CustomLayer name="x"/></WrongRoot>)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    EXPECT_THROW(parser.loadConfigs(configs), std::exception);
}

TEST_F(ConfigParserTest, P4_MissingTargetAttr) {
    TempFile tmp{writeTempXml(R"(<CustomLayerCollection><CustomLayer name="x"/></CustomLayerCollection>)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    EXPECT_THROW(parser.loadConfigs(configs), std::exception);
}

TEST_F(ConfigParserTest, P5_WrongTarget) {
    TempFile tmp{
            writeTempXml(R"(<CustomLayerCollection target="CPU"><CustomLayer name="x"/></CustomLayerCollection>)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    EXPECT_THROW(parser.loadConfigs(configs), std::exception);
}

TEST_F(ConfigParserTest, P6_MinimalValid) {
    TempFile tmp{writeTempXml(R"(
<CustomLayerCollection target="NPU">
  <CustomLayer name="myKernel">
    <Kernel type="PREBUILT">
      <Source path="/path/to/kernel.elf"/>
    </Kernel>
  </CustomLayer>
</CustomLayerCollection>
)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    ASSERT_NO_THROW(parser.loadConfigs(configs));

    ASSERT_EQ(configs.size(), 1u);
    ASSERT_TRUE(configs.count("myKernel"));
    const auto& cfg = *configs.at("myKernel");
    EXPECT_EQ(cfg.getUniqueId(), "myKernel");
    EXPECT_EQ(cfg.kernelConfig.sourcePath, "/path/to/kernel.elf");
    EXPECT_EQ(cfg.kernelConfig.kernelType, KernelType::PREBUILT);
    EXPECT_FALSE(cfg.kernelConfig.usesDMA);
    EXPECT_FALSE(cfg.kernelConfig.usesDPU);
}

TEST_F(ConfigParserTest, P7_UsesDmaFlag) {
    TempFile tmp{writeTempXml(R"(
<CustomLayerCollection target="NPU">
  <CustomLayer name="dmaKernel">
    <Kernel type="PREBUILT">
      <Source path="/path/to/kernel.elf"/>
      <CompilerOptions uses-dma="true"/>
    </Kernel>
  </CustomLayer>
</CustomLayerCollection>
)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    ASSERT_NO_THROW(parser.loadConfigs(configs));
    ASSERT_TRUE(configs.count("dmaKernel"));
    const auto& cfg = *configs.at("dmaKernel");
    EXPECT_TRUE(cfg.kernelConfig.usesDMA);
    EXPECT_FALSE(cfg.kernelConfig.usesDPU);
}

TEST_F(ConfigParserTest, P8_UsesDpuFlag) {
    TempFile tmp{writeTempXml(R"(
<CustomLayerCollection target="NPU">
  <CustomLayer name="dpuKernel">
    <Kernel type="PREBUILT">
      <Source path="/path/to/kernel.elf"/>
      <CompilerOptions uses-dpu="true"/>
    </Kernel>
  </CustomLayer>
</CustomLayerCollection>
)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    ASSERT_NO_THROW(parser.loadConfigs(configs));
    const auto& cfg = *configs.at("dpuKernel");
    EXPECT_FALSE(cfg.kernelConfig.usesDMA);
    EXPECT_TRUE(cfg.kernelConfig.usesDPU);
}

TEST_F(ConfigParserTest, P9_IOConfig_ScratchOnly) {
    TempFile tmp{writeTempXml(R"(
<CustomLayerCollection target="NPU">
  <CustomLayer name="scratchKernel">
    <Kernel type="PREBUILT">
      <Source path="/path/to/kernel.elf"/>
    </Kernel>
    <Buffers>
      <Scratch arg-index="2" size="256" location="CMX"/>
    </Buffers>
  </CustomLayer>
</CustomLayerCollection>
)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    ASSERT_NO_THROW(parser.loadConfigs(configs));
    const auto& io = configs.at("scratchKernel")->ioConfig;
    EXPECT_TRUE(io.ioTensorConfigs.empty());
    ASSERT_EQ(io.scratchConfigs.size(), 1u);
    EXPECT_EQ(io.scratchConfigs[0].argIndex, 2u);
    EXPECT_EQ(io.scratchConfigs[0].size, 256u);
    EXPECT_EQ(io.scratchConfigs[0].location, ScratchLocation::CMX);
}

TEST_F(ConfigParserTest, P10_IOConfig_TensorEnforcedAll) {
    TempFile tmp{writeTempXml(R"(
<CustomLayerCollection target="NPU">
  <CustomLayer name="fullKernel">
    <Kernel type="PREBUILT">
      <Source path="/kernel.elf"/>
    </Kernel>
    <Buffers>
      <Tensor arg-index="0" type="Input" enforce-rank="4D" enforce-data-type="FP16" enforce-layout="NHWC"/>
      <Tensor arg-index="0" type="Output" enforce-rank="4D" enforce-data-type="FP32" enforce-layout="NCHW"/>
    </Buffers>
  </CustomLayer>
</CustomLayerCollection>
)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    ASSERT_NO_THROW(parser.loadConfigs(configs));
    const auto& io = configs.at("fullKernel")->ioConfig;
    ASSERT_EQ(io.ioTensorConfigs.size(), 2u);

    const auto& in = io.ioTensorConfigs[0];
    EXPECT_EQ(in.argIndex, 0u);
    EXPECT_EQ(in.ioType, IOType::Input);
    ASSERT_TRUE(in.enforcedRank.has_value());
    EXPECT_EQ(in.enforcedRank.value(), TensorRank::_4D);
    ASSERT_TRUE(in.enforcedDataType.has_value());
    EXPECT_EQ(in.enforcedDataType.value(), TensorDataType::FP16);
    ASSERT_TRUE(in.enforcedLayout.has_value());
    EXPECT_EQ(in.enforcedLayout.value(), TensorLayout::NHWC);

    const auto& out = io.ioTensorConfigs[1];
    EXPECT_EQ(out.ioType, IOType::Output);
    ASSERT_TRUE(out.enforcedRank.has_value());
    EXPECT_EQ(out.enforcedRank.value(), TensorRank::_4D);
    ASSERT_TRUE(out.enforcedDataType.has_value());
    EXPECT_EQ(out.enforcedDataType.value(), TensorDataType::FP32);
    ASSERT_TRUE(out.enforcedLayout.has_value());
    EXPECT_EQ(out.enforcedLayout.value(), TensorLayout::NCHW);
}

TEST_F(ConfigParserTest, P11_IOConfig_SupportedLists) {
    TempFile tmp{writeTempXml(R"(
<CustomLayerCollection target="NPU">
  <CustomLayer name="listKernel">
    <Kernel type="PREBUILT">
      <Source path="/kernel.elf"/>
    </Kernel>
    <Buffers>
      <Tensor arg-index="0" type="Input" supported-ranks="4D" supported-data-types="FP16,FP32"/>
      <Tensor arg-index="0" type="Output" supported-ranks="4D" supported-data-types="FP16"/>
    </Buffers>
  </CustomLayer>
</CustomLayerCollection>
)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    ASSERT_NO_THROW(parser.loadConfigs(configs));
    const auto& io = configs.at("listKernel")->ioConfig;
    ASSERT_EQ(io.ioTensorConfigs.size(), 2u);

    const auto& in = io.ioTensorConfigs[0];
    ASSERT_EQ(in.supportedRanks.size(), 1u);
    EXPECT_EQ(in.supportedRanks[0], TensorRank::_4D);
    ASSERT_EQ(in.supportedDataTypes.size(), 2u);
    EXPECT_EQ(in.supportedDataTypes[0], TensorDataType::FP16);
    EXPECT_EQ(in.supportedDataTypes[1], TensorDataType::FP32);
}

TEST_F(ConfigParserTest, P12_HeapConfig) {
    TempFile tmp{writeTempXml(R"(
<CustomLayerCollection target="NPU">
  <CustomLayer name="heapKernel">
    <Kernel type="PREBUILT">
      <Source path="/kernel.elf"/>
    </Kernel>
    <Heap max_size="4096"/>
  </CustomLayer>
</CustomLayerCollection>
)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    ASSERT_NO_THROW(parser.loadConfigs(configs));
    EXPECT_EQ(configs.at("heapKernel")->heapConfig.size, 4096u);
}

TEST_F(ConfigParserTest, P13_TilingConfig_Repeated) {
    TempFile tmp{writeTempXml(R"(
<CustomLayerCollection target="NPU">
  <CustomLayer name="tiledKernel">
    <Kernel type="PREBUILT">
      <Source path="/kernel.elf"/>
    </Kernel>
    <Tiling mode="REPEATED"/>
  </CustomLayer>
</CustomLayerCollection>
)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    ASSERT_NO_THROW(parser.loadConfigs(configs));
    EXPECT_EQ(configs.at("tiledKernel")->tilingConfig.tilingMode, Shave::TilingMode::REPEATED);
}

TEST_F(ConfigParserTest, P14_DuplicateName) {
    TempFile tmp{writeTempXml(R"(
<CustomLayerCollection target="NPU">
  <CustomLayer name="dup">
    <Kernel type="PREBUILT"><Source path="/a.elf"/></Kernel>
  </CustomLayer>
  <CustomLayer name="dup">
    <Kernel type="PREBUILT"><Source path="/b.elf"/></Kernel>
  </CustomLayer>
</CustomLayerCollection>
)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    EXPECT_THROW(parser.loadConfigs(configs), std::exception);
}

TEST_F(ConfigParserTest, P15_MultipleKernels) {
    TempFile tmp{writeTempXml(R"(
<CustomLayerCollection target="NPU">
  <CustomLayer name="kernelA">
    <Kernel type="PREBUILT"><Source path="/a.elf"/></Kernel>
  </CustomLayer>
  <CustomLayer name="kernelB">
    <Kernel type="PREBUILT"><Source path="/b.elf"/></Kernel>
  </CustomLayer>
</CustomLayerCollection>
)")};
    CustomLayerConfigMap configs;
    CustomLayerConfigParser parser(tmp.path);
    ASSERT_NO_THROW(parser.loadConfigs(configs));
    ASSERT_EQ(configs.size(), 2u);
    EXPECT_TRUE(configs.count("kernelA"));
    EXPECT_TRUE(configs.count("kernelB"));
    EXPECT_EQ(configs.at("kernelA")->kernelConfig.sourcePath, "/a.elf");
    EXPECT_EQ(configs.at("kernelB")->kernelConfig.sourcePath, "/b.elf");
}

// =============================================================================
// ConfigIrUtilsTest – builds Shave ops with OpBuilder, calls parse* helpers
// =============================================================================

struct ConfigIrUtilsTest : MLIR_UnitBase {
    mlir::MLIRContext ctx;
    mlir::OwningOpRef<mlir::ModuleOp> mod;
    mlir::OpBuilder builder;

    ConfigIrUtilsTest(): MLIR_UnitBase(), ctx(), builder(&ctx) {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Shave::ShaveDialect>();
        ctx.loadDialect<mlir::func::FuncDialect>();
        mod = mlir::ModuleOp::create(builder.getUnknownLoc());
        builder.setInsertionPointToEnd(mod->getBody());
    }

    /// Creates a standalone CustomOpConfigOp with the given unique_id.
    /// Returns the builder positioned inside the config op's region.
    Shave::CustomOpConfigOp makeConfigOp(llvm::StringRef uniqueId) {
        auto configOp = Shave::CustomOpConfigOp::create(builder, builder.getUnknownLoc(), uniqueId);
        auto& region = configOp.getConfigs();
        region.emplaceBlock();
        builder.setInsertionPointToEnd(&region.front());
        return configOp;
    }
};

TEST_F(ConfigIrUtilsTest, I1_ParseKernelConfig_Basic) {
    makeConfigOp("k1");
    auto kernelOp = Shave::CustomOpKernelConfigOp::create(
            builder, builder.getUnknownLoc(), builder.getStringAttr("/path/to/kernel.elf"),
            Shave::KernelTypeAttr::get(&ctx, KernelType::PREBUILT), builder.getStringAttr(""),
            /*uses_dma=*/mlir::UnitAttr{},
            /*uses_dpu=*/mlir::UnitAttr{});

    auto cfg = parseKernelConfig(kernelOp);
    EXPECT_EQ(cfg.sourcePath, "/path/to/kernel.elf");
    EXPECT_EQ(cfg.kernelType, KernelType::PREBUILT);
    EXPECT_FALSE(cfg.usesDMA);
    EXPECT_FALSE(cfg.usesDPU);
}

TEST_F(ConfigIrUtilsTest, I2_ParseKernelConfig_UsesDma) {
    makeConfigOp("k2");
    auto kernelOp = Shave::CustomOpKernelConfigOp::create(
            builder, builder.getUnknownLoc(), builder.getStringAttr("/k.elf"),
            Shave::KernelTypeAttr::get(&ctx, KernelType::PREBUILT), builder.getStringAttr(""),
            /*uses_dma=*/mlir::UnitAttr::get(&ctx),
            /*uses_dpu=*/mlir::UnitAttr{});

    auto cfg = parseKernelConfig(kernelOp);
    EXPECT_TRUE(cfg.usesDMA);
    EXPECT_FALSE(cfg.usesDPU);
}

TEST_F(ConfigIrUtilsTest, I3_ParseKernelConfig_UsesDpu) {
    makeConfigOp("k3");
    auto kernelOp = Shave::CustomOpKernelConfigOp::create(
            builder, builder.getUnknownLoc(), builder.getStringAttr("/k.elf"),
            Shave::KernelTypeAttr::get(&ctx, KernelType::PREBUILT), builder.getStringAttr(""),
            /*uses_dma=*/mlir::UnitAttr{},
            /*uses_dpu=*/mlir::UnitAttr::get(&ctx));

    auto cfg = parseKernelConfig(kernelOp);
    EXPECT_FALSE(cfg.usesDMA);
    EXPECT_TRUE(cfg.usesDPU);
}

TEST_F(ConfigIrUtilsTest, I4_ParseHeapConfig) {
    makeConfigOp("k4");
    auto heapOp = Shave::CustomOpHeapConfigOp::create(builder, builder.getUnknownLoc(), (uint32_t)1024);

    auto cfg = parseHeapConfig(heapOp);
    EXPECT_EQ(cfg.size, 1024u);
}

TEST_F(ConfigIrUtilsTest, I5_ParseTilingConfig_Off) {
    makeConfigOp("k5");
    auto tilingOp = Shave::CustomOpTilingConfigOp::create(builder, builder.getUnknownLoc(), Shave::TilingMode::OFF);

    auto cfg = parseTilingConfig(tilingOp);
    EXPECT_EQ(cfg.tilingMode, Shave::TilingMode::OFF);
}

TEST_F(ConfigIrUtilsTest, I6_ParseTilingConfig_Repeated) {
    makeConfigOp("k6");
    auto tilingOp =
            Shave::CustomOpTilingConfigOp::create(builder, builder.getUnknownLoc(), Shave::TilingMode::REPEATED);

    auto cfg = parseTilingConfig(tilingOp);
    EXPECT_EQ(cfg.tilingMode, Shave::TilingMode::REPEATED);
}

TEST_F(ConfigIrUtilsTest, I7_ParseIOConfig_TensorOnly) {
    makeConfigOp("k7");
    auto ioOp = Shave::CustomOpIOConfigOp::create(builder, builder.getUnknownLoc());
    auto& ioRegion = ioOp.getConfigs();
    ioRegion.emplaceBlock();
    builder.setInsertionPointToEnd(&ioRegion.front());

    Shave::CustomOpIOTensorConfigOp::create(
            builder, builder.getUnknownLoc(),
            /*arg_index=*/(uint32_t)0,
            /*ioType=*/IOType::Input,
            /*supported_rank=*/builder.getArrayAttr({}),
            /*supported_dtype=*/builder.getArrayAttr({}),
            /*enforced_rank=*/Shave::TensorRankAttr::get(&ctx, TensorRank::_4D),
            /*enforced_dtype=*/Shave::TensorDataTypeAttr::get(&ctx, TensorDataType::FP16),
            /*enforced_layout=*/nullptr);

    auto cfg = parseIOConfig(ioOp);
    ASSERT_EQ(cfg.ioTensorConfigs.size(), 1u);
    EXPECT_EQ(cfg.scratchConfigs.size(), 0u);
    const auto& t = cfg.ioTensorConfigs[0];
    EXPECT_EQ(t.argIndex, 0u);
    EXPECT_EQ(t.ioType, IOType::Input);
    ASSERT_TRUE(t.enforcedRank.has_value());
    EXPECT_EQ(t.enforcedRank.value(), TensorRank::_4D);
    ASSERT_TRUE(t.enforcedDataType.has_value());
    EXPECT_EQ(t.enforcedDataType.value(), TensorDataType::FP16);
    EXPECT_FALSE(t.enforcedLayout.has_value());
}

TEST_F(ConfigIrUtilsTest, I8_ParseIOConfig_ScratchOnly) {
    makeConfigOp("k8");
    auto ioOp = Shave::CustomOpIOConfigOp::create(builder, builder.getUnknownLoc());
    auto& ioRegion = ioOp.getConfigs();
    ioRegion.emplaceBlock();
    builder.setInsertionPointToEnd(&ioRegion.front());

    Shave::CustomOpIOScratchConfigOp::create(builder, builder.getUnknownLoc(),
                                             /*arg_index=*/(uint32_t)2,
                                             /*size=*/(uint32_t)512, ScratchLocation::DDR);

    auto cfg = parseIOConfig(ioOp);
    EXPECT_EQ(cfg.ioTensorConfigs.size(), 0u);
    ASSERT_EQ(cfg.scratchConfigs.size(), 1u);
    EXPECT_EQ(cfg.scratchConfigs[0].argIndex, 2u);
    EXPECT_EQ(cfg.scratchConfigs[0].size, 512u);
    EXPECT_EQ(cfg.scratchConfigs[0].location, ScratchLocation::DDR);
}

TEST_F(ConfigIrUtilsTest, I9_ParseIOConfig_Mixed) {
    makeConfigOp("k9");
    auto ioOp = Shave::CustomOpIOConfigOp::create(builder, builder.getUnknownLoc());
    auto& ioRegion = ioOp.getConfigs();
    ioRegion.emplaceBlock();
    builder.setInsertionPointToEnd(&ioRegion.front());

    Shave::CustomOpIOTensorConfigOp::create(builder, builder.getUnknownLoc(), (uint32_t)0, IOType::Output,
                                            builder.getArrayAttr({}), builder.getArrayAttr({}), nullptr, nullptr,
                                            nullptr);

    Shave::CustomOpIOScratchConfigOp::create(builder, builder.getUnknownLoc(), (uint32_t)1, (uint32_t)64,
                                             ScratchLocation::AUTO);

    auto cfg = parseIOConfig(ioOp);
    EXPECT_EQ(cfg.ioTensorConfigs.size(), 1u);
    EXPECT_EQ(cfg.scratchConfigs.size(), 1u);
    EXPECT_EQ(cfg.ioTensorConfigs[0].ioType, IOType::Output);
    EXPECT_EQ(cfg.scratchConfigs[0].location, ScratchLocation::AUTO);
}

TEST_F(ConfigIrUtilsTest, I10_ParseIOConfig_AllEnforcedFields) {
    makeConfigOp("k10");
    auto ioOp = Shave::CustomOpIOConfigOp::create(builder, builder.getUnknownLoc());
    auto& ioRegion = ioOp.getConfigs();
    ioRegion.emplaceBlock();
    builder.setInsertionPointToEnd(&ioRegion.front());

    Shave::CustomOpIOTensorConfigOp::create(builder, builder.getUnknownLoc(), (uint32_t)0, IOType::Input,
                                            builder.getArrayAttr({}), builder.getArrayAttr({}),
                                            Shave::TensorRankAttr::get(&ctx, TensorRank::_4D),
                                            Shave::TensorDataTypeAttr::get(&ctx, TensorDataType::BF16),
                                            Shave::TensorLayoutAttr::get(&ctx, TensorLayout::NHWC));

    auto cfg = parseIOConfig(ioOp);
    ASSERT_EQ(cfg.ioTensorConfigs.size(), 1u);
    const auto& t = cfg.ioTensorConfigs[0];
    ASSERT_TRUE(t.enforcedRank.has_value());
    EXPECT_EQ(t.enforcedRank.value(), TensorRank::_4D);
    ASSERT_TRUE(t.enforcedDataType.has_value());
    EXPECT_EQ(t.enforcedDataType.value(), TensorDataType::BF16);
    ASSERT_TRUE(t.enforcedLayout.has_value());
    EXPECT_EQ(t.enforcedLayout.value(), TensorLayout::NHWC);
}

// =============================================================================
// ConfigFactoryTest – builds a config map, runs factory, inspects emitted IR
// =============================================================================

struct ConfigFactoryTest : MLIR_UnitBase {
    mlir::MLIRContext ctx;
    mlir::OwningOpRef<mlir::ModuleOp> mod;
    mlir::OpBuilder builder;

    ConfigFactoryTest(): MLIR_UnitBase(), ctx(), builder(&ctx) {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Shave::ShaveDialect>();
        ctx.loadDialect<mlir::func::FuncDialect>();
        mod = mlir::ModuleOp::create(builder.getUnknownLoc());
        builder.setInsertionPointToEnd(mod->getBody());
    }

    Shave::CustomOpConfigCollectionOp runFactory(CustomLayerConfigMap& configMap) {
        Shave::CustomLayerConfigFactory factory(configMap, builder);
        factory.setLocation(builder.getUnknownLoc());
        return factory.buildConfigCollection();
    }
};

TEST_F(ConfigFactoryTest, F1_EmptyMap) {
    CustomLayerConfigMap configMap;
    auto collectionOp = runFactory(configMap);

    ASSERT_TRUE(collectionOp);
    EXPECT_EQ(collectionOp.getSymName(), "Config");
    // No children
    size_t childCount = 0;
    for (auto& op : collectionOp.getConfigs().getOps()) {
        (void)op;
        ++childCount;
    }
    EXPECT_EQ(childCount, 0u);
}

TEST_F(ConfigFactoryTest, F2_SingleKernel_KernelConfigOnly) {
    auto cfg = std::make_shared<CustomLayerConfig>();
    cfg->setUniqueId("myKernel");
    cfg->kernelConfig.sourcePath = "/path/to/kernel.elf";
    cfg->kernelConfig.kernelType = KernelType::PREBUILT;

    CustomLayerConfigMap configMap;
    configMap["myKernel"] = cfg;

    auto collectionOp = runFactory(configMap);
    ASSERT_TRUE(collectionOp);

    auto configOp = mlir::dyn_cast_or_null<Shave::CustomOpConfigOp>(collectionOp.lookupSymbol("myKernel"));
    ASSERT_TRUE(configOp);

    auto kernelConfigOps = configOp.getOps<Shave::CustomOpKernelConfigOp>();
    ASSERT_FALSE(kernelConfigOps.empty());
    auto kernelConfigOp = *kernelConfigOps.begin();
    EXPECT_EQ(kernelConfigOp.getSource(), "/path/to/kernel.elf");
    EXPECT_EQ(kernelConfigOp.getKernelType(), KernelType::PREBUILT);
    EXPECT_FALSE(kernelConfigOp.getUsesDma());
    EXPECT_FALSE(kernelConfigOp.getUsesDpu());
}

TEST_F(ConfigFactoryTest, F3_SingleKernel_WithIOConfig) {
    auto cfg = std::make_shared<CustomLayerConfig>();
    cfg->setUniqueId("ioKernel");
    cfg->kernelConfig.sourcePath = "/io.elf";
    cfg->kernelConfig.kernelType = KernelType::PREBUILT;

    CustomLayerConfig::IOTensorConfig tensorCfg;
    tensorCfg.argIndex = 0;
    tensorCfg.ioType = IOType::Input;
    cfg->ioConfig.ioTensorConfigs.push_back(tensorCfg);

    CustomLayerConfig::ScratchConfig scratchCfg;
    scratchCfg.argIndex = 1;
    scratchCfg.size = 128;
    scratchCfg.location = ScratchLocation::CMX;
    cfg->ioConfig.scratchConfigs.push_back(scratchCfg);

    CustomLayerConfigMap configMap;
    configMap["ioKernel"] = cfg;

    auto collectionOp = runFactory(configMap);
    auto configOp = mlir::dyn_cast_or_null<Shave::CustomOpConfigOp>(collectionOp.lookupSymbol("ioKernel"));
    ASSERT_TRUE(configOp);

    auto ioConfigOps = configOp.getOps<Shave::CustomOpIOConfigOp>();
    ASSERT_FALSE(ioConfigOps.empty());
    auto ioConfigOp = *ioConfigOps.begin();

    auto tensorOps = ioConfigOp.getOps<Shave::CustomOpIOTensorConfigOp>();
    ASSERT_FALSE(tensorOps.empty());
    EXPECT_EQ((*tensorOps.begin()).getArgIndex(), 0u);
    EXPECT_EQ((*tensorOps.begin()).getIoType(), IOType::Input);

    auto scratchOps = ioConfigOp.getOps<Shave::CustomOpIOScratchConfigOp>();
    ASSERT_FALSE(scratchOps.empty());
    auto scratchOp = *scratchOps.begin();
    EXPECT_EQ(scratchOp.getArgIndex(), 1u);
    EXPECT_EQ(scratchOp.getSize(), 128u);
    EXPECT_EQ(scratchOp.getLocation(), ScratchLocation::CMX);
}

TEST_F(ConfigFactoryTest, F4_SingleKernel_WithHeapAndTiling) {
    auto cfg = std::make_shared<CustomLayerConfig>();
    cfg->setUniqueId("htKernel");
    cfg->kernelConfig.sourcePath = "/ht.elf";
    cfg->kernelConfig.kernelType = KernelType::PREBUILT;
    cfg->heapConfig.size = 8192;
    cfg->tilingConfig.tilingMode = Shave::TilingMode::REPEATED;

    CustomLayerConfigMap configMap;
    configMap["htKernel"] = cfg;

    auto collectionOp = runFactory(configMap);
    auto configOp = mlir::dyn_cast_or_null<Shave::CustomOpConfigOp>(collectionOp.lookupSymbol("htKernel"));
    ASSERT_TRUE(configOp);

    auto heapOps = configOp.getOps<Shave::CustomOpHeapConfigOp>();
    ASSERT_FALSE(heapOps.empty());
    EXPECT_EQ((*heapOps.begin()).getSize(), 8192u);

    auto tilingOps = configOp.getOps<Shave::CustomOpTilingConfigOp>();
    ASSERT_FALSE(tilingOps.empty());
    EXPECT_EQ((*tilingOps.begin()).getMode(), Shave::TilingMode::REPEATED);
}

TEST_F(ConfigFactoryTest, F5_MultipleKernels) {
    auto cfgA = std::make_shared<CustomLayerConfig>();
    cfgA->setUniqueId("kernelA");
    cfgA->kernelConfig.sourcePath = "/a.elf";
    cfgA->kernelConfig.kernelType = KernelType::PREBUILT;

    auto cfgB = std::make_shared<CustomLayerConfig>();
    cfgB->setUniqueId("kernelB");
    cfgB->kernelConfig.sourcePath = "/b.elf";
    cfgB->kernelConfig.kernelType = KernelType::PREBUILT;

    CustomLayerConfigMap configMap;
    configMap["kernelA"] = cfgA;
    configMap["kernelB"] = cfgB;

    auto collectionOp = runFactory(configMap);

    size_t childCount = 0;
    for (auto& op : collectionOp.getConfigs().getOps()) {
        (void)op;
        ++childCount;
    }
    EXPECT_EQ(childCount, 2u);

    EXPECT_TRUE(collectionOp.lookupSymbol("kernelA"));
    EXPECT_TRUE(collectionOp.lookupSymbol("kernelB"));
}

TEST_F(ConfigFactoryTest, F6_UsesDma_UsesDpu_Propagation) {
    auto cfg = std::make_shared<CustomLayerConfig>();
    cfg->setUniqueId("flagKernel");
    cfg->kernelConfig.sourcePath = "/flag.elf";
    cfg->kernelConfig.kernelType = KernelType::PREBUILT;
    cfg->kernelConfig.usesDMA = true;
    cfg->kernelConfig.usesDPU = true;

    CustomLayerConfigMap configMap;
    configMap["flagKernel"] = cfg;

    auto collectionOp = runFactory(configMap);
    auto configOp = mlir::dyn_cast_or_null<Shave::CustomOpConfigOp>(collectionOp.lookupSymbol("flagKernel"));
    ASSERT_TRUE(configOp);

    auto kernelConfigOps = configOp.getOps<Shave::CustomOpKernelConfigOp>();
    ASSERT_FALSE(kernelConfigOps.empty());
    auto kernelOp = *kernelConfigOps.begin();
    EXPECT_TRUE(kernelOp.getUsesDma());
    EXPECT_TRUE(kernelOp.getUsesDpu());
}

TEST_F(ConfigFactoryTest, F7_RoundTrip) {
    // Build original config struct.
    auto original = std::make_shared<CustomLayerConfig>();
    original->setUniqueId("rtKernel");
    original->kernelConfig.sourcePath = "/rt.elf";
    original->kernelConfig.kernelType = KernelType::PREBUILT;
    original->kernelConfig.usesDMA = true;
    original->kernelConfig.usesDPU = false;

    CustomLayerConfig::IOTensorConfig tensorIn;
    tensorIn.argIndex = 0;
    tensorIn.ioType = IOType::Input;
    tensorIn.enforcedDataType = TensorDataType::FP16;
    original->ioConfig.ioTensorConfigs.push_back(tensorIn);

    CustomLayerConfig::ScratchConfig scratch;
    scratch.argIndex = 1;
    scratch.size = 256;
    scratch.location = ScratchLocation::CMX;
    original->ioConfig.scratchConfigs.push_back(scratch);

    original->heapConfig.size = 2048;
    original->tilingConfig.tilingMode = Shave::TilingMode::OFF;

    CustomLayerConfigMap configMap;
    configMap["rtKernel"] = original;

    // Factory -> IR.
    auto collectionOp = runFactory(configMap);
    auto configOp = mlir::dyn_cast_or_null<Shave::CustomOpConfigOp>(collectionOp.lookupSymbol("rtKernel"));
    ASSERT_TRUE(configOp);

    // Recover kernel config via ir_utils.
    {
        auto kernelConfigOps = configOp.getOps<Shave::CustomOpKernelConfigOp>();
        ASSERT_FALSE(kernelConfigOps.empty());
        auto recovered = parseKernelConfig(*kernelConfigOps.begin());
        EXPECT_EQ(recovered.sourcePath, "/rt.elf");
        EXPECT_EQ(recovered.kernelType, KernelType::PREBUILT);
        EXPECT_TRUE(recovered.usesDMA);
        EXPECT_FALSE(recovered.usesDPU);
    }

    // Recover IO config via ir_utils.
    {
        auto ioConfigOps = configOp.getOps<Shave::CustomOpIOConfigOp>();
        ASSERT_FALSE(ioConfigOps.empty());
        auto recovered = parseIOConfig(*ioConfigOps.begin());

        ASSERT_EQ(recovered.ioTensorConfigs.size(), 1u);
        EXPECT_EQ(recovered.ioTensorConfigs[0].argIndex, 0u);
        EXPECT_EQ(recovered.ioTensorConfigs[0].ioType, IOType::Input);
        ASSERT_TRUE(recovered.ioTensorConfigs[0].enforcedDataType.has_value());
        EXPECT_EQ(recovered.ioTensorConfigs[0].enforcedDataType.value(), TensorDataType::FP16);

        ASSERT_EQ(recovered.scratchConfigs.size(), 1u);
        EXPECT_EQ(recovered.scratchConfigs[0].argIndex, 1u);
        EXPECT_EQ(recovered.scratchConfigs[0].size, 256u);
        EXPECT_EQ(recovered.scratchConfigs[0].location, ScratchLocation::CMX);
    }

    // Recover heap config via ir_utils.
    {
        auto heapOps = configOp.getOps<Shave::CustomOpHeapConfigOp>();
        ASSERT_FALSE(heapOps.empty());
        auto recovered = parseHeapConfig(*heapOps.begin());
        EXPECT_EQ(recovered.size, 2048u);
    }

    // Recover tiling config via ir_utils.
    {
        auto tilingOps = configOp.getOps<Shave::CustomOpTilingConfigOp>();
        ASSERT_FALSE(tilingOps.empty());
        auto recovered = parseTilingConfig(*tilingOps.begin());
        EXPECT_EQ(recovered.tilingMode, Shave::TilingMode::OFF);
    }
}

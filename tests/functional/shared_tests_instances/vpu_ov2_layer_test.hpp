//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/npu_test_env_cfg.hpp"
#include "vpu_test_tool.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <intel_npu/npu_private_properties.hpp>
#include <shared_test_classes/base/ov_subgraph.hpp>

#include "common/tensor_comparison.hpp"

#include <functional>
#include <optional>
#include <sstream>

namespace ov::test::utils {

using SkipMessage = std::optional<std::string>;
using SkipCallback = std::function<void(std::stringstream&)>;
namespace Platform = ov::intel_npu::Platform;

enum class VpuCompilationMode {
    ReferenceSW,
    DefaultHW,
};

class VpuOv2LayerTest : virtual public ov::test::SubgraphBaseTest {
protected:
    static const ov::test::utils::VpuTestEnvConfig& envConfig;
    VpuTestTool testTool;

public:
    VpuOv2LayerTest();

    void setSkipCompilationCallback(SkipCallback skipCallback);
    void setSkipInferenceCallback(SkipCallback skipCallback);

protected:
    void importModel();
    void exportModel();
    void importInput();
    void exportInput();
    void exportOutput();
    std::vector<ov::Tensor> importReference();
    void exportReference(const std::vector<ov::Tensor>& refs);

private:
    bool skipCompilationImpl();

    void printNetworkConfig() const;

    using ErrorMessage = std::optional<std::string>;
    [[nodiscard]] ErrorMessage runTest();
    [[nodiscard]] ErrorMessage skipInferenceImpl();
    bool ensureCICompliantName() const;

public:
    void setReferenceSoftwareMode();
    void setDefaultHardwareMode();
    void setHostCompileMode(std::string_view mode = "HostCompile");

    void setPluginCompilerType();
    void setBatchCompilerMode(const std::string& mode);
    void setBypassUmdCaching();

    void setSingleClusterMode();
    void setPerformanceHintLatency();

    void enableProfiling();
    void enableTurbo();

    bool isReferenceSoftwareMode() const;
    bool isDefaultHardwareMode() const;

    void run(const std::string_view platform);

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void validate() override;

private:
    // use public run(const std::string_view platform) function to always set platform explicitly
    void run() override;
    void setPlatform(const std::string_view platform);

    SkipCallback skipCompilationCallback = nullptr;
    SkipCallback skipInferenceCallback = nullptr;
    vpux::Logger _log = vpux::Logger::global();
};

std::vector<std::vector<ov::Shape>> combineStaticShapes(const std::vector<ov::test::InputShape>& inputs);
ov::PartialShape getBoundedShape(const ov::test::InputShape& shape);

}  // namespace ov::test::utils

namespace ov::test::subgraph {
namespace Platform = ov::test::utils::Platform;
using ov::test::utils::VpuOv2LayerTest;
}  // namespace ov::test::subgraph

namespace LayerTestsDefinitions {
namespace Platform = ov::test::utils::Platform;
using ov::test::utils::VpuOv2LayerTest;
}  // namespace LayerTestsDefinitions

namespace SubgraphTestsDefinitions {
namespace Platform = ov::test::utils::Platform;
using ov::test::utils::VpuOv2LayerTest;
}  // namespace SubgraphTestsDefinitions

namespace ov::test {
namespace Platform = ov::test::utils::Platform;
using ov::test::utils::VpuOv2LayerTest;
}  // namespace ov::test

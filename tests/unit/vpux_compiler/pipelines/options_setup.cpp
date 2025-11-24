//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include "vpux/compiler/pipelines/options_setup.hpp"

#include "common/utils.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;
using MLIR_OptionsSetup = MLIR_UnitBase;

class OptionsSetupTest : public MLIR_UnitBase {
public:
    void SetUp() override {
        _optionDesc = std::make_shared<intel_npu::OptionsDesc>();
        _optionDesc->add<intel_npu::PLATFORM>();

        _config = intel_npu::Config(_optionDesc);
        // Required, otherwise we get an missing PERFORMANCE_OVERRIDE hint exception down the line.
        _config->update({{std::string(intel_npu::PLATFORM::key()), "VPU4000"}});

        Logger::global().debug("Using config:\n{0}\n", _config.value().toString());
    }

    VPU::InitCompilerOptions _initCompilerOptions;
    PublicOptions _publicOptions;
    std::shared_ptr<intel_npu::OptionsDesc> _optionDesc;
    std::optional<intel_npu::Config> _config;
};

TEST_F(OptionsSetupTest, NoImplementation) {
    class NoImplementation : public OptionsSetup<NoImplementation, PublicOptions> {
    public:
        using Base = OptionsSetup<NoImplementation, PublicOptions>;
        using Base::Base;
    };

    EXPECT_THROW(NoImplementation(&_initCompilerOptions, &_publicOptions), vpux::Exception);
    EXPECT_THROW(NoImplementation(_config.value()), vpux::Exception);
}

TEST_F(OptionsSetupTest, PartialImplementation1) {
    class PartialImplementation : public OptionsSetup<PartialImplementation, PublicOptions> {
    public:
        using Base = OptionsSetup<PartialImplementation, PublicOptions>;
        using Base::Base;
        using Base::setupOptionsImpl;

        static void setupLitTestOptionsImpl(PublicOptions&, VPU::InitCompilerOptions&) {
        }
    };

    std::ignore = PartialImplementation(&_initCompilerOptions, &_publicOptions);
    EXPECT_THROW(PartialImplementation(_config.value()), vpux::Exception);
}

TEST_F(OptionsSetupTest, PartialImplementation2) {
    class PartialImplementation : public OptionsSetup<PartialImplementation, PublicOptions> {
    public:
        using Base = OptionsSetup<PartialImplementation, PublicOptions>;
        using Base::Base;
        using Base::setupLitTestOptionsImpl;

        static void setupOptionsImpl(PublicOptions&, VPU::InitCompilerOptions&, const intel_npu::Config&) {
        }
    };

    EXPECT_THROW(PartialImplementation(&_initCompilerOptions, &_publicOptions), vpux::Exception);
    std::ignore = PartialImplementation(_config.value());
}

// Templates don't work inside test fixtures.
template <class ConcreteModel>
class IntermediateImplementation : public OptionsSetup<ConcreteModel, PublicOptions> {
public:
    using Base = OptionsSetup<ConcreteModel, PublicOptions>;
    using Base::Base;

protected:
    static void setupLitTestOptionsImpl(PublicOptions&, VPU::InitCompilerOptions&) {
    }

    static void setupOptionsImpl(PublicOptions&, VPU::InitCompilerOptions&, const intel_npu::Config&) {
    }
};

TEST_F(OptionsSetupTest, IntermediateImplementation) {
    class ConcreteImplementation : public IntermediateImplementation<ConcreteImplementation> {
    public:
        using Base = IntermediateImplementation<ConcreteImplementation>;
        using Base::Base;
        using Base::setupLitTestOptionsImpl;

        static void setupOptionsImpl(PublicOptions& options, VPU::InitCompilerOptions& initCompilerOptions,
                                     const intel_npu::Config& config) {
            Base::setupOptionsImpl(options, initCompilerOptions, config);
            EXPECT_THROW(Base::Base::setupOptionsImpl(options, initCompilerOptions, config), vpux::Exception);
        }
    };

    std::ignore = ConcreteImplementation(&_initCompilerOptions, &_publicOptions);
    std::ignore = ConcreteImplementation(_config.value());
}

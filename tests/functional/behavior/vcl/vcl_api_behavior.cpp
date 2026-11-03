//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compatibility/npu_driver_compiler.h"
#include "gtest/gtest.h"
#include "vcl_api.hpp"

#include <algorithm>
#include <cstdint>
#include <ios>
#include <iterator>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace VCLTest {

namespace {

template <typename Handle>
struct HandleTraits;

template <>
struct HandleTraits<vcl_compiler_handle_t> {
    static void destroy(vcl_compiler_handle_t handle) {
        vclCompilerDestroy(handle);
    }
};

template <typename Handle>
class ScopedHandle {
public:
    ScopedHandle() = default;
    ScopedHandle(const ScopedHandle&) = delete;
    ScopedHandle(ScopedHandle&&) = delete;
    ScopedHandle& operator=(const ScopedHandle&) = delete;
    ScopedHandle& operator=(ScopedHandle&&) = delete;

    ~ScopedHandle() {
        reset();
    }

    Handle* out() {
        reset();
        return &_handle;
    }

    Handle get() const {
        return _handle;
    }

    void reset() {
        if (_handle != nullptr) {
            HandleTraits<Handle>::destroy(_handle);
            _handle = nullptr;
        }
    }

private:
    Handle _handle = nullptr;
};

using ScopedCompiler = ScopedHandle<vcl_compiler_handle_t>;

/// Creates a compiler bound to @p deviceDesc, or a device-independent one when it is nullptr.
///
/// @details vcl_api.hpp's wrappers load the library on first use, so nothing has to be initialized
/// before calling this.
vcl_result_t createCompiler(vcl_device_desc_t* deviceDesc, ScopedCompiler& compiler, vcl_log_handle_t& logHandle) {
    vcl_compiler_desc_t compilerDesc{};
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_INFO;

    logHandle = nullptr;
    return vclCompilerCreate(&compilerDesc, deviceDesc, compiler.out(), &logHandle);
}

/// The descriptor the driver reports for the device with @p deviceID.
///
/// @details Only the device ID selects a platform, and with it the option set; the revision and
/// the tile count play no part in it, so they are the same for every device here.
vcl_device_desc_t deviceDescFor(uint32_t deviceID) {
    return {sizeof(vcl_device_desc_t), deviceID, /* revision */ 3, /* tileCount */ 2};
}

/// The public compilation options @p compiler advertises, as a set.
std::set<std::string> advertisedOptions(vcl_compiler_handle_t compiler) {
    uint64_t optionsSize = 0;
    if (vclGetCompilerSupportedOptions(compiler, nullptr, &optionsSize) != VCL_RESULT_SUCCESS || optionsSize == 0) {
        ADD_FAILURE() << "Cannot size the supported options list";
        return {};
    }

    std::vector<char> buffer(optionsSize, '\0');
    if (vclGetCompilerSupportedOptions(compiler, buffer.data(), &optionsSize) != VCL_RESULT_SUCCESS) {
        ADD_FAILURE() << "Cannot read the supported options list";
        return {};
    }

    std::set<std::string> options;
    std::istringstream optionStream{std::string(buffer.data())};
    for (std::string option; optionStream >> option;) {
        options.insert(std::move(option));
    }
    return options;
}

}  // namespace

TEST(VCLApiBehaviorTest, VclGetVersionReturnsRuntimeVersion) {
    vcl_version_info_t compilerVersion{};
    vcl_version_info_t profilingVersion{};

    const auto ret = vclGetVersion(&compilerVersion, &profilingVersion);
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "vclGetVersion failed! Result:0x" << std::hex << uint64_t(ret) << std::dec
                                       << std::endl;
    EXPECT_NE(compilerVersion.major, 0);
    EXPECT_NE(profilingVersion.major, 0);
}

TEST(VCLApiBehaviorTest, CompilerSupportedOptionsApi) {
    ScopedCompiler compiler;
    vcl_log_handle_t logHandle = nullptr;
    auto ret = createCompiler(/* deviceDesc */ nullptr, compiler, logHandle);
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS);

    uint64_t optionsSize = 0;

    ret = vclGetCompilerSupportedOptions(compiler.get(), nullptr, &optionsSize);
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS) << "vclGetCompilerSupportedOptions(size) failed! Result:0x" << std::hex
                                       << uint64_t(ret) << std::dec << std::endl;
    ASSERT_NE(optionsSize, 0U);

    std::vector<char> options(optionsSize, '\0');
    ret = vclGetCompilerSupportedOptions(compiler.get(), options.data(), &optionsSize);
    ASSERT_EQ(ret, VCL_RESULT_SUCCESS) << "vclGetCompilerSupportedOptions(data) failed! Result:0x" << std::hex
                                       << uint64_t(ret) << std::dec << std::endl;

    EXPECT_EQ(options.back(), '\0') << "Supported options list is expected to be null-terminated";
    const std::string_view optionsList(options.data());
    EXPECT_FALSE(optionsList.empty()) << "Supported options list should not be empty";
}

// The device a compiler is created for narrows its options to the ones that device's platform
// supports. NPU_ENABLE_STRIDES_FOR is the only option that differs between platforms today -
// NPU3720 cannot honor it, the compiler throwing "ENABLE_STRIDES_FOR not supported on NPU3720" -
// so it is the probe for the narrowing, on both of the ways a compiler reports its option set.
//
// This case has to bind a device, and every other case in this file deliberately does not: a
// compiler created without a descriptor has a device ID of 0, which names no platform, so it is
// handed every known option and the narrowing is never reached. Whether the platform table or the
// selection is wrong, the difference only shows up here.
TEST(VCLApiBehaviorTest, DeviceNarrowsSupportedOptionsToItsPlatform) {
    /// The option NPU3720 does not support, and a value to check it with.
    constexpr const char* narrowedOption = "NPU_ENABLE_STRIDES_FOR";
    constexpr const char* narrowedOptionValue = "input_0";

    /// Asserts both reports agree on whether this device supports @p narrowedOption, and hands the
    /// advertised set back so the two platforms can be compared as a whole below.
    const auto optionsOfDevice = [&](uint32_t deviceID, std::string_view platform, bool expectSupported) {
        std::ostringstream deviceLabel;
        deviceLabel << "NPU" << platform << " (device ID 0x" << std::hex << deviceID << std::dec << ")";
        const auto device = deviceLabel.str();

        vcl_device_desc_t deviceDesc = deviceDescFor(deviceID);
        ScopedCompiler compiler;
        vcl_log_handle_t logHandle = nullptr;
        EXPECT_EQ(createCompiler(&deviceDesc, compiler, logHandle), VCL_RESULT_SUCCESS) << device;

        const auto expectedResult = expectSupported ? VCL_RESULT_SUCCESS : VCL_RESULT_ERROR_UNSUPPORTED_FEATURE;
        auto options = advertisedOptions(compiler.get());
        EXPECT_EQ(options.count(narrowedOption) != 0, expectSupported)
                << device << " advertises the wrong answer for " << narrowedOption;
        EXPECT_EQ(vclGetCompilerIsOptionSupported(compiler.get(), narrowedOption, nullptr), expectedResult) << device;
        EXPECT_EQ(vclGetCompilerIsOptionSupported(compiler.get(), narrowedOption, narrowedOptionValue), expectedResult)
                << device;
        return options;
    };

    // Both NPU3720 device IDs, so that the table is checked by row and not just by platform.
    const auto mtlOptions = optionsOfDevice(0x7D1D, "3720", /* expectSupported */ false);
    const auto arlOptions = optionsOfDevice(0xAD1D, "3720", /* expectSupported */ false);
    const auto lnlOptions = optionsOfDevice(0x643E, "4000", /* expectSupported */ true);
    ASSERT_FALSE(mtlOptions.empty());
    EXPECT_EQ(mtlOptions, arlOptions) << "Two device IDs of one platform must yield one option set";

    // The narrowing takes that option and nothing else: an NPU3720 device is an NPU4000 one minus
    // exactly NPU_ENABLE_STRIDES_FOR. This is what fails if the platform table gains or loses an
    // entry it should not have.
    std::vector<std::string> withheldOnNPU3720;
    std::set_difference(lnlOptions.begin(), lnlOptions.end(), mtlOptions.begin(), mtlOptions.end(),
                        std::back_inserter(withheldOnNPU3720));
    EXPECT_EQ(withheldOnNPU3720, std::vector<std::string>{narrowedOption});

    std::vector<std::string> extraOnNPU3720;
    std::set_difference(mtlOptions.begin(), mtlOptions.end(), lnlOptions.begin(), lnlOptions.end(),
                        std::back_inserter(extraOnNPU3720));
    EXPECT_TRUE(extraOnNPU3720.empty()) << "NPU3720 advertises options NPU4000 does not: "
                                        << testing::PrintToString(extraOnNPU3720);

    // A compiler with no device has no platform to narrow to, so it keeps every known option.
    // This is the set the rest of the file's cases see, and why they cannot catch the above.
    ScopedCompiler deviceIndependentCompiler;
    vcl_log_handle_t logHandle = nullptr;
    ASSERT_EQ(createCompiler(/* deviceDesc */ nullptr, deviceIndependentCompiler, logHandle), VCL_RESULT_SUCCESS);
    EXPECT_EQ(advertisedOptions(deviceIndependentCompiler.get()), lnlOptions);
    EXPECT_EQ(vclGetCompilerIsOptionSupported(deviceIndependentCompiler.get(), narrowedOption, nullptr),
              VCL_RESULT_SUCCESS);
}

}  // namespace VCLTest

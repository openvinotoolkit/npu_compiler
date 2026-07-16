//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_tests_common.h"

#include <stdint.h>
#include <stdlib.h>
#include <functional>
#include <iostream>
#include <string_view>
#include <type_traits>
namespace VCLTest {
class VCLSingleThreadTest : public VCLTestsCommon {
public:
    /**
     * @brief Call L0 compiler to compile model to blob
     *
     * @param options Build flags of a model
     */
    vcl_result_t run(const std::string& options);
};

vcl_result_t VCLSingleThreadTest::run(const std::string& options) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    /// Default device is 4000, can be updated by test config
    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_ERROR;
    vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t), 0x643e, 3, 5};
    vcl_compiler_handle_t compiler = nullptr;
    ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, nullptr);
    if (ret) {
        printErrorInfo("Failed to create compiler! Result: 0x", ret);
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret) {
        printErrorInfo("Failed to query compiler props! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        std::cout << "\n############################################\n\n";
        std::cout << "Current compiler info:\n";
        std::cout << "ID: " << compilerProp.id << std::endl;
        std::cout << "Version: " << compilerProp.version.major << "." << compilerProp.version.minor << std::endl;
        std::cout << "\tSupported opsets: " << compilerProp.supportedOpsets << std::endl;
        std::cout << "\n############################################\n\n";
    }

    vcl_executable_handle_t executable = nullptr;
    vcl_executable_desc_t exeDesc = {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};

    ret = vclExecutableCreate(compiler, exeDesc, &executable);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create executable handle! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }
    uint64_t blobSize = 0;
    ret = vclExecutableGetSerializableBlob(executable, nullptr, &blobSize);
    if (ret != VCL_RESULT_SUCCESS || blobSize == 0) {
        printErrorInfo("Failed to get blob size! Result: 0x", ret);
        vclExecutableDestroy(executable);
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        uint8_t* blob = (uint8_t*)malloc(blobSize);
        if (!blob) {
            std::cerr << "Failed to alloc memory for blob!\n";
            vclExecutableDestroy(executable);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_OUT_OF_MEMORY;
        }
        ret = vclExecutableGetSerializableBlob(executable, blob, &blobSize);
        if (ret == VCL_RESULT_SUCCESS) {
#ifdef BLOB_DUMP
            const std::string blobName = std::string("output.net");
            std::ofstream bfos(blobName, std::ios::binary);
            if (!bfos.is_open()) {
                std::cerr << "Can not open " << blobName << ", skip dump!\n";
            } else {
                bfos.write(reinterpret_cast<char*>(blob), blobSize);
                if (bfos.fail()) {
                    std::cerr << "Short write to " << blobName << ", the file is invalid!\n";
                }
            }
            bfos.close();
#endif  // BLOB_DUMP
        }
        free(blob);
    }

    ret = vclExecutableDestroy(executable);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy executable! Result: 0x", ret);
        ret = vclCompilerDestroy(compiler);
        return ret;
    }
    executable = nullptr;

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy compiler! Result: 0x", ret);
        return ret;
    }
    return ret;
}

TEST_P(VCLSingleThreadTest, compileModel) {
    EXPECT_EQ(run(getNetOptions()), VCL_RESULT_SUCCESS);
}

/// The path of config files for tests
const auto cidTool = VCLSingleThreadTest::getCidToolPath();
/// Models and configs for smoke test
const auto smokeIRInfos = VCLSingleThreadTest::readJson2Vec(cidTool + VCLTest::SMOKE_TEST_CONFIG);
/// Models and configs for normal test
const auto irInfos = VCLSingleThreadTest::readJson2Vec(cidTool + VCLTest::TEST_CONFIG);
/// Parameters for smoke tests
const auto smokeParams = testing::Combine(testing::ValuesIn(smokeIRInfos));
/// Parameters for normal tests
const auto params = testing::Combine(testing::ValuesIn(irInfos));

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadCompilation, VCLSingleThreadTest, smokeParams,
                         VCLSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadCompilation, VCLSingleThreadTest, params, VCLSingleThreadTest::getTestCaseName);

template <typename VclAllocT>
class VCLAllocatorSingleThreadTestBase : public VCLTestsCommon {
public:
    /**
     * @brief Call L0 compiler to compile model to blob
     *
     * @param options Build flags of a model
     */
    vcl_result_t run(const std::string& options);
};

template <typename VclAllocT>
vcl_result_t VCLAllocatorSingleThreadTestBase<VclAllocT>::run(const std::string& options) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    /// Default device is 4000, can be updated by test config
    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_INFO;
    vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t), 0x643e, 3, 5};
    vcl_compiler_handle_t compiler = nullptr;
    ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, nullptr);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create compiler! Result: 0x", ret);
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to query compiler props! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }
    std::cout << "\n############################################\n\n";
    std::cout << " Current compiler info:\n"
              << " ID: " << compilerProp.id << "\n"
              << " Version: " << compilerProp.version.major << "." << compilerProp.version.minor << "\n"
              << "\tSupported opsets: " << compilerProp.supportedOpsets << "\n";
    std::cout << "\n############################################\n\n";

    uint8_t* blob = nullptr;
    uint64_t size = 0;

    vcl_executable_desc_t exeDesc = {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};
    VclAllocT allocator;
    std::function<void()> deallocate;

    if constexpr (std::is_same_v<VclAllocT, vcl_allocator2_t>) {
        allocator.allocate = VCLTest::allocateBlob2;
        allocator.deallocate = VCLTest::deallocateBlob2;
        deallocate = [&] {
            allocator.deallocate(&allocator, blob);
        };
        ret = vclAllocatedExecutableCreate2(compiler, exeDesc, &allocator, &blob, &size);
    } else {
        static_assert(std::is_same_v<VclAllocT, vcl_allocator_t>);
        allocator.allocate = VCLTest::allocateBlob;
        allocator.deallocate = VCLTest::deallocateBlob;
        deallocate = [&] {
            allocator.deallocate(blob);
        };
        ret = vclAllocatedExecutableCreate(compiler, exeDesc, &allocator, &blob, &size);
    }

    if (ret != VCL_RESULT_SUCCESS || blob == nullptr || size == 0) {
        printErrorInfo("Failed to create executable handle! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }

#ifdef BLOB_DUMP
    auto ir = GetParam();
    auto netInfo = std::get<0>(ir);
    const std::string blobName = "ct0_" + netInfo.at("network") + ".net.allocator";
    std::ofstream bfos(blobName, std::ios::binary);
    if (!bfos.is_open()) {
        std::cerr << "Cannot open " << blobName << ", skip dump!" << std::endl;
    } else {
        bfos.write(reinterpret_cast<char*>(blob), size);
        if (bfos.fail()) {
            std::cerr << "Short write to " << blobName << ", the file is invalid!" << std::endl;
        }
    }
    bfos.close();
#endif  // BLOB_DUMP

    deallocate();

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy compiler! Result: 0x", ret);
        return ret;
    }
    return ret;
}

struct VCLAllocatorSingleThreadTest : public VCLAllocatorSingleThreadTestBase<vcl_allocator_t> {};
struct VCLAllocator2SingleThreadTest : public VCLAllocatorSingleThreadTestBase<vcl_allocator2_t> {};

TEST_P(VCLAllocatorSingleThreadTest, compileModel) {
    EXPECT_EQ(run(getNetOptions()), VCL_RESULT_SUCCESS);
}

TEST_P(VCLAllocator2SingleThreadTest, compileModel) {
    EXPECT_EQ(run(getNetOptions()), VCL_RESULT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadCompilation, VCLAllocatorSingleThreadTest, smokeParams,
                         VCLAllocatorSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadCompilation, VCLAllocator2SingleThreadTest, smokeParams,
                         VCLAllocatorSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadCompilation, VCLAllocatorSingleThreadTest, params,
                         VCLAllocatorSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadCompilation, VCLAllocator2SingleThreadTest, params,
                         VCLAllocatorSingleThreadTest::getTestCaseName);

class VCLAllocator3SingleThreadTest : public VCLTestsCommon {
public:
    vcl_result_t run(const std::string& options);
};

vcl_result_t VCLAllocator3SingleThreadTest::run(const std::string& options) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_INFO;
    vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t), 0x643e, 3, 5};
    vcl_compiler_handle_t compiler = nullptr;
    ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, nullptr);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create compiler! Result: 0x", ret);
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to query compiler props! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }
    std::cout << "\n############################################\n\n";
    std::cout << " Current compiler info:\n"
              << " ID: " << compilerProp.id << "\n"
              << " Version: " << compilerProp.version.major << "." << compilerProp.version.minor << "\n"
              << "\tSupported opsets: " << compilerProp.supportedOpsets << "\n";
    std::cout << "\n############################################\n\n";

    uint8_t* blob = nullptr;
    uint64_t blobSize = 0;
    uint8_t* compatibilityReqBuffer = nullptr;
    uint64_t compatibilityReqSize = 0;

    vcl_executable_desc_t exeDesc = {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};
    vcl_allocator2_t allocator;
    allocator.allocate = VCLTest::allocateBlob2;
    allocator.deallocate = VCLTest::deallocateBlob2;

    ret = vclAllocatedExecutableCreate3(compiler, exeDesc, &allocator, &blob, &blobSize, &compatibilityReqBuffer,
                                        &compatibilityReqSize);
    if (ret != VCL_RESULT_SUCCESS || blob == nullptr || blobSize == 0 || compatibilityReqBuffer == nullptr ||
        compatibilityReqSize == 0) {
        printErrorInfo("Failed to create executable via vclAllocatedExecutableCreate3! Result: 0x", ret);
        if (compatibilityReqBuffer != nullptr) {
            allocator.deallocate(&allocator, compatibilityReqBuffer);
        }
        if (blob != nullptr) {
            allocator.deallocate(&allocator, blob);
        }
        vclCompilerDestroy(compiler);
        return ret != VCL_RESULT_SUCCESS ? ret : VCL_RESULT_ERROR_UNKNOWN;
    }

    ret = vclGetCompilerIsOptionSupported(compiler, "COMPATIBILITY_CHECK",
                                          reinterpret_cast<const char*>(compatibilityReqBuffer));
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("vclGetCompilerIsOptionSupported failed! Result: 0x", ret);
        allocator.deallocate(&allocator, compatibilityReqBuffer);
        allocator.deallocate(&allocator, blob);
        vclCompilerDestroy(compiler);
        return ret;
    }

    allocator.deallocate(&allocator, compatibilityReqBuffer);
    allocator.deallocate(&allocator, blob);

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy compiler! Result: 0x", ret);
        return ret;
    }
    return ret;
}

TEST_P(VCLAllocator3SingleThreadTest, compileModelWithCompatibilityString) {
    EXPECT_EQ(run(getNetOptions()), VCL_RESULT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadCompilation, VCLAllocator3SingleThreadTest, smokeParams,
                         VCLAllocatorSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadCompilation, VCLAllocator3SingleThreadTest, params,
                         VCLAllocatorSingleThreadTest::getTestCaseName);

class VCLAllocator4SingleThreadTest : public VCLTestsCommon {
public:
    vcl_result_t run(const std::string& options);
};

vcl_result_t VCLAllocator4SingleThreadTest::run(const std::string& options) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_INFO;
    vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t), 0x643e, 3, 5};
    vcl_compiler_handle_t compiler = nullptr;
    ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, nullptr);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create compiler! Result: 0x", ret);
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to query compiler props! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }
    std::cout << "\n############################################\n\n";
    std::cout << " Current compiler info:\n"
              << " ID: " << compilerProp.id << "\n"
              << " Version: " << compilerProp.version.major << "." << compilerProp.version.minor << "\n"
              << "\tSupported opsets: " << compilerProp.supportedOpsets << "\n";
    std::cout << "\n############################################\n\n";

    uint8_t* blob = nullptr;
    uint64_t blobSize = 0;
    vcl_executable_handle_t executable = nullptr;

    vcl_executable_desc_t exeDesc = {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};
    vcl_allocator2_t allocator;
    allocator.allocate = VCLTest::allocateBlob2;
    allocator.deallocate = VCLTest::deallocateBlob2;

    ret = vclAllocatedExecutableCreate4(compiler, exeDesc, &allocator, &blob, &blobSize, &executable);
    if (ret != VCL_RESULT_SUCCESS || blob == nullptr || blobSize == 0 || executable == nullptr) {
        printErrorInfo("Failed to create executable via vclAllocatedExecutableCreate4! Result: 0x", ret);
        if (blob != nullptr) {
            allocator.deallocate(&allocator, blob);
        }
        if (executable != nullptr) {
            vclExecutableDestroy(executable);
        }
        vclCompilerDestroy(compiler);
        return ret != VCL_RESULT_SUCCESS ? ret : VCL_RESULT_ERROR_UNKNOWN;
    }

    /// First call: get the required buffer size for the compatibility string
    uint64_t compatibilityStringSize = 0;
    ret = vclExecutableGetCompatibilityString(executable, nullptr, &compatibilityStringSize);
    if (ret != VCL_RESULT_SUCCESS || compatibilityStringSize == 0) {
        printErrorInfo("Failed to get compatibility string size! Result: 0x", ret);
        allocator.deallocate(&allocator, blob);
        vclExecutableDestroy(executable);
        vclCompilerDestroy(compiler);
        return ret != VCL_RESULT_SUCCESS ? ret : VCL_RESULT_ERROR_UNKNOWN;
    }

    /// Second call: retrieve the compatibility string into the allocated buffer
    std::vector<char> compatibilityString(compatibilityStringSize);
    ret = vclExecutableGetCompatibilityString(executable, compatibilityString.data(), &compatibilityStringSize);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to get compatibility string! Result: 0x", ret);
        allocator.deallocate(&allocator, blob);
        vclExecutableDestroy(executable);
        vclCompilerDestroy(compiler);
        return ret;
    }
    std::cout << "Compatibility string: " << compatibilityString.data() << "\n";
    EXPECT_TRUE(std::string_view(compatibilityString.data(), compatibilityString.size()).substr(0, 9) == "compiler=");

    allocator.deallocate(&allocator, blob);

    ret = vclExecutableDestroy(executable);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy executable! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy compiler! Result: 0x", ret);
        return ret;
    }
    return ret;
}

TEST_P(VCLAllocator4SingleThreadTest, compileModelWithCompatibilityString) {
    EXPECT_EQ(run(getNetOptions()), VCL_RESULT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadCompilation, VCLAllocator4SingleThreadTest, smokeParams,
                         VCLAllocatorSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadCompilation, VCLAllocator4SingleThreadTest, params,
                         VCLAllocatorSingleThreadTest::getTestCaseName);

class VCLQueryNetworkSingleThreadTest : public VCLTestsCommon {
public:
    /**
     * @brief Call query network APIs in a single thread for one model
     *
     * @param options Build flags of a model
     */
    vcl_result_t run(const std::string& options);
};

vcl_result_t VCLQueryNetworkSingleThreadTest::run(const std::string& options) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_ERROR;

    vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t), 0x643e, 3, 5};
    vcl_compiler_handle_t compiler = nullptr;
    ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, nullptr);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create compiler! Result: 0x", ret);
        return ret;
    }

    vcl_query_handle_t query = nullptr;
    vcl_query_desc_t desc = {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};

    ret = vclQueryNetworkCreate(compiler, desc, &query);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create query handle! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }

    uint64_t layerSize = 0;
    ret = vclQueryNetwork(query, nullptr, &layerSize);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to get query result size! Result: 0x", ret);
        vclQueryNetworkDestroy(query);
        vclCompilerDestroy(compiler);
        return ret;
    }
    if (layerSize == 0) {
        std::cerr << "Query result size is zero after first vclQueryNetwork call." << std::endl;
        vclQueryNetworkDestroy(query);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    std::vector<uint8_t> layerRawData(layerSize);
    ret = vclQueryNetwork(query, layerRawData.data(), &layerSize);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to get query result! Result: 0x", ret);
        vclQueryNetworkDestroy(query);
        vclCompilerDestroy(compiler);
        return ret;
    }

    ret = vclQueryNetworkDestroy(query);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy query handle! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }
    query = nullptr;

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy compiler! Result: 0x", ret);
        return ret;
    }

    return ret;
}

TEST_P(VCLQueryNetworkSingleThreadTest, queryNetwork) {
    EXPECT_EQ(run(getNetOptions()), VCL_RESULT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadQueryNetwork, VCLQueryNetworkSingleThreadTest, smokeParams,
                         VCLQueryNetworkSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadQueryNetwork, VCLQueryNetworkSingleThreadTest, params,
                         VCLQueryNetworkSingleThreadTest::getTestCaseName);

}  // namespace VCLTest

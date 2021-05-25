//
// Copyright 2019-2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

// FIXME: cannot be run on x86 the tests below use vpusmm allocator and requires vpusmm driver instaled
// can be enabled with other allocator
// [Track number: S#28136]
#ifdef __arm__

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <ie_blob.h>
#include <unistd.h>

#include <cstring>
#include <fstream>

#include "kmb_vpusmm_allocator.h"

using namespace vpu::KmbPlugin;

class kmbAllocatorUnitTests : public ::testing::Test {
protected:
    void SetUp() override {
        std::ifstream modulesLoaded("/proc/modules");
        std::string line;
        while (std::getline(modulesLoaded, line)) {
            if (line.find("vpusmm_driver") != std::string::npos) {
                isVPUSMMDriverFound = true;
                break;
            }
        }
    }

    bool isVPUSMMDriverFound = false;
};

TEST_F(kmbAllocatorUnitTests, canFreeMemory) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    KmbVpusmmAllocator allocator;

    size_t size = 10;
    auto data = allocator.alloc(size);
    ASSERT_NE(data, nullptr);

    ASSERT_TRUE(allocator.free(data));
}

TEST_F(kmbAllocatorUnitTests, canWriteAndReadAllocatedMemory) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    KmbVpusmmAllocator allocator;

    size_t size = 10;
    void* data = allocator.alloc(size);
    ASSERT_NE(data, nullptr);

    // this memory should be accessible for manipulations
    char* lockedData = reinterpret_cast<char*>(allocator.lock(data, InferenceEngine::LockOp::LOCK_FOR_WRITE));

    const int MAGIC_NUMBER = 0x13;
    std::memset(lockedData, MAGIC_NUMBER, size);

    std::vector<char> actual(size);
    std::memcpy(actual.data(), data, size);
    ASSERT_TRUE(static_cast<size_t>(std::count(actual.begin(), actual.end(), MAGIC_NUMBER)) == size);
}

TEST_F(kmbAllocatorUnitTests, cannotFreeInvalidAddressMemory) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    KmbVpusmmAllocator allocator;

    auto data = nullptr;

    ASSERT_FALSE(allocator.free(data));
}

TEST_F(kmbAllocatorUnitTests, cannotDoDoubleFree) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    KmbVpusmmAllocator allocator;

    size_t size = 10;
    auto data = allocator.alloc(size);
    ASSERT_NE(data, nullptr);

    ASSERT_TRUE(allocator.free(data));
    ASSERT_FALSE(allocator.free(data));
}

TEST_F(kmbAllocatorUnitTests, canCreateBlobBasedOnAllocator) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    const std::shared_ptr<InferenceEngine::IAllocator> customAllocator(new KmbVpusmmAllocator());

    const InferenceEngine::TensorDesc tensorDesc(
        InferenceEngine::Precision::U8, {1, 1, 1, 1}, InferenceEngine::Layout::NCHW);
    auto blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc, customAllocator);

    ASSERT_NE(blob, nullptr);
}

TEST_F(kmbAllocatorUnitTests, canWriteToBlobMemory) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    const std::shared_ptr<InferenceEngine::IAllocator> customAllocator(new KmbVpusmmAllocator());

    const InferenceEngine::TensorDesc tensorDesc(
        InferenceEngine::Precision::U8, {1, 1, 1, 1}, InferenceEngine::Layout::NCHW);
    auto blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc, customAllocator);

    ASSERT_NE(blob, nullptr);

    blob->allocate();
    blob->buffer().as<uint8_t*>()[0] = 0;
}

class kmbAllocatorDifferentSizeUnitTests : public kmbAllocatorUnitTests, public ::testing::WithParamInterface<bool> {
public:
    struct PrintToStringParamName {
        template <class ParamType>
        std::string operator()(testing::TestParamInfo<ParamType> const& info) const {
            auto isAlignedAllocation = static_cast<bool>(info.param);
            return isAlignedAllocation ? "isAlignedAllocation=true" : "isAlignedAllocation=false";
        }
    };

protected:
    long pageSize = getpagesize();
};

TEST_P(kmbAllocatorDifferentSizeUnitTests, canAllocate) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }
    auto isAlignedAllocation = GetParam();

    KmbVpusmmAllocator allocator;

    size_t alignedSize = 2 * pageSize;
    size_t size = alignedSize;
    if (!isAlignedAllocation) {
        size -= 1;
    }
    ASSERT_NE(allocator.alloc(size), nullptr);
}

INSTANTIATE_TEST_CASE_P(unit, kmbAllocatorDifferentSizeUnitTests, ::testing::Values(true, false),
    kmbAllocatorDifferentSizeUnitTests::PrintToStringParamName());

TEST_F(kmbAllocatorUnitTests, checkValidPtrOnVpusmm) {
#if !defined(__arm__) && !defined(__aarch64__)
    SKIP();
#endif

    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    KmbVpusmmAllocator allocator;

    size_t size = 10;
    auto data = allocator.alloc(size);

    ASSERT_NE(data, nullptr);
    ASSERT_TRUE(allocator.isValidPtr(data));
    ASSERT_TRUE(allocator.free(data));
}
#endif  //  __arm__

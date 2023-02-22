//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"

#include "kmb_test_mvn_def.hpp"
#include "kmb_test_power_def.hpp"
#include "kmb_test_reshape_def.hpp"
#include "kmb_test_softmax_def.hpp"

#include <vpux/vpux_plugin_config.hpp>

#include <vpux/utils/core/format.hpp>

#include <common_test_utils/test_common.hpp>

#include <gtest/gtest.h>
#include <yolo_helpers.hpp>

using namespace InferenceEngine;

#define PRETTY_PARAM(name, type)                                                  \
    class name {                                                                  \
    public:                                                                       \
        typedef type param_type;                                                  \
        name(param_type arg = param_type()): val_(arg) {                          \
        }                                                                         \
        operator param_type() const {                                             \
            return val_;                                                          \
        }                                                                         \
                                                                                  \
    private:                                                                      \
        param_type val_;                                                          \
    };                                                                            \
    static inline void PrintTo(name param, ::std::ostream* os) {                  \
        *os << #name ": " << ::testing::PrintToString((name::param_type)(param)); \
    }

// #define RUN_SKIPPED_TESTS

#ifdef RUN_SKIPPED_TESTS
#define SKIP_INFER_ON(...)
#define SKIP_INFER(_reason_)
#define SKIP_ON(...)
#else

#define SKIP_ON1(_backend0_, _reason_)                                            \
    do {                                                                          \
        std::set<std::string> backends({_backend0_});                             \
        if (backends.count(BACKEND_NAME)) {                                       \
            GTEST_SKIP() << "Skip on " << BACKEND_NAME << " due to " << _reason_; \
        }                                                                         \
    } while (false)

#define SKIP_ON2(_backend0_, _backend1_, _reason_)                                \
    do {                                                                          \
        std::set<std::string> backends({_backend0_, _backend1_});                 \
        if (backends.count(BACKEND_NAME)) {                                       \
            GTEST_SKIP() << "Skip on " << BACKEND_NAME << " due to " << _reason_; \
        }                                                                         \
    } while (false)

#define SKIP_ON3(_backend0_, _backend1_, _backend2_, _reason_)                    \
    do {                                                                          \
        std::set<std::string> backends({_backend0_, _backend1_, _backend2_});     \
        if (backends.count(BACKEND_NAME)) {                                       \
            GTEST_SKIP() << "Skip on " << BACKEND_NAME << " due to " << _reason_; \
        }                                                                         \
    } while (false)

#define SKIP_INFER_ON1(_backend0_, _reason_)                                            \
    do {                                                                                \
        std::set<std::string> backends({_backend0_});                                   \
        if (KmbTestBase::RUN_INFER && backends.count(BACKEND_NAME)) {                   \
            GTEST_SKIP() << "Skip infer on " << BACKEND_NAME << " due to " << _reason_; \
        }                                                                               \
    } while (false)

#define SKIP_INFER_ON2(_backend0_, _backend1_, _reason_)                                \
    do {                                                                                \
        std::set<std::string> backends({_backend0_, _backend1_});                       \
        if (KmbTestBase::RUN_INFER && backends.count(BACKEND_NAME)) {                   \
            GTEST_SKIP() << "Skip infer on " << BACKEND_NAME << " due to " << _reason_; \
        }                                                                               \
    } while (false)

#define SKIP_INFER_ON3(_backend0_, _backend1_, _backend2_, _reason_)                    \
    do {                                                                                \
        std::set<std::string> backends({_backend0_, _backend1_, _backend2_});           \
        if (KmbTestBase::RUN_INFER && backends.count(BACKEND_NAME)) {                   \
            GTEST_SKIP() << "Skip infer on " << BACKEND_NAME << " due to " << _reason_; \
        }                                                                               \
    } while (false)

#define SKIP_INFER(_reason_)                                  \
    do {                                                      \
        if (KmbTestBase::RUN_INFER) {                         \
            GTEST_SKIP() << "Skip infer due to " << _reason_; \
        }                                                     \
    } while (false)

#endif

#define GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME
#ifndef RUN_SKIPPED_TESTS
#define SKIP_INFER_ON(...) GET_MACRO(__VA_ARGS__, SKIP_INFER_ON3, SKIP_INFER_ON2, SKIP_INFER_ON1)(__VA_ARGS__)
#define SKIP_ON(...) GET_MACRO(__VA_ARGS__, SKIP_ON3, SKIP_ON2, SKIP_ON1)(__VA_ARGS__)
#endif

//
// Kmb test parameters accessors
//

#define PARAMETER(Type, Name)                        \
private:                                             \
    Type _##Name{};                                  \
                                                     \
public:                                              \
    auto Name(const Type& value)->decltype(*this)& { \
        this->_##Name = value;                       \
        return *this;                                \
    }                                                \
    const Type& Name() const {                       \
        return this->_##Name;                        \
    }

#define LAYER_PARAMETER(Type, Name)                  \
public:                                              \
    auto Name(const Type& value)->decltype(*this)& { \
        params.Name = value;                         \
        return *this;                                \
    }                                                \
    const Type& Name() const {                       \
        return params.Name;                          \
    }

enum ImageFormat { RGB, BGR };

//
// KmbTestBase
//

class KmbTestBase : public CommonTestUtils::TestsCommon {
public:
    using BlobGenerator = std::function<Blob::Ptr(const TensorDesc& desc)>;
    using CompileConfig = std::map<std::string, std::string>;
    class import_error : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

public:
    static const std::string DEVICE_NAME;
    static const std::string REF_DEVICE_NAME;
    static const bool RUN_COMPILER;
    static const bool RUN_REF_CODE;
    // RUN_INFER was made non-const to be able
    // to disable inference but keep compilation for tests
    bool RUN_INFER;
    // BACKEND_NAME was made non-const due to
    // necessity to use InferenceEngine core object
    std::string BACKEND_NAME;
    static const std::string DUMP_PATH;
    static const bool EXPORT_NETWORK;
    static const bool RAW_EXPORT;
    static const bool GENERATE_BLOBS;
    static const bool EXPORT_BLOBS;
    static const std::string LOG_LEVEL;
    static const bool PRINT_PERF_COUNTERS;

public:
    void registerBlobGenerator(const std::string& blobName, const TensorDesc& desc, const BlobGenerator& generator) {
        blobGenerators[blobName] = {desc, generator};
    }

    Blob::Ptr getBlobByName(const std::string& blobName);

    BlobMap getInputs(const ExecutableNetwork& testNet);

protected:
    void SetUp() override;
    void TearDown() override;

protected:
    ExecutableNetwork getExecNetwork(const std::function<CNNNetwork()>& netCreator,
                                     const std::function<CompileConfig()>& configCreator,
                                     const bool forceCompilation = false);

    void compareWithReference(const BlobMap& actualOutputs, const BlobMap& refOutputs, const float tolerance,
                              const CompareMethod method = CompareMethod::Absolute);

    void compareOutputs(const Blob::Ptr& refOutput, const Blob::Ptr& actualOutput, const float tolerance,
                        const CompareMethod method = CompareMethod::Absolute);

    void checkWithOutputsInfo(const BlobMap& actualOutputs, const std::vector<DataPtr>& outputsInfo);

protected:
    void exportNetwork(ExecutableNetwork& exeNet);

    ExecutableNetwork importNetwork(const std::map<std::string, std::string>& importConfig = {});

    void dumpBlob(const std::string& blobName, const Blob::Ptr& blob);

    void dumpBlobs(const BlobMap& blobs);

    Blob::Ptr importBlob(const std::string& name, const TensorDesc& desc);

    BlobMap runInfer(ExecutableNetwork& exeNet, const BlobMap& inputs, bool printTime);

protected:
    std::default_random_engine rd;
    std::shared_ptr<Core> core;
    std::string dumpBaseName;
    std::unordered_map<std::string, Blob::Ptr> blobs;
    std::unordered_map<std::string, std::pair<TensorDesc, BlobGenerator>> blobGenerators;
    bool enable_CPU_lpt = false;
    bool skipInfer = false;
    std::string skipMessage;
};

//
// KmbLayerTestBase
//

class KmbLayerTestBase : public KmbTestBase {
    using NetworkBuilder = std::function<void(TestNetwork& testNet)>;

public:
    void runTest(const NetworkBuilder& builder, const float tolerance,
                 const CompareMethod method = CompareMethod::Absolute);

protected:
    ExecutableNetwork getExecNetwork(TestNetwork& testNet);

    BlobMap getRefOutputs(TestNetwork& testNet, const BlobMap& inputs);
};

//
// TestNetworkDesc
//

class TestNetworkDesc final {
public:
    explicit TestNetworkDesc(std::string irFileName): _irFileName(std::move(irFileName)) {
    }
    explicit TestNetworkDesc(std::string irFileName, bool isExperimental)
            : _irFileName(std::move(irFileName)), _isExperimental(isExperimental) {
    }

    TestNetworkDesc& setUserInputPrecision(const std::string& name, const Precision& precision) {
        _inputPrecisions[name] = precision;
        return *this;
    }

    TestNetworkDesc& setUserInputLayout(const std::string& name, const Layout& layout) {
        _inputLayouts[name] = layout;
        return *this;
    }

    TestNetworkDesc& setUserOutputPrecision(const std::string& name, const Precision& precision) {
        _outputPrecisions[name] = precision;
        return *this;
    }

    TestNetworkDesc& setUserOutputLayout(const std::string& name, const Layout& layout) {
        _outputLayouts[name] = layout;
        return *this;
    }

    TestNetworkDesc& setCompileConfig(const std::map<std::string, std::string>& compileConfig) {
        _compileConfig = compileConfig;
        return *this;
    }

    TestNetworkDesc& enableLPTRefMode() {
        _useLPTRefMode = true;
        return *this;
    }

    const std::string& irFileName() const {
        return _irFileName;
    }

    void fillUserInputInfo(InputsDataMap& info) const;
    void fillUserOutputInfo(OutputsDataMap& info) const;

    const std::unordered_map<std::string, Precision>& outputPrecisions() const {
        return _outputPrecisions;
    }

    const std::unordered_map<std::string, Layout>& outputLayouts() const {
        return _outputLayouts;
    }

    const std::map<std::string, std::string>& compileConfig() const {
        return _compileConfig;
    }

    bool isExperimental() const {
        return _isExperimental;
    }

    bool isCompilationForced() const {
        return _forceCompilation;
    }

    bool isLPTRefModeEnabled() const {
        return _useLPTRefMode;
    }

    TestNetworkDesc& enableForcedCompilation() {
        _forceCompilation = true;
        return *this;
    }

    TestNetworkDesc& disableForcedCompilation() {
        _forceCompilation = false;
        return *this;
    }

private:
    std::string _irFileName;

    std::unordered_map<std::string, Precision> _inputPrecisions;
    std::unordered_map<std::string, Layout> _inputLayouts;

    std::unordered_map<std::string, Precision> _outputPrecisions;
    std::unordered_map<std::string, Layout> _outputLayouts;

    std::map<std::string, std::string> _compileConfig;
    bool _forceCompilation = false;
    const bool _isExperimental = false;
    bool _useLPTRefMode = false;
};

//
// TestImageDesc
//

class TestImageDesc final {
public:
    TestImageDesc(const char* imageFileName, ImageFormat imageFormat = ImageFormat::BGR)
            : _imageFileName(imageFileName), _imageFormat(imageFormat) {
    }
    TestImageDesc(std::string imageFileName, ImageFormat imageFormat = ImageFormat::BGR)
            : _imageFileName(std::move(imageFileName)), _imageFormat(imageFormat) {
    }

    const std::string& imageFileName() const {
        return _imageFileName;
    }

    bool isBGR() const {
        return (_imageFormat == ImageFormat::BGR);
    }

private:
    std::string _imageFileName;
    ImageFormat _imageFormat = ImageFormat::BGR;
};

//
// TestBinFileDesc
//

class TestBinFileDesc final {
public:
    TestBinFileDesc(const char* fileName, std::vector<size_t> shape, Precision precision)
            : _fileName(fileName), _shape(std::move(shape)), _precision(precision) {
    }
    TestBinFileDesc(std::string fileName, std::vector<size_t> shape, Precision precision)
            : _fileName(std::move(fileName)), _shape(std::move(shape)), _precision(precision) {
    }

    const std::string& fileName() const {
        return _fileName;
    }

    const std::vector<size_t>& getShape() const {
        return _shape;
    }

    Precision getPrecision() const {
        return _precision;
    }

    size_t getSize() const {
        size_t totalSize = _precision.size();
        for (auto dim : _shape)
            totalSize *= dim;

        return totalSize;
    }

private:
    std::string _fileName;
    std::vector<size_t> _shape;  // shape {N, C, H, W}
    Precision _precision;
};

//
// KmbNetworkTestBase
//

class KmbNetworkTestBase : public KmbTestBase {
protected:
    using CheckCallback = std::function<void(const BlobMap& actualBlob, const BlobMap& refBlob,
                                             const ConstInputsDataMap& inputsDesc)>;

    using InitIntputCallback = std::function<void(const ConstInputsDataMap& inputs)>;

protected:
    static std::string getTestDataPath();
    static std::string getTestModelsPath();

    static Blob::Ptr loadImage(const TestImageDesc& image, size_t channels, size_t height, size_t width);
    static Blob::Ptr loadBinFile(const TestBinFileDesc& binFile, size_t channels, size_t height, size_t width);

    void registerSingleImage(const TestImageDesc& image, const std::string& inputName, const TensorDesc inputDesc);

    void registerSingleBinFile(const TestBinFileDesc& file, const std::string& inputName, const TensorDesc inputDesc);

    CNNNetwork readNetwork(const TestNetworkDesc& netDesc, bool fillUserInfo);

    ExecutableNetwork getExecNetwork(const TestNetworkDesc& netDesc);

    BlobMap calcRefOutput(const TestNetworkDesc& netDesc, const BlobMap& inputs, const bool& enableLPTRef = false);

    void runTest(const TestNetworkDesc& netDesc, const InitIntputCallback& image, const CheckCallback& checkCallback);

    void checkLayouts(const BlobMap& actualOutputs, const std::unordered_map<std::string, Layout>& layouts) const;

    void checkPrecisions(const BlobMap& actualOutputs, const std::unordered_map<std::string, Precision>&) const;
};

//
// KmbClassifyNetworkTest
//

class KmbClassifyNetworkTest : public KmbNetworkTestBase {
public:
    void runTest(const TestNetworkDesc& netDesc, const TestImageDesc& image, const size_t topK,
                 const float probTolerance);

    void runTest(const TestNetworkDesc& netDesc, const TestBinFileDesc& file, const size_t topK,
                 const float probTolerance);

protected:
    static std::vector<std::pair<int, float>> parseOutput(const Blob::Ptr& blob);

private:
    void checkCallbackHelper(const BlobMap& actualBlobs, const BlobMap& refBlobs, const size_t topK,
                             const float probTolerance);
};

//
// KmbDetectionNetworkTest
//

class KmbDetectionNetworkTest : public KmbNetworkTestBase {
public:
    void runTest(const TestNetworkDesc& netDesc, const TestImageDesc& image, const float confThresh,
                 const float boxTolerance, const float probTolerance);

    void runTest(const TestNetworkDesc& netDesc, const float confThresh, const float boxTolerance,
                 const float probTolerance);

protected:
    static std::vector<utils::BoundingBox> parseOutput(const Blob::Ptr& blob, const size_t imgWidth,
                                                       const size_t imgHeight, const float confThresh);

    void checkBBoxOutputs(std::vector<utils::BoundingBox>& actual, std::vector<utils::BoundingBox>& ref,
                          const size_t imgWidth, const size_t imgHeight, const float boxTolerance,
                          const float probTolerance);
};

class KmbYoloV2NetworkTest : public KmbDetectionNetworkTest {
public:
    void runTest(const TestNetworkDesc& netDesc, const TestImageDesc& image, const float confThresh,
                 const float boxTolerance, const float probTolerance, const bool isTiny);
};

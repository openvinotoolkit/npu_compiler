//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <type_traits>

#include "vcl_api.hpp"

std::unique_ptr<char[]> readFile(const char* fileName, size_t& size) {
    size = 0;
    auto fileCloser = [](FILE* filePtr) {
        if (filePtr != nullptr) {
            (void)std::fclose(filePtr);
        }
    };
    std::unique_ptr<FILE, decltype(fileCloser)> file(fopen(fileName, "rb"), fileCloser);
    if (!file) {
        perror("Can't open blob file");
        return nullptr;
    }

    if (fseek(file.get(), 0L, SEEK_END) != 0) {
        fprintf(stderr, "fseek failed for %s", fileName);
        return nullptr;
    }
    long fileSize = ftell(file.get());
    if (fileSize < 0) {
        printf("Ftell method returns failure.");
        return nullptr;
    }
    size_t unsignedFileSize = static_cast<size_t>(fileSize);
    if (unsignedFileSize == 0) {
        fprintf(stderr, "Binary buffer is empty");
        return nullptr;
    }
    if (fseek(file.get(), 0L, SEEK_SET) != 0) {
        fprintf(stderr, "fseek failed for %s", fileName);
        return nullptr;
    }

    std::unique_ptr<char[]> binaryBuffer(new (std::nothrow) char[unsignedFileSize]);
    if (!binaryBuffer) {
        fprintf(stderr, "Can't allocate %zu bytes to read %s.", unsignedFileSize, fileName);
        return nullptr;
    }

    const size_t bytesRead = fread(binaryBuffer.get(), sizeof(char), unsignedFileSize, file.get());
    if (bytesRead != unsignedFileSize) {
        fprintf(stderr, "Failed to read complete file %s", fileName);
        return nullptr;
    }

    size = bytesRead;
    return binaryBuffer;
}

int runProfilingFlow(const std::unique_ptr<char[]>& blobBuffer, size_t blobSize,
                     const std::unique_ptr<char[]>& profBuffer, size_t profSize) {
    struct ProfilingHandleDeleter {
        void operator()(vcl_profiling_handle_t handle) const noexcept {
            if (handle == nullptr) {
                return;
            }
            try {
                (void)VCLTest::vclProfilingDestroy(handle);
            } catch (...) {
                std::fprintf(stderr, "Exception occurred while destroying profiling handle\n");
            }
        }
    };
    using ProfilingHandlePtr = std::unique_ptr<std::remove_pointer_t<vcl_profiling_handle_t>, ProfilingHandleDeleter>;
    ProfilingHandlePtr profHandle;

    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_profiling_input_t profilingApiInput = {};
    profilingApiInput.blobData = reinterpret_cast<uint8_t*>(blobBuffer.get());
    profilingApiInput.blobSize = blobSize;
    profilingApiInput.profData = reinterpret_cast<uint8_t*>(profBuffer.get());
    profilingApiInput.profSize = profSize;
    vcl_profiling_handle_t rawProfHandle = nullptr;
    ret = VCLTest::vclProfilingCreate(&profilingApiInput, &rawProfHandle, nullptr);
    if (ret != VCL_RESULT_SUCCESS) {
        return EXIT_FAILURE;
    }
    profHandle.reset(rawProfHandle);

    vcl_profiling_properties_t profProperties = {};
    ret = VCLTest::vclProfilingGetProperties(profHandle.get(), &profProperties);
    if (ret != VCL_RESULT_SUCCESS) {
        return EXIT_FAILURE;
    }
    printf("Using profiling version %hu.%hu\n", profProperties.version.major, profProperties.version.minor);

    vcl_profiling_output_t profOutput = {};
    ret = VCLTest::vclGetDecodedProfilingBuffer(profHandle.get(), VCL_PROFILING_LAYER_LEVEL, &profOutput);
    if (ret != VCL_RESULT_SUCCESS || profOutput.data == nullptr) {
        return EXIT_FAILURE;
    }

    profOutput = {};
    ret = VCLTest::vclGetDecodedProfilingBuffer(profHandle.get(), VCL_PROFILING_TASK_LEVEL, &profOutput);
    if (ret != VCL_RESULT_SUCCESS || profOutput.data == nullptr) {
        return EXIT_FAILURE;
    }

    profOutput = {};
    ret = VCLTest::vclGetDecodedProfilingBuffer(profHandle.get(), VCL_PROFILING_RAW, &profOutput);
    if (ret != VCL_RESULT_SUCCESS || profOutput.data == nullptr) {
        return EXIT_FAILURE;
    }

    printf("Test passed. Profiling API works! Great success!\n");
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    int result = EXIT_SUCCESS;
    std::unique_ptr<char[]> blobBuffer;
    std::unique_ptr<char[]> profBuffer;

    if (argc != 3) {
        printf("usage:\n"
               "\tprofilingTest network.blob profiling_output.bin\n"
               "where\n"
               "\tnetwork.blob - blob with profiling enabled ('PERF_COUNT YES' parameter in the compiler)"
               "\tprofiling_output.bin - raw profiling output acquired from InferenceManagerDemo according to "
               "guides/how_to_use_profiling.md\n");
        return EXIT_FAILURE;
    }

    const char* blobFileName = argv[1];
    const char* profFileName = argv[2];

    size_t blobSize = 0;
    blobBuffer = readFile(blobFileName, blobSize);
    if (!blobBuffer) {
        return EXIT_FAILURE;
    }

    size_t profSize = 0;
    profBuffer = readFile(profFileName, profSize);
    if (!profBuffer) {
        return EXIT_FAILURE;
    }

    try {
        printf("VCL step: Load VCL library.\n");
        (void)VCLTest::VCLApi::getInstance();
        result = runProfilingFlow(blobBuffer, blobSize, profBuffer, profSize);
        if (result != EXIT_SUCCESS) {
            return result;
        }
    } catch (const std::exception& ex) {
        fprintf(stderr, "profilingTest failed with exception: %s\n", ex.what());
        return EXIT_FAILURE;
    } catch (...) {
        fprintf(stderr, "profilingTest failed with unknown exception\n");
        return EXIT_FAILURE;
    }
    return result;
}

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npu_driver_compiler.h"

void getLastError(vcl_log_handle_t logHandle) {
    /// Get latest error info
    if (logHandle != NULL) {
        size_t logSize = 0;
        vcl_result_t logRet = vclLogHandleGetString(logHandle, &logSize, NULL);
        if (logRet != VCL_RESULT_SUCCESS) {
            printf("Failed to get size of error message\n");
        } else if (logSize == 0) {
            printf("No error during compilation\n");
        } else {
            char* log = (char*)malloc(logSize);
            if (!log) {
                printf("Failed to alloc memory to store error log!");
                return;
            }
            logRet = vclLogHandleGetString(logHandle, &logSize, log);
            if (logRet != VCL_RESULT_SUCCESS) {
                printf("Failed to get content of error message\n");
            } else {
                printf("The last error: %s\n", log);
            }
            free(log);
        }
    }
}

vcl_result_t testCompiler(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        printf("usage:\n\tcompilerTest net.xml weight.bin output.net\n");
        printf("usage:\n\tcompilerTest net.xml weight.bin output.net $configFile\n");
        return -1;
    }

    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_version_info_t compilerVersion;
    vcl_version_info_t profilingVersion;
    ret = vclGetVersion(&compilerVersion, &profilingVersion);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to get version! Result:%x\n", ret);
        return ret;
    } else {
        printf("\n############################################\n\n");
        printf("Library VCL API version info:\n");
        printf("Compiler version:%d.%d\n", compilerVersion.major, compilerVersion.minor);
        printf("Profiling version:%d.%d\n", profilingVersion.major, profilingVersion.minor);
        printf("\n############################################\n\n");
    }

    /// Control if we save error log or output to terminal
    char* saveErrorLog = getenv("VCL_SAVE_ERROR");
    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_TRACE;
    vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t), 0x643e, 3, 5};
    vcl_compiler_handle_t compiler = NULL;
    vcl_log_handle_t logHandle = NULL;
    if (saveErrorLog == NULL) {
        ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, NULL);
    } else {
        ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, &logHandle);
    }
    if (ret) {
        printf("Failed to create compiler! Result:%x\n", ret);
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret) {
        printf("Failed to query compiler props! Result: %x\n", ret);
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        printf("\n############################################\n\n");
        printf("Current compiler info:\n");
        printf("ID: %s\n", compilerProp.id);
        printf("Version:%d.%d\n", compilerProp.version.major, compilerProp.version.minor);
        printf("\tSupported opsets:%d\n", compilerProp.supportedOpsets);
        printf("\n############################################\n\n");
    }

    /// Read buffer, add net.xml
    size_t bytesRead = 0;
    char* netName = argv[1];
    FILE* fpN = fopen(netName, "rb");
    if (!fpN) {
        printf("Cannot open file %s\n", netName);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    fseek(fpN, 0, SEEK_END);
    long fileXmlSize = ftell(fpN);
    if (fileXmlSize < 0) {
        printf("Ftell method returns failure.");
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    uint64_t xmlSize = (uint64_t)fileXmlSize;
    fseek(fpN, 0, SEEK_SET);

    /// Read weights size, add weight.bin
    char* weightName = argv[2];
    FILE* fpW = fopen(weightName, "rb");
    if (!fpW) {
        printf("Cannot open file %s\n", weightName);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    fseek(fpW, 0, SEEK_END);
    long fileWeightsSize = ftell(fpW);
    if (fileWeightsSize < 0) {
        printf("Ftell method returns failure.");
        fclose(fpN);
        fclose(fpW);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    uint64_t weightsSize = (uint64_t)fileWeightsSize;

    /// Init modelIR
    vcl_version_info_t version = compilerProp.version;
    uint32_t numberOfInputData = 2;
    uint64_t modelIRSize =
            sizeof(version) + sizeof(numberOfInputData) + sizeof(xmlSize) + xmlSize + sizeof(weightsSize) + weightsSize;
    uint8_t* modelIR = (uint8_t*)malloc(modelIRSize);
    if (!modelIR) {
        printf("Failed to alloc memory for IR!\n");
        fclose(fpW);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }
    uint64_t offset = 0;
    memcpy(modelIR, &version, sizeof(version));
    offset += sizeof(version);
    memcpy(modelIR + offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    memcpy(modelIR + offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    uint8_t* xmlData = modelIR + offset;
    bytesRead = fread(xmlData, 1, xmlSize, fpN);
    if ((uint64_t)bytesRead != xmlSize) {
        printf("Short read on network buffer!!!\n");
        free(modelIR);
        fclose(fpW);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    int cret = fclose(fpN);
    if (cret) {
        printf("Failed to close %s. Result:%d\n", netName, cret);
        free(modelIR);
        fclose(fpW);
        vclCompilerDestroy(compiler);
        return cret;
    }

    offset += xmlSize;
    memcpy(modelIR + offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    uint8_t* weights = NULL;
    if (weightsSize != 0) {
        weights = modelIR + offset;
        fseek(fpW, 0, SEEK_SET);
        bytesRead = fread(weights, 1, weightsSize, fpW);
        if ((uint64_t)bytesRead != weightsSize) {
            printf("Short read on weights file!!!\n");
            free(modelIR);
            fclose(fpW);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
    }
    cret = fclose(fpW);
    if (cret) {
        printf("Failed to close %s. Result:%d\n", weightName, cret);
        free(modelIR);
        vclCompilerDestroy(compiler);
        return cret;
    }

    /// The options are for googlenet-v1
    char defaultOptions[] =
            "--inputs_precisions=\"input:U8\" --inputs_layouts=\"input:NCHW\" "
            "--outputs_precisions=\"InceptionV1/Logits/Predictions/Softmax:FP32\" "
            "--outputs_layouts=\"InceptionV1/Logits/Predictions/Softmax:NC\" --config LOG_LEVEL=\"LOG_INFO\"";
    char* newOptions = NULL;
    uint64_t configSize = 0;
    if (argc == 5) {
        char* configFile = argv[4];
        FILE* fpC = fopen(configFile, "rb");
        if (!fpC) {
            printf("Cannot open file %s\n", configFile);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
        fseek(fpC, 0, SEEK_END);
        long fileConfigSize = ftell(fpC);
        if (fileConfigSize < 0) {
            printf("Ftell method returns failure.");
            fclose(fpC);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
        configSize = (uint64_t)fileConfigSize;
        fseek(fpC, 0, SEEK_SET);
        if (configSize == 0) {
            printf("The config file %s is empty\n", configFile);
            fclose(fpC);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        } else {
            newOptions = (char*)malloc(configSize);
            if (!newOptions) {
                printf("Failed to alloc memory for options\n");
                fclose(fpC);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return VCL_RESULT_ERROR_OUT_OF_MEMORY;
            }
            bytesRead = fread(newOptions, 1, configSize, fpC);
            if ((uint32_t)bytesRead != configSize) {
                printf("Short read on config file buffer!!!\n");
                free(newOptions);
                fclose(fpC);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return VCL_RESULT_ERROR_IO;
            }
            cret = fclose(fpC);
            if (cret) {
                printf("Failed to close %s. Result:%d\n", configFile, cret);
                free(newOptions);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return cret;
            }
        }
    }

    /// Test query network, create query handle first
    vcl_query_handle_t query = NULL;
    if (newOptions) {
        vcl_query_desc_t desc = {modelIR, modelIRSize, newOptions, configSize};
        ret = vclQueryNetworkCreate(compiler, desc, &query);
    } else {
        vcl_query_desc_t desc = {modelIR, modelIRSize, defaultOptions, sizeof(defaultOptions)};
        ret = vclQueryNetworkCreate(compiler, desc, &query);
    }
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(logHandle);
        printf("Failed to query network! Result:%x\n", ret);
        free(newOptions);
        free(modelIR);
        vclCompilerDestroy(compiler);
        return ret;
    }
    uint8_t* layerRawData = NULL;
    uint64_t layerSize = 0;
    /// First time calling vclQueryNetwork, layerRawData is nullptr, get layerSize
    ret = vclQueryNetwork(query, layerRawData, &layerSize);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to get size of query result! Result:%x\n", ret);
        vclQueryNetworkDestroy(query);
        vclCompilerDestroy(compiler);
        return ret;
    }
    /// layerRawData should be allocated with layerSize
    layerRawData = malloc(layerSize);
    if (layerRawData == NULL) {
        printf("Failed to malloc memory to store layer info!\n");
        vclQueryNetworkDestroy(query);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }
    /// Second time calling vclQueryNetwork, copy queryResultString to layerRawData
    ret = vclQueryNetwork(query, layerRawData, &layerSize);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to get data of query result! Result:%x\n", ret);
        free(layerRawData);
        vclQueryNetworkDestroy(query);
        vclCompilerDestroy(compiler);
        return ret;
    }
    /// Print the whole layerRawData
    printf("Print layerRawData as the result string of query: \n%.*s", (int)layerSize, layerRawData);
    printf("\n");
    /// Destroy query network handle
    ret = vclQueryNetworkDestroy(query);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to destroy query handle! Result:%x\n", ret);
        free(layerRawData);
        vclCompilerDestroy(compiler);
        return ret;
    }
    free(layerRawData);
    query = NULL;

    vcl_executable_handle_t executable = NULL;
    if (argc != 5) {
        vcl_executable_desc_t exeDesc = {modelIR, modelIRSize, defaultOptions, sizeof(defaultOptions)};
        ret = vclExecutableCreate(compiler, exeDesc, &executable);
    } else {
        vcl_executable_desc_t exeDesc = {modelIR, modelIRSize, newOptions, configSize};
        ret = vclExecutableCreate(compiler, exeDesc, &executable);
        free(newOptions);
    }
    free(modelIR);
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(logHandle);
        printf("Failed to create executable handle! Result:%x\n", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }
    uint64_t blobSize = 0;
    ret = vclExecutableGetSerializableBlob(executable, NULL, &blobSize);
    if (ret != VCL_RESULT_SUCCESS || blobSize == 0) {
        printf("Failed to get blob size! Result:%x\n", ret);
        vclExecutableDestroy(executable);
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        uint8_t* blob = (uint8_t*)malloc(blobSize);
        if (!blob) {
            printf("Failed to alloc memory for blob!\n");
            vclExecutableDestroy(executable);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_OUT_OF_MEMORY;
        }
        ret = vclExecutableGetSerializableBlob(executable, blob, &blobSize);
        if (ret == VCL_RESULT_SUCCESS) {
            char* blobName = argv[3];
            FILE* fpB = fopen(blobName, "wb");
            if (!fpB) {
                printf("Can not open %s, skip dump!\n", blobName);
            } else {
                uint64_t bytesWrite = fwrite(blob, 1, blobSize, fpB);
                if (bytesWrite != blobSize) {
                    printf("Short write to %s, the file is invalid!\n", blobName);
                }
                int cret = fclose(fpB);
                if (cret) {
                    printf("Failed to close %s. Result:%d\n", blobName, cret);
                } else {
                    printf("The output name:%s\n", blobName);
                }
            }
        }
        free(blob);
    }

    ret = vclExecutableDestroy(executable);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to destroy executable! Result:%x\n", ret);
        ret = vclCompilerDestroy(compiler);
        return ret;
    }
    executable = NULL;

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to destroy compiler! Result:%x\n", ret);
        return ret;
    }
    return ret;
}

uint8_t* allocateBlob(uint64_t size) {
    return (uint8_t*)calloc((size_t)size, sizeof(uint8_t));
}

void deallocateBlob(uint8_t* ptr) {
    free((void*)ptr);
}

vcl_result_t testCompilerAllocator(int argc, char** argv, const char* blobFileName) {
    if (argc != 4 && argc != 5) {
        printf("usage:\n\tcompilerTest net.xml weight.bin output.net\n");
        printf("usage:\n\tcompilerTest net.xml weight.bin output.net $configFile\n");
        return -1;
    }

    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_version_info_t compilerVersion;
    vcl_version_info_t profilingVersion;
    ret = vclGetVersion(&compilerVersion, &profilingVersion);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to get version! Result:%x\n", ret);
        return ret;
    } else {
        printf("\n############################################\n\n");
        printf("Library VCL API version info:\n");
        printf("Compiler version:%d.%d\n", compilerVersion.major, compilerVersion.minor);
        printf("Profiling version:%d.%d\n", profilingVersion.major, profilingVersion.minor);
        printf("\n############################################\n\n");
    }

    /// Control if we save error log or output to terminal
    char* saveErrorLog = getenv("VCL_SAVE_ERROR");
    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_ERROR;
    vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t), 0x643e, 3, 5};
    vcl_compiler_handle_t compiler = NULL;
    vcl_log_handle_t logHandle = NULL;
    if (saveErrorLog == NULL) {
        ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, NULL);
    } else {
        ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, &logHandle);
    }
    if (ret) {
        printf("Failed to create compiler! Result:%x\n", ret);
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret) {
        printf("Failed to query compiler props! Result: %x\n", ret);
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        printf("\n############################################\n\n");
        printf("Current compiler info:\n");
        printf("ID: %s\n", compilerProp.id);
        printf("Version:%d.%d\n", compilerProp.version.major, compilerProp.version.minor);
        printf("\tSupported opsets:%d\n", compilerProp.supportedOpsets);
        printf("\n############################################\n\n");
    }

    /// Read buffer, add net.xml
    size_t bytesRead = 0;
    char* netName = argv[1];
    FILE* fpN = fopen(netName, "rb");
    if (!fpN) {
        printf("Cannot open file %s\n", netName);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    fseek(fpN, 0, SEEK_END);
    long fileXmlSize = ftell(fpN);
    if (fileXmlSize < 0) {
        printf("Ftell method returns failure.");
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    uint64_t xmlSize = (uint64_t)fileXmlSize;
    fseek(fpN, 0, SEEK_SET);

    /// Read weights size, add weight.bin
    char* weightName = argv[2];
    FILE* fpW = fopen(weightName, "rb");
    if (!fpW) {
        printf("Cannot open file %s\n", weightName);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    fseek(fpW, 0, SEEK_END);
    long fileWeightsSize = ftell(fpW);
    if (fileWeightsSize < 0) {
        printf("Ftell method returns failure.");
        fclose(fpN);
        fclose(fpW);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    uint64_t weightsSize = (uint64_t)fileWeightsSize;

    /// Init modelIR
    vcl_version_info_t version = compilerProp.version;
    uint32_t numberOfInputData = 2;
    uint64_t modelIRSize =
            sizeof(version) + sizeof(numberOfInputData) + sizeof(xmlSize) + xmlSize + sizeof(weightsSize) + weightsSize;
    uint8_t* modelIR = (uint8_t*)malloc(modelIRSize);
    if (!modelIR) {
        printf("Failed to alloc memory for IR!\n");
        fclose(fpW);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }
    uint64_t offset = 0;
    memcpy(modelIR, &version, sizeof(version));
    offset += sizeof(version);
    memcpy(modelIR + offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    memcpy(modelIR + offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    uint8_t* xmlData = modelIR + offset;
    bytesRead = fread(xmlData, 1, xmlSize, fpN);
    if ((uint64_t)bytesRead != xmlSize) {
        printf("Short read on network buffer!!!\n");
        free(modelIR);
        fclose(fpW);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    int cret = fclose(fpN);
    if (cret) {
        printf("Failed to close %s. Result:%d\n", netName, cret);
        free(modelIR);
        fclose(fpW);
        vclCompilerDestroy(compiler);
        return cret;
    }

    offset += xmlSize;
    memcpy(modelIR + offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    uint8_t* weights = NULL;
    if (weightsSize != 0) {
        weights = modelIR + offset;
        fseek(fpW, 0, SEEK_SET);
        bytesRead = fread(weights, 1, weightsSize, fpW);
        if ((uint64_t)bytesRead != weightsSize) {
            printf("Short read on weights file!!!\n");
            free(modelIR);
            fclose(fpW);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
    }
    cret = fclose(fpW);
    if (cret) {
        printf("Failed to close %s. Result:%d\n", weightName, cret);
        free(modelIR);
        vclCompilerDestroy(compiler);
        return cret;
    }

    /// The options are for googlenet-v1
    char defaultOptions[] =
            "--inputs_precisions=\"input:U8\" --inputs_layouts=\"input:NCHW\" "
            "--outputs_precisions=\"InceptionV1/Logits/Predictions/Softmax:FP32\" "
            "--outputs_layouts=\"InceptionV1/Logits/Predictions/Softmax:NC\" --config LOG_LEVEL=\"LOG_INFO\"";
    char* newOptions = NULL;
    uint64_t configSize = 0;
    if (argc == 5) {
        char* configFile = argv[4];
        FILE* fpC = fopen(configFile, "rb");
        if (!fpC) {
            printf("Cannot open file %s\n", configFile);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
        fseek(fpC, 0, SEEK_END);
        long fileConfigSize = ftell(fpC);
        if (fileConfigSize < 0) {
            printf("Ftell method returns failure.");
            fclose(fpC);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
        configSize = (uint64_t)fileConfigSize;
        fseek(fpC, 0, SEEK_SET);
        if (configSize == 0) {
            printf("The config file %s is empty\n", configFile);
            fclose(fpC);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        } else {
            newOptions = (char*)malloc(configSize);
            if (!newOptions) {
                printf("Failed to alloc memory for options\n");
                fclose(fpC);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return VCL_RESULT_ERROR_OUT_OF_MEMORY;
            }
            bytesRead = fread(newOptions, 1, configSize, fpC);
            if ((uint32_t)bytesRead != configSize) {
                printf("Short read on config file buffer!!!\n");
                free(newOptions);
                fclose(fpC);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return VCL_RESULT_ERROR_IO;
            }
            cret = fclose(fpC);
            if (cret) {
                printf("Failed to close %s. Result:%d\n", configFile, cret);
                free(newOptions);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return cret;
            }
        }
    }

    /// Test query network, create query handle first
    vcl_query_handle_t query = NULL;
    if (newOptions) {
        vcl_query_desc_t desc = {modelIR, modelIRSize, newOptions, configSize};
        ret = vclQueryNetworkCreate(compiler, desc, &query);
    } else {
        vcl_query_desc_t desc = {modelIR, modelIRSize, defaultOptions, sizeof(defaultOptions)};
        ret = vclQueryNetworkCreate(compiler, desc, &query);
    }
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(logHandle);
        printf("Failed to query network! Result:%x\n", ret);
        free(newOptions);
        free(modelIR);
        vclCompilerDestroy(compiler);
        return ret;
    }
    uint8_t* layerRawData = NULL;
    uint64_t layerSize = 0;
    /// First time calling vclQueryNetwork, layerRawData is nullptr, get layerSize
    ret = vclQueryNetwork(query, layerRawData, &layerSize);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to get size of query result! Result:%x\n", ret);
        vclQueryNetworkDestroy(query);
        vclCompilerDestroy(compiler);
        return ret;
    }
    /// layerRawData should be allocated with layerSize
    layerRawData = malloc(layerSize);
    if (layerRawData == NULL) {
        printf("Failed to malloc memory to store layer info!\n");
        vclQueryNetworkDestroy(query);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }
    /// Second time calling vclQueryNetwork, copy queryResultString to layerRawData
    ret = vclQueryNetwork(query, layerRawData, &layerSize);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to get data of query result! Result:%x\n", ret);
        free(layerRawData);
        vclQueryNetworkDestroy(query);
        vclCompilerDestroy(compiler);
        return ret;
    }
    /// Print the whole layerRawData
    printf("Print layerRawData as the result string of query: \n%.*s", (int)layerSize, layerRawData);
    printf("\n");
    /// Destroy query network handle
    ret = vclQueryNetworkDestroy(query);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to destroy query handle! Result:%x\n", ret);
        free(layerRawData);
        vclCompilerDestroy(compiler);
        return ret;
    }
    free(layerRawData);
    query = NULL;

    vcl_allocator_t allocator;
    allocator.allocate = allocateBlob;
    allocator.deallocate = deallocateBlob;

    uint8_t* blob = NULL;
    uint64_t size = 0;
    if (argc != 5) {
        vcl_executable_desc_t exeDesc = {modelIR, modelIRSize, defaultOptions, sizeof(defaultOptions)};
        ret = vclAllocatedExecutableCreate(compiler, exeDesc, &allocator, &blob, &size);
    } else {
        vcl_executable_desc_t exeDesc = {modelIR, modelIRSize, newOptions, configSize};
        ret = vclAllocatedExecutableCreate(compiler, exeDesc, &allocator, &blob, &size);
        free(newOptions);
    }
    free(modelIR);
    if (ret != VCL_RESULT_SUCCESS || blob == NULL || size == 0) {
        getLastError(logHandle);
        printf("Failed to create executable handle! Result:%x\n", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }

    FILE* fpB = fopen(blobFileName, "wb");
    if (!fpB) {
        printf("Can not open %s, skip dump!\n", blobFileName);
    } else {
        uint64_t bytesWrite = fwrite(blob, 1, size, fpB);
        if (bytesWrite != size) {
            printf("Short write to %s, the file is invalid!\n", blobFileName);
        }
        int cret = fclose(fpB);
        if (cret) {
            printf("Failed to close %s. Result:%d\n", blobFileName, cret);
        } else {
            printf("The output name:%s\n", blobFileName);
        }
    }
    allocator.deallocate(blob);

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to destroy compiler! Result:%x\n", ret);
        return ret;
    }
    return ret;
}

void printErrorMessage(vcl_result_t status) {
    const char* message = NULL;
    switch (status) {
    case VCL_RESULT_ERROR_OUT_OF_MEMORY:
        message = "VCL_RESULT_ERROR_OUT_OF_MEMORY";
        break;
    case VCL_RESULT_ERROR_INVALID_ARGUMENT:
        message = "VCL_RESULT_ERROR_INVALID_ARGUMENT";
        break;
    case VCL_RESULT_ERROR_INVALID_NULL_HANDLE:
        message = "VCL_RESULT_ERROR_INVALID_NULL_HANDLE";
        break;
    case VCL_RESULT_ERROR_IO:
        message = "VCL_RESULT_ERROR_IO";
        break;
    case VCL_RESULT_ERROR_INVALID_IR:
        message = "VCL_RESULT_ERROR_INVALID_IR";
        break;
    case VCL_RESULT_ERROR_UNKNOWN:
        message = "VCL_RESULT_ERROR_UNKNOWN";
        break;
    default:
        message = "UNKNOWN_ERROR";
    }
    fprintf(stderr, "Error message: %s\n", message);
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        printf("usage:\n\tcompilerTest net.xml weight.bin output.net\n");
        printf("usage:\n\tcompilerTest net.xml weight.bin output.net $configFile\n");
        return -1;
    }

    const char* blobFileName = argv[3];
    const size_t blobFileNameSize = strlen(blobFileName);
    const char* allocatedBlobFileNameSuffix = ".allocator";
    const size_t allocatedBlobFileNameSuffixSize = strlen(allocatedBlobFileNameSuffix);
    const size_t allocatedBlobFileNameSize = blobFileNameSize + allocatedBlobFileNameSuffixSize + 1;

    char* allocatedBlobFileName = NULL;
    if ((allocatedBlobFileName = (char*)malloc(allocatedBlobFileNameSize)) == NULL) {
        fprintf(stderr, "failed to allocate memory for allocated blob file name\n");
        return -2;
    }

    allocatedBlobFileName[0] = '\0';
    strcat(allocatedBlobFileName, blobFileName);
    strcat(allocatedBlobFileName, allocatedBlobFileNameSuffix);

    vcl_result_t status = testCompiler(argc, argv);
    if (status != VCL_RESULT_SUCCESS) {
        printErrorMessage(status);
        return (int)status;
    }
    status = testCompilerAllocator(argc, argv, allocatedBlobFileName);
    if (status != VCL_RESULT_SUCCESS) {
        printErrorMessage(status);
        return (int)status;
    }
    return 0;
}

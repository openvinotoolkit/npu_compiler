//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "VPUXCompilerL0.h"

vcl_result_t testCompiler(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        printf("usage:\n\tcompilerTest net.xml weight.bin output.net\n");
        printf("usage:\n\tcompilerTest net.xml weight.bin output.net $configFile\n");
        return -1;
    }

    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_compiler_desc_t compilerDesc = {VCL_PLATFORM_VPU3700, 5};
    vcl_compiler_handle_t compiler = NULL;
    ret = vclCompilerCreate(compilerDesc, &compiler);
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

    // Read buffer, add net.xml
    size_t bytesRead = 0;
    char* netName = argv[1];
    FILE* fpN = fopen(netName, "rb");
    if (!fpN) {
        printf("Cannot open file %s\n", netName);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    fseek(fpN, 0, SEEK_END);
    uint64_t xmlSize = ftell(fpN);
    fseek(fpN, 0, SEEK_SET);

    // Read weights size, add weight.bin
    char* weightName = argv[2];
    FILE* fpW = fopen(weightName, "rb");
    if (!fpW) {
        printf("Cannot open file %s\n", weightName);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    fseek(fpW, 0, SEEK_END);
    uint64_t weightsSize = ftell(fpW);

    // Init modelIR
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

    vcl_executable_handle_t executable = NULL;
    if (argc != 5) {
        // The options are for googlenet-v1
        char options[] =
                "--inputs_precisions=\"input:U8\" --inputs_layouts=\"input:NCHW\" "
                "--outputs_precisions=\"InceptionV1/Logits/Predictions/Softmax:FP32\" "
                "--outputs_layouts=\"InceptionV1/Logits/Predictions/Softmax:NC\" --config LOG_LEVEL=\"LOG_INFO\" "
                "VPUX_COMPILATION_MODE_PARAMS=\"use-user-precision=false propagate-quant-dequant=0\"";
        vcl_executable_desc_t exeDesc = {modelIR, modelIRSize, options, sizeof(options)};
        ret = vclExecutableCreate(compiler, exeDesc, &executable);
    } else {
        char* configFile = argv[4];
        FILE* fpC = fopen(configFile, "rb");
        if (!fpC) {
            printf("Cannot open file %s\n", configFile);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
        fseek(fpC, 0, SEEK_END);
        uint32_t configSize = ftell(fpC);
        fseek(fpC, 0, SEEK_SET);
        if (configSize == 0) {
            printf("The config file %s is empty\n", configFile);
            fclose(fpC);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        } else {
            char* options = (char*)malloc(configSize);
            if (!options) {
                printf("Failed to alloc memory for options\n");
                fclose(fpC);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return VCL_RESULT_ERROR_OUT_OF_MEMORY;
            }
            bytesRead = fread(options, 1, configSize, fpC);
            if ((uint32_t)bytesRead != configSize) {
                printf("Short read on config file buffer!!!\n");
                free(options);
                fclose(fpC);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return VCL_RESULT_ERROR_IO;
            }
            cret = fclose(fpC);
            if (cret) {
                printf("Failed to close %s. Result:%d\n", configFile, cret);
                free(options);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return cret;
            }

            vcl_executable_desc_t exeDesc = {modelIR, modelIRSize, options, configSize};
            ret = vclExecutableCreate(compiler, exeDesc, &executable);
            free(options);
        }
    }
    free(modelIR);
    if (ret != VCL_RESULT_SUCCESS) {
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

int main(int argc, char** argv) {
    testCompiler(argc, argv);
    // Multiple tests
    // testCompiler(lib, argc, argv);
    return 0;
}

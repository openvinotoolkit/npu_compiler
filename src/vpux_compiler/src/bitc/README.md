# Bitcompactor

Standalone build and validation. The application takes in arguments as seen in the running section, based on these it (de)compresses the data and checks it against the reference data. After that a short conclusion is printed out showing how many tests have failed.
## Build

cd to validation/
- ```mkdir build```
- ```cd build```
- ```cmake ..```
- ```make```

You can also activate different build options using the following flags:
- ```cmake -DDEBUG=ON```, enables __BITC__EN_DBG__ code, basically shows some debug information
- ```cmake -DENCODE_PERCENTAGE=ON```, enables __BITC__EN_ENCODE_PERCENTAGE_RATE__ code, shows encode percentage rates
- ```cmake -DPROFILING=ON```, enables __BITC__EN_PROFILING__ code, shows time taken to (de)compress
- ```cmake -DENABLE_WRITE=ON```, enables __BITC__EN_OUT_WRITE__ code, writes to a file the output of BITCLite

TIP: You can also combine these options

## Running

cd to validation/build/
- ```./bitc [config_file_path]```
- **[config_file_path]** - a file with the configuration for the test. Contains the following
    - **arch_type** - architecture type - can be NPU27, NPU4
      - or NPU5
    - **data_type** - data type of data (either u8 or fp16)
    - **sparse_mode_enable** - can be true or false (supported from NPU5)
    - **weight_compress_enable** - can be true or false, false for activation compression (supported from NPU4)
    - **bypass_compression** - can be true or false
    - **mode_fp16_enable** - can be true or false (supported from NPU4)
    - **decompressed_data_path** - path to decompressed data
    - **decompressed_data** - list with the decompressed dataset
        - This can contain ranges like: file[1-10].bin which resolves to file1.bin file2.bin ... file10.bin
    - **compressed_data_path** - path to compressed data
    - **compressed_data** - list with the compressed dataset
        - This can contain ranges like: file[1-10].bin which resolves to file1.bin file2.bin ... file10.bin
    - **bitmap_data_path** - path to bitmap dataset 
      - required when sparse_mode_enabled = true
    - **bitmap_data** - list with the bitmap dataset
      - This can contain ranges like: file[1-10].bin which resolves to file1.bin file2.bin ... file10.bin
      - required when sparse_mode_enabled = true
    - **sparse_block_size** - a number divisible by 16 
      - required when sparse_mode_enabled = true

## Ref-data

Contains reference data and some config files

https://af01p-ir.devtools.intel.com/artifactory/vpu_openvino-ir-local/dl_score_engine/bitcompactor/ref-data/

### AGS roles: 

- Admin rights  -  DevTools - Artifactory - VPU_OpenVINO - AF01P-IR - Project Administrator
- Developers  - DevTools - Artifactory - VPU_OpenVINO - AF01P-IR - Project Developer
- Read only access - DevTools - Artifactory - VPU_OpenVINO - AF01P-IR - Project Viewer

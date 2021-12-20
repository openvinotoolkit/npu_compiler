# Demo Usage

## Linux

### Folder Structure

```
── CiD_Linux_XXXX
   ├── data
   ├── lib
   ├── README.md
   ├── vpux_compiler_l0.h
   ├── compilerTest
   ├── compilerThreadTest
   └── compilerThreadTest2
```

- `data` contains an xml and bin for test.
- `lib` contains compiler module with all dependent dlls.
- `vpux_compiler_l0.h`  is the header file for exported functions.
- `compilerTest` `compilerThreadTest` `compilerThreadTest2` are executables for test.

```
cd CiD_Linux_XXXX
LD_LIBRARY_PATH=./lib/ ./compilerTest xxx.xml xxx.bin output.net
LD_LIBRARY_PATH=./lib/ ./compilerTest xxx.xml xxx.bin output.net FP16 C FP16 C config.file
LD_LIBRARY_PATH=./lib/ ./compilerThreadTest xxx.xml xxx.bin
LD_LIBRARY_PATH=./lib/ ./compilerThreadTest2 xxx.xml xxx.bin
```

`output.net`  is the generated blob.

## Windows

### Folder Structure

```
── CiD_WIN_XXXX
   ├── data
   ├── lib
   ├── pdb
   ├── README.md
   ├── vpux_compiler_l0.h
   ├── compilerTest
   ├── compilerThreadTest
   └── compilerThreadTest2
```

- `data` contains an xml and bin for test. E.g. if you want to test resnet-50, you can get its IR from `$KMB_PLUGIN_HOME/temp/models/src/models/KMB_models/FP16/resnet_50_pytorch/`
- `lib` contains compiler module with all dependent dlls.
- `pdb` contains pdb files for each dll.
- `vpux_compiler_l0.h`  is the header file for exported functions.
- `compilerTest` `compilerThreadTest` `compilerThreadTest2` are executables for test.

### Windows (git bash)

```
cd CiD_WIN_XXXX
PATH=$PATH:./lib/ ./compilerTest.exe xxx.xml xxx.bin output.net
PATH=$PATH:./lib/ ./compilerTest.exe xxx.xml xxx.bin output.net FP16 C FP16 C config.file
PATH=$PATH:./lib/ ./compilerThreadTest xxx.xml xxx.bin
PATH=$PATH:./lib/ ./compilerThreadTest2 xxx.xml xxx.bin
```
### Windows (PowerShell)

```
cd .\CiD_WIN_XXXX\
$Env:Path +=";.\lib"
.\compilerTest.exe xxx.xml xxx.bin output.net
.\compilerTest.exe xxx.xml xxx.bin output.net FP16 C FP16 C config.file
.\compilerThreadTest xxx.xml xxx.bin
.\compilerThreadTest2 xxx.xml xxx.bin
```

`output.net`  is the generated blob.

# Develop Info

### applications.ai.vpu-accelerators.vpux-plugin
The lib is developed based on

- Branch

```
master
```

**Note: This package provides a thin wrapper/API to generate blob.**

The main entrance is `vclCompilerCreate`. Check full API demo - compilerTest | compilerThreadTest | compilerThreadTest2.

- Example:
```
...
vclCompilerCreate
...
vclCompilerGetProperties
...
/* Fill buffer/weights with data read from command line arguments. Will set result blob size. */
...
vclExecutableCreate
...
vclExecutableGetSeriablizableBlob
...
blobSize > 0
blob = (uint8_t*)malloc(blobSize)
vclExecutableGetSeriablizableBlob
...
vclExecutableDestroy
vclCompilerDestroy
...

```


### OpenVINO

- Branch

```
master
```

#! /bin/bash
env_is_set=1

if [ -z ${MV_TOOLS_DIR} ]; then echo "MV_TOOLS_DIR is not set"; env_is_set=0; fi
if [ -z ${MV_TOOLS_VERSION} ]; then echo "MV_TOOLS_VERSION is not set"; env_is_set=0; fi
if [ -z ${KERNEL_DIR} ] 
then 
ABSOLUTE_FILENAME=`readlink -e "$0"`
kernel_dir=`dirname "$ABSOLUTE_FILENAME"`
kernel_dir=`dirname "$kernel_dir"`
echo $kernel_dir
 if [ -f "$kernel_dir/sigmoid_fp16.c" ] 
 then KERNEL_DIR=$kernel_dir
 else
  echo "KERNEL_DIR is not set"
  env_is_set=0
 fi
fi

if [ $env_is_set = 0 ]; then exit 1; fi

rm -f ${KERNEL_DIR}/prebuild/sigmoid_fp16_3010xx.o ${KERNEL_DIR}/prebuild/sigmoid_fp16_3010xx.elf ${KERNEL_DIR}/prebuild/sk.sigmoid_fp16.3010xx.text ${KERNEL_DIR}/prebuild/sk.sigmoid_fp16.3010xx.data

${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile -mcpu=3010xx -O3 \
 -c ${KERNEL_DIR}/sigmoid_fp16.c -o ${KERNEL_DIR}/prebuild/sigmoid_fp16_3010xx.o \
 -I ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION} \
 -I ${KERNEL_DIR}/inc \
 -D CONFIG_TARGET_SOC_3720 -D__shave_nn__

if [ $? -ne 0 ]; then exit $?; fi

${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld \
--script ${KERNEL_DIR}/prebuild/shave_kernel.ld \
-entry sigmoid_fp16 \
--gc-sections \
--strip-debug \
--discard-all \
-zmax-page-size=16 \
${KERNEL_DIR}/prebuild/sigmoid_fp16_3010xx.o \
 -EL ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibm.a \
 --output ${KERNEL_DIR}/prebuild/sigmoid_fp16_3010xx.elf

if [ $? -ne 0 ]; then echo $'\nLinking of sigmoid_fp16_3010.elf failed exit $?\n'; exit $?; fi
${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy -O binary --only-section=.text ${KERNEL_DIR}/prebuild/sigmoid_fp16_3010xx.elf ${KERNEL_DIR}/prebuild/sk.sigmoid_fp16.3010xx.text
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.sigmoid_fp16.3010xx.text failed exit $?\n'; exit $?; fi
${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy -O binary --only-section=.arg.data ${KERNEL_DIR}/prebuild/sigmoid_fp16_3010xx.elf ${KERNEL_DIR}/prebuild/sk.sigmoid_fp16.3010xx.data
if [ $? -ne 0 ]; then echo $'\nExtracting of sk.sigmoid_fp16.3010xx.data failed exit $?\n'; exit $?; fi
rm ${KERNEL_DIR}/prebuild/sigmoid_fp16_3010xx.o
printf "\n ${KERNEL_DIR}/prebuild/sk.sigmoid_fp16.3010xx.text\n ${KERNEL_DIR}/prebuild/sk.sigmoid_fp16.3010xx.data\nhave been created successfully\n"
exit $?


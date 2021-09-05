# {% copyright %}
sys-nn-common-components-inc += inc
sys-nn-common-components-srcs-y += $(wildcard src/*.c*)
sys-nn-common-components-lrt-srcs-y += $(wildcard src/common_lrt/*.c*)
sys-nn-common-components-srcs-$(CONFIG_TARGET_SOC_MA2490) += $(wildcard src/2490/*.c*)
sys-nn-common-components-srcs-$(CONFIG_TARGET_SOC_3100) += $(wildcard src/2490/*.c*)

# CONFIG_TARGET_SOC_* options are mutually exclusive. Only one can be enabled at a time
target-soc-37xx = $(CONFIG_TARGET_SOC_3600)$(CONFIG_TARGET_SOC_3710)$(CONFIG_TARGET_SOC_3720)
sys-nn-common-components-srcs-$(target-soc-37xx) += $(wildcard src/3600/*.c*)

srcs-lrt-$(CONFIG_HAS_LRT_SRCS) += $(sys-nn-common-components-srcs-y) $(sys-nn-common-components-lrt-srcs-y)
srcs-lnn-$(CONFIG_HAS_LNN_SRCS) += $(sys-nn-common-components-srcs-y)
srcs-shave-$(CONFIG_HAS_SHAVE_SRCS) += $(sys-nn-common-components-srcs-y)
srcs-shave_nn-$(CONFIG_HAS_SHAVE_NN_SRCS) += $(sys-nn-common-components-srcs-y)

include-dirs-lrt-$(CONFIG_HAS_LRT_SRCS) += $(sys-nn-common-components-inc)
include-dirs-lnn-$(CONFIG_HAS_LNN_SRCS) += $(sys-nn-common-components-inc)
include-dirs-shave-$(CONFIG_HAS_SHAVE_SRCS) += $(sys-nn-common-components-inc)
include-dirs-shave_nn-$(CONFIG_HAS_SHAVE_NN_SRCS) += $(sys-nn-common-components-inc)
#include-dirs-shave_nn-$(CONFIG_TARGET_SOC_3720) += $(sys-nn-common-components-inc)

ccopt-lrt-$(CONFIG_NN_LOG_VERBOSITY_LRT_NONE)  += -DNN_LOG_VERBOSITY=0
ccopt-lrt-$(CONFIG_NN_LOG_VERBOSITY_LRT_FATAL) += -DNN_LOG_VERBOSITY=1
ccopt-lrt-$(CONFIG_NN_LOG_VERBOSITY_LRT_ERROR) += -DNN_LOG_VERBOSITY=2
ccopt-lrt-$(CONFIG_NN_LOG_VERBOSITY_LRT_WARN)  += -DNN_LOG_VERBOSITY=3
ccopt-lrt-$(CONFIG_NN_LOG_VERBOSITY_LRT_PERF)  += -DNN_LOG_VERBOSITY=4
ccopt-lrt-$(CONFIG_NN_LOG_VERBOSITY_LRT_INFO)  += -DNN_LOG_VERBOSITY=5
ccopt-lrt-$(CONFIG_NN_LOG_VERBOSITY_LRT_DEBUG) += -DNN_LOG_VERBOSITY=6
ccopt-lrt-$(CONFIG_NN_LOG_VERBOSITY_LRT_ALL)   += -DNN_LOG_VERBOSITY=7

ccopt-lnn-$(CONFIG_NN_LOG_VERBOSITY_LNN_NONE)  += -DNN_LOG_VERBOSITY=0
ccopt-lnn-$(CONFIG_NN_LOG_VERBOSITY_LNN_FATAL) += -DNN_LOG_VERBOSITY=1
ccopt-lnn-$(CONFIG_NN_LOG_VERBOSITY_LNN_ERROR) += -DNN_LOG_VERBOSITY=2
ccopt-lnn-$(CONFIG_NN_LOG_VERBOSITY_LNN_WARN)  += -DNN_LOG_VERBOSITY=3
ccopt-lnn-$(CONFIG_NN_LOG_VERBOSITY_LNN_PERF)  += -DNN_LOG_VERBOSITY=4
ccopt-lnn-$(CONFIG_NN_LOG_VERBOSITY_LNN_INFO)  += -DNN_LOG_VERBOSITY=5
ccopt-lnn-$(CONFIG_NN_LOG_VERBOSITY_LNN_DEBUG) += -DNN_LOG_VERBOSITY=6
ccopt-lnn-$(CONFIG_NN_LOG_VERBOSITY_LNN_ALL)   += -DNN_LOG_VERBOSITY=7

ccopt-shave-$(CONFIG_NN_LOG_VERBOSITY_SVU_NONE)  += -DNN_LOG_VERBOSITY=0
ccopt-shave-$(CONFIG_NN_LOG_VERBOSITY_SVU_FATAL) += -DNN_LOG_VERBOSITY=1
ccopt-shave-$(CONFIG_NN_LOG_VERBOSITY_SVU_ERROR) += -DNN_LOG_VERBOSITY=2
ccopt-shave-$(CONFIG_NN_LOG_VERBOSITY_SVU_WARN)  += -DNN_LOG_VERBOSITY=3
ccopt-shave-$(CONFIG_NN_LOG_VERBOSITY_SVU_PERF)  += -DNN_LOG_VERBOSITY=4
ccopt-shave-$(CONFIG_NN_LOG_VERBOSITY_SVU_INFO)  += -DNN_LOG_VERBOSITY=5
ccopt-shave-$(CONFIG_NN_LOG_VERBOSITY_SVU_DEBUG) += -DNN_LOG_VERBOSITY=6
ccopt-shave-$(CONFIG_NN_LOG_VERBOSITY_SVU_ALL)   += -DNN_LOG_VERBOSITY=7

ccopt-shave_nn-$(CONFIG_NN_LOG_VERBOSITY_SNN_NONE)  += -DNN_LOG_VERBOSITY=0
ccopt-shave_nn-$(CONFIG_NN_LOG_VERBOSITY_SNN_FATAL) += -DNN_LOG_VERBOSITY=1
ccopt-shave_nn-$(CONFIG_NN_LOG_VERBOSITY_SNN_ERROR) += -DNN_LOG_VERBOSITY=2
ccopt-shave_nn-$(CONFIG_NN_LOG_VERBOSITY_SNN_WARN)  += -DNN_LOG_VERBOSITY=3
ccopt-shave_nn-$(CONFIG_NN_LOG_VERBOSITY_SNN_PERF)  += -DNN_LOG_VERBOSITY=4
ccopt-shave_nn-$(CONFIG_NN_LOG_VERBOSITY_SNN_INFO)  += -DNN_LOG_VERBOSITY=5
ccopt-shave_nn-$(CONFIG_NN_LOG_VERBOSITY_SNN_DEBUG) += -DNN_LOG_VERBOSITY=6
ccopt-shave_nn-$(CONFIG_NN_LOG_VERBOSITY_SNN_ALL)   += -DNN_LOG_VERBOSITY=7

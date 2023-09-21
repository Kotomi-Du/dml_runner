`git submodule init`
`git submodule update`
cmake -D OpenCL_INCLUDE_DIR=C:\Users\GAME\Documents\Project\AIOP\drivers.gpu.compute.ai.directml\DXCrossCompilerTester\opencl\include -D OpenCL_LIBRARY=C:\Users\GAME\Documents\Project\AIOP\drivers.gpu.compute.ai.directml\DXCrossCompilerTester\opencl\lib\OpenCL.lib -D ONEDNN_BUILD_GRAPH=OFF ..

`cross_runner.exe --type=conv_cm --iters=1 conv_opts --input_shape=1,8,13,13 --filter_shape=8,8,2,2 --in_pad=0 --out_pad=0 --stride=1,1,1,1 --data_type=fp16 --layout=nchw --no_bias conv_cm_opts --dump_asm --print_reg_usage --block_w=2 --block_oc=8 --large_grf `


`cross_runner.exe --type=conv_cm --iters=1 --no_conform=1 conv_opts --input_shape=1,128,64,64 --filter_shape=16,128,1,1 --in_pad=0 --out_pad=0 --stride=1,1,1,1 --data_type=fp16 --input_layout=nhwc --output_layout=nhwc --no_bias  conv_cm_opts --dump_asm --print_reg_usage --lws=1,1,1 --block_w=8 --block_oc=16 --large_grf`

> this command uses WEIGHTS_IN_OPTIMAL_FORMAT,HAS_LEFTOVER;

### DML command
`cross_runner.exe  --type=conv_dml --iters=1 --no_conform=0 conv_opts --input_shape=1,64,254,254 --filter_shape=64,64,3,3 --in_pad=0 --out_pad=0 --stride=1,1,1,1 --data_type=fp16 --input_layout=nhwc --output_layout=nhwc --no_bias=0 --managed_weights --activation=1 --reuse_cmd=1`

> 6us
`dump_asm` is used to dump the build option

### The example of build option
#### weight reorder
```
 -I " " -DINPUT_TYPE=half -DOUTPUT_TYPE=half -DWEI_OFFSET=0 -DIC=8 -DOC=8 -DK_SIZE=2 -DLAYOUT_OIYX=1001 -DLAYOUT_IO_i8_o8_i2=1002 -DLAYOUT_OYXI_o8=1003 -DLAYOUT_OYXI_o16=1004 -DINPUT_LAYOUT=1001 -DOUTPUT_LAYOUT=1003  -mdump_asm -mCM_printregusage -DLWS_SIZE_X=1 -DLWS_SIZE_Y=1 -DLWS_SIZE_Z=1
```

#### convolution
```
-I " " -DDT_ACCU=float -DINPUT_WIDTH=13 -DINPUT_HEIGHT=13 -DINPUT_CHANNELS=8 -DOUTPUT_WIDTH=12 -DOUTPUT_HEIGHT=12 -DOUTPUT_CHANNELS=8 -DBATCH=1 -DINPUT_PAD=0 -DOUTPUT_PAD=0 -DUSE_BIAS=0 -DKERNEL_SIZE=2 -DSTRIDE_W=1 -DSTRIDE_H=1 -DSLICE_IC=1 -DBLOCK_W=2 -DBLOCK_H=1 -DBLOCK_OC=8 -DBLOCK_BATCH=1 -DWEIGHTS_IN_OPTIMAL_FORMAT=1  -mdump_asm -Qxcm_doubleGRF -mCM_printregusage -DLWS_SIZE_X=1 -DLWS_SIZE_Y=1 -DLWS_SIZE_Z=1
```

stride=2 make the results incorrect
out_pad is prepared for next in_pad?
can run ocl kernel? using dnnl? can I pick other kernel?

`validate_conformance` is to compare the results of cm and dnnl ref_convolution.
not support `conv_nhwc`

cm kernel need to copy to somewhere, how driver handle this?

how to set dml runtime?
how to set dnnl runtime?

Task:
use directml_runner to run conv_1x1_nhwc_ob32_maxbw_fp16?
1. test the perf
2. test the accuracy


create an app to run onnx model and get output
1. general impact

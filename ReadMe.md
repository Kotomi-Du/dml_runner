# Test Case
|Engine| Type|Command line|
|----|---|---|
|DML|MHA|`cross_runner.exe --iters 1 --type mha_dml mha_opts --data_type fp16 --layout nchw --mha_type qkv --shape_input 2,64,8,3,160`|
|DML|CONV|`cross_runner.exe  --type=conv_dml --iters=1 --no_conform=0 conv_opts --input_shape=1,64,254,254 --filter_shape=64,64,3,3 --in_pad=0 --out_pad=0 --stride=1,1,1,1 --data_type=fp16 --input_layout=nhwc --output_layout=nhwc --no_bias=0 --reuse_cmd=0 --managed_weights=1`|
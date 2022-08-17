#!/bin/bash
tengine_home='/zoujianfa/tengine-lite-new/'
zj_sdk_home='/zoujianfa/tengine-lite-new/weights/zj_sdk/reg_engine_inference'


# convert onnx ====>  onnx_opt
python3 yolox_opt.py --input $1$2 --output $1$3 --in_tensor 647 --out_tensor output

# extract quant img
python3 extract_quant_img.py -json_path $6 -img_path $8 -save_path $1$9 -single_class_img_quantity 100

convert_tools_path=$tengine_home'build/tools/convert_tool/'
quantize_tools_path=$tengine_home'build/tools/quantize/'


#convert onnx ====>> tmfile
cd $convert_tools_path
./convert_tool -f onnx -m $1$3 -o $1$4

#convert tmfile  ======>> tmfile uint8
cd $quantize_tools_path
./quant_tool_uint8 -m $1$4 -i $1$9 -o $1$5 -g 12,$7/2,$7/2 -k 1 -y $7,$7 -w 0,0,0 -s 1,1,1

cd $zj_sdk_home
rm -rf build
mkdir build
cd build
cmake ..
make
cd ..
./conver_file  $1$5  $1u_model.tmfile

./json_result  $1u_model.tmfile  $1'u_model_result.json'  ${10}
echo "=============================generate det json done============================="


echo "=============================start convert det json============================="
cd /zoujianfa/tengine-lite-new/weights
python3 convert_to_local_json.py -gt_path ${11} -dt_path $1'u_model_result.json'

echo "...............success................"

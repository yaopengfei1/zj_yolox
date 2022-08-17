# GPU002 docker: 18867ff09e4d     yolov5-zjf
# $1  权重根目录  /zoujianfa/tengine-lite-new/weights
# $2  放在权重根目录下的onnx模型名称 res2net.onnx 
# $3  opt之后的onnx模型  	只需要输入名称，默认保存在权重根目录  res2net_opt.onnx
# $4  onnx转换为 tmfile模型，只需要输入名称，默认保存在权重根目录 test.tmfile
# $5  tmfile模型 量化后模型 只需要输入名称，默认保存在权重根目录 test_u.tmfile uint8量化
# $6  量化需要的校准数据集,从训练时的train.json中提取，输入完整json路径 /zoujianfa/data/zhongyuan/train.json
# $7  量化参数：模型输入大小 默认320
# $8  训练图片路径 从训练图片路径中提取量化校准数据集  /zoujianfa/data/zhongyuan/train
# $9  保存量化校准数据集的路径 默认在权重根目录 只写名称 test_cer2
# $10  需要测试的test数据集的图片路径 /zoujianfa/data/zhongyuan/test
# $11  test.json和上面的测试图片必须对应 /zoujianfa/data/zhongyuan/test.json

./convert_and_quant.sh /zoujianfa/tengine-lite-new/weights/ \
	res2net.onnx \
	res2net_opt.onnx\
	test.tmfile \
	test_u.tmfile \
    /zoujianfa/data/yingyu_1/train.json \
	640 \
	/zoujianfa/data/yingyu_1/train \
	test_cer1 \
	/zoujianfa/data/yingyu_1/val \
	/zoujianfa/data/yingyu_1/val.json

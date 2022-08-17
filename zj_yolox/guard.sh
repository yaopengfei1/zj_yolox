export CUDA_VISIBLE_DEVICES=3

PRO_NAME=yolox_m_all_data

python tools/train.py \
    -f exps/example/custom/${PRO_NAME}.py \
    -c YOLOX_outputs/${PRO_NAME}/last_epoch_ckpt.pth \
    -b 16
  
while true ; do
#    用ps获取$PRO_NAME进程数量
  NUM=`ps aux | grep ${PRO_NAME} | grep -v grep |wc -l`
#  echo $NUM
#    少于1，重启进程
  if [ "${NUM}" -lt "1" ];then
    echo "${PRO_NAME} was killed"
    python tools/train.py \
    -f exps/example/custom/${PRO_NAME}.py \
    --resume -c YOLOX_outputs/${PRO_NAME}/last_epoch_ckpt.pth \
    -b 16 
#    大于1，杀掉所有进程，重启
  elif [ "${NUM}" -gt "1" ];then
    echo "more than 1 ${PRO_NAME},killall ${PRO_NAME}"
    killall -9 $PRO_NAME
  fi
#    kill僵尸进程
  NUM_STAT=`ps aux | grep ${PRO_NAME} | grep T | grep -v grep | wc -l`
  
  if [ "${NUM_STAT}" -gt "0" ];then
    killall -9 ${PRO_NAME}
  fi
done
  
exit 0
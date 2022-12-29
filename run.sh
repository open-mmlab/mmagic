PORT=22111 CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_test.sh \
configs/ifrnet/ifrnet_in2out7_8xb4_adobe.py \
checkpoint/IFRNet_GoPro_trans.pth \
4

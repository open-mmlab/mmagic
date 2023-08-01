# export PYTHONPATH=./my_code/torch_utils:$PYTHONPATH
export PYTHONPATH=./my_code:$PYTHONPATH
python demo/mmagic_inference_demo.py \
        --model-name styleganv2 \
        --model-config stylegan2_lion_512x512 \
        --result-out-dir my_code/tmp_res/1.jpg \
        --model-ckpt  ./my_code/new_ckpts/stylegan2_lions_512_pytorch_mmagic.pth \
        --seed 0 \

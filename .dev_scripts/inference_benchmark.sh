python demo/download_inference_resources.py

# Text-to-Image
python demo/mmagic_inference_demo.py \
        --model-name stable_diffusion \
        --text "A panda is having dinner at KFC" \
        --result-out-dir demo_text2image_stable_diffusion_res.png

python demo/mmagic_inference_demo.py \
        --model-name controlnet \
        --model-setting 1 \
        --text "Room with blue walls and a yellow ceiling." \
        --control 'https://user-images.githubusercontent.com/28132635/230297033-4f5c32df-365c-4cf4-8e4f-1b76a4cbb0b7.png' \
        --result-out-dir demo_text2image_controlnet_canny_res.png

python demo/mmagic_inference_demo.py \
        --model-name controlnet \
        --model-setting 2 \
        --text "masterpiece, best quality, sky, black hair, skirt, sailor collar, looking at viewer, short hair, building, bangs, neckerchief, long sleeves, cloudy sky, power lines, shirt, cityscape, pleated skirt, scenery, blunt bangs, city, night, black sailor collar, closed mouth" \
        --control 'https://user-images.githubusercontent.com/28132635/230380893-2eae68af-d610-4f7f-aa68-c2f22c2abf7e.png' \
        --result-out-dir demo_text2image_controlnet_pose_res.png

python demo/mmagic_inference_demo.py \
        --model-name controlnet \
        --model-setting 3 \
        --text "black house, blue sky" \
        --control 'https://github-production-user-asset-6210df.s3.amazonaws.com/49083766/243599897-553a4c46-c61d-46df-b820-59a49aaf6678.png' \
        --result-out-dir demo_text2image_controlnet_seg_res.png

# Conditional GANs
python demo/mmagic_inference_demo.py \
        --model-name biggan \
        --model-setting 3 \
        --label 1 \
        --result-out-dir demo_conditional_biggan_res.jpg

# Unconditional GANs
python demo/mmagic_inference_demo.py \
        --model-name styleganv1 \
        --result-out-dir demo_unconditional_styleganv1_res.jpg

# Image Translation
python demo/mmagic_inference_demo.py \
        --model-name pix2pix \
        --img ./resources/input/translation/gt_mask_0.png \
        --result-out-dir ./resources/output/translation/demo_translation_pix2pix_res.png

# Inpainting
python demo/mmagic_inference_demo.py \
        --model-name deepfillv2  \
        --img ./resources/input/inpainting/celeba_test.png \
        --mask ./resources/input/inpainting/bbox_mask.png \
        --result-out-dir ./resources/output/inpainting/demo_inpainting_deepfillv2_res.

# Matting
python demo/mmagic_inference_demo.py \
        --model-name aot_gan  \
        --img ./resources/input/matting/GT05.jpg \
        --trimap ./resources/input/matting/GT05_trimap.jpg \
        --result-out-dir ./resources/output/matting/demo_matting_gca_res.png

# Image Restoration
python demo/mmagic_inference_demo.py \
        --model-name nafnet \
        --img ./resources/input/restoration/0901x2.png \
        --result-out-dir ./resources/output/restoration/demo_restoration_nafnet_res.png

# Image Super-resolution
python demo/mmagic_inference_demo.py \
        --model-name esrgan \
        --img ./resources/input/restoration/0901x2.png \
        --result-out-dir ./resources/output/restoration/demo_restoration_esrgan_res.png

python demo/mmagic_inference_demo.py \
        --model-name ttsr \
        --img ./resources/input/restoration/000001.png \
        --ref ./resources/input/restoration/000001.png \
        --result-out-dir ./resources/output/restoration/demo_restoration_ttsr_res.png

# Video Super-Resolution
python demo/mmagic_inference_demo.py \
        --model-name basicvsr \
        --video ./resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ./resources/output/video_restoration/demo_video_restoration_basicvsr_res.mp4

python demo/mmagic_inference_demo.py \
        --model-name edvr \
        --extra-parameters window_size=5 \
        --video ./resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ./resources/output/video_restoration/demo_video_restoration_edvr_res.mp4

python demo/mmagic_inference_demo.py \
        --model-name tdan \
        --model-setting 2 \
        --extra-parameters window_size=5 \
        --video ./resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ./resources/output/video_restoration/demo_video_restoration_tdan_res.mp4

# Video interpolation
python demo/mmagic_inference_demo.py \
        --model-name flavr \
        --video ./resources/input/video_interpolation/b-3LLDhc4EU_000000_000010.mp4 \
        --result-out-dir ./resources/output/video_interpolation/demo_video_interpolation_flavr_res.mp4

# Image Colorization
python demo/mmagic_inference_demo.py \
        --model-name inst_colorization \
        --img https://github-production-user-asset-6210df.s3.amazonaws.com/49083766/245713512-de973677-2be8-4915-911f-fab90bb17c40.jpg \
        --result-out-dir demo_colorization_res.png

# 3D-aware Generation
python demo/mmagic_inference_demo.py \
    --model-name eg3d \
    --result-out-dir ./resources/output/eg3d-output

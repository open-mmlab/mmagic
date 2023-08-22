# TDAN (CVPR'2020)

> [TDAN: Temporally Deformable Alignment Network for Video Super-Resolution](https://arxiv.org/abs/1812.02898)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Video super-resolution (VSR) aims to restore a photo-realistic high-resolution (HR) video frame from both its corresponding low-resolution (LR) frame (reference frame) and multiple neighboring frames (supporting frames). Due to varying motion of cameras or objects, the reference frame and each support frame are not aligned. Therefore, temporal alignment is a challenging yet important problem for VSR. Previous VSR methods usually utilize optical flow between the reference frame and each supporting frame to wrap the supporting frame for temporal alignment. Therefore, the performance of these image-level wrapping-based models will highly depend on the prediction accuracy of optical flow, and inaccurate optical flow will lead to artifacts in the wrapped supporting frames, which also will be propagated into the reconstructed HR video frame. To overcome the limitation, in this paper, we propose a temporal deformable alignment network (TDAN) to adaptively align the reference frame and each supporting frame at the feature level without computing optical flow. The TDAN uses features from both the reference frame and each supporting frame to dynamically predict offsets of sampling convolution kernels. By using the corresponding kernels, TDAN transforms supporting frames to align with the reference frame. To predict the HR video frame, a reconstruction network taking aligned frames and the reference frame is utilized. Experimental results demonstrate the effectiveness of the proposed TDAN-based VSR model.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144035224-a87cc41e-1352-4ffa-8b07-eda5ace8a0b1.png" width="400"/>
</div >

## Results and models

Evaluated on Y-channel. 8 pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                Method                                 |   Vid4 (BIx4)   | SPMCS-30 (BIx4) |   Vid4 (BDx4)   | SPMCS-30 (BDx4) |                                Download                                 |
| :-------------------------------------------------------------------: | :-------------: | :-------------: | :-------------: | :-------------: | :---------------------------------------------------------------------: |
| [tdan_vimeo90k_bix4_ft_lr5e-5_400k](/configs/restorers/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py) | **26.49/0.792** | **30.42/0.856** |   25.93/0.772   |   29.69/0.842   | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528_135616.log.json) |
| [tdan_vimeo90k_bdx4_ft_lr5e-5_800k](/configs/restorers/tdan/tdan_vimeo90k_bdx4_ft_lr5e-5_800k.py) |   25.80/0.784   |   29.56/0.851   | **26.87/0.815** | **30.77/0.868** | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528-c53ab844.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528_122401.log.json) |

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following command to train a model.

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

TDAN is trained with two stages.

**Stage 1**: Train with a larger learning rate (1e-4)

```shell
./tools/dist_train.sh configs/restorers/tdan/tdan_vimeo90k_bix4_lr1e-4_400k.py 8
```

**Stage 2**: Fine-tune with a smaller learning rate (5e-5)

```shell
./tools/dist_train.sh configs/restorers/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py 8
```

For more details, you can refer to **Train a model** part in [getting_started](/docs/en/getting_started.md#train-a-model).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]
```

Example: Test TDAN on SPMCS-30 using Bicubic downsampling.

```shell
python tools/test.py configs/restorers/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py  checkpoints/SOME_CHECKPOINT.pth --save_path outputs/
```

For more details, you can refer to **Inference with pretrained models** part in [getting_started](/docs/en/getting_started.md#inference-with-pretrained-models).

</details>

## Citation

```bibtex
@InProceedings{tian2020tdan,
  title={TDAN: Temporally-Deformable Alignment Network for Video Super-Resolution},
  author={Tian, Yapeng and Zhang, Yulun and Fu, Yun and Xu, Chenliang},
  booktitle = {Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  year = {2020}
}
```

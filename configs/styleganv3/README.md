# StyleGANv3 (NeurIPS'2021)

> [Alias-Free Generative Adversarial Networks](https://nvlabs-fi-cdn.nvidia.com/stylegan3/stylegan3-paper.pdf)

> **Task**: Unconditional GANs

<!-- [ALGORITHM] -->

## Abstract

We observe that despite their hierarchical convolutional nature, the synthesis
process of typical generative adversarial networks depends on absolute pixel coordinates in an unhealthy manner. This manifests itself as, e.g., detail appearing to
be glued to image coordinates instead of the surfaces of depicted objects. We trace
the root cause to careless signal processing that causes aliasing in the generator
network. Interpreting all signals in the network as continuous, we derive generally
applicable, small architectural changes that guarantee that unwanted information
cannot leak into the hierarchical synthesis process. The resulting networks match
the FID of StyleGAN2 but differ dramatically in their internal representations, and
they are fully equivariant to translation and rotation even at subpixel scales. Our
results pave the way for generative models better suited for video and animation.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/22982797/150353023-8f7eeaea-8783-4ed4-98d5-67a226e00cff.png"/>
</div>

## Results and Models

<div align="center">
  <b> Results (compressed) from StyleGAN3 config-T converted by mmagic</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/150450502-c182834f-796f-4397-bd38-df1efe4a8a47.png" width="800"/>
</div>

We perform experiments on StyleGANv3 paper settings and also experimental settings.
For user convenience, we also offer the converted version of official weights.

### Paper Settings

|                                     Model                                     |      Dataset      |  Iter  |      FID50k       |                                      Download                                       |
| :---------------------------------------------------------------------------: | :---------------: | :----: | :---------------: | :---------------------------------------------------------------------------------: |
|   [stylegan3-t](./stylegan3-t_gamma32.8_8xb4-fp16-noaug_ffhq-1024x1024.py)    |  ffhq 1024x1024   | 490000 | 3.37<sup>\*</sup> | [ckpt](https://download.openmmlab.com/mmediting/stylegan3/stylegan3_t_noaug_fp16_gamma32.8_ffhq_1024_b4x8_best_fid_iter_490000_20220401_120733-4ff83434.pth) \| [log](https://download.openmmlab.com/mmediting/stylegan3/stylegan3_t_noaug_fp16_gamma32.8_ffhq_1024_b4x8_20220322_090417.log.json) |
| [stylegan3-t-ada](./stylegan3-t_ada-gamma6.6_8xb4-fp16_metfaces-1024x1024.py) | metface 1024x1024 | 130000 |       15.09       | [ckpt](https://download.openmmlab.com/mmediting/stylegan3/stylegan3_t_ada_fp16_gamma6.6_metfaces_1024_b4x8_best_fid_iter_130000_20220401_115101-f2ef498e.pth) \| [log](https://download.openmmlab.com/mmediting/stylegan3/stylegan3_t_ada_fp16_gamma6.6_metfaces_1024_b4x8_20220328_142211.log.json) |

### Experimental Settings

|                                     Model                                     |    Dataset     |  Iter  | FID50k |                                             Download                                              |
| :---------------------------------------------------------------------------: | :------------: | :----: | :----: | :-----------------------------------------------------------------------------------------------: |
|     [stylegan3-t](./stylegan3-t_gamma2.0_8xb4-fp16-noaug_ffhq-256x256.py)     |  ffhq 256x256  | 740000 |  4.51  | [ckpt](https://download.openmmlab.com/mmediting/stylegan3/stylegan3_t_noaug_fp16_gamma2.0_ffhq_256_b4x8_best_fid_iter_740000_20220401_122456-730e1fba.pth) \| [log](https://download.openmmlab.com/mmediting/stylegan3/stylegan3_t_noaug_fp16_gamma2.0_ffhq_256_b4x8_20220323_144815.log.json) |
| [stylegan3-r-ada](./stylegan3-r_ada-gamma3.3_8xb4-fp16_metfaces-1024x1024.py) | ffhq 1024x1024 |   -    |   -    |                                            [ckpt](<>)                                             |

### Converted Weights

|                                 Model                                  |    Dataset     |     Comment     | FID50k | EQ-T  | EQ-R  |                                       Download                                        |
| :--------------------------------------------------------------------: | :------------: | :-------------: | :----: | :---: | :---: | :-----------------------------------------------------------------------------------: |
|  [stylegan3-t](./stylegan3-t_cvt-official-rgb_8xb4_ffhqu-256x256.py)   | ffhqu 256x256  | official weight |  4.62  | 63.01 | 13.12 | [ckpt](https://download.openmmlab.com/mmediting/stylegan3/stylegan3_t_ffhqu_256_b4x8_cvt_official_rgb_20220329_235046-153df4c8.pth) |
|  [stylegan3-t](./stylegan3-t_cvt-official-rgb_8xb4_afhqv2-512x512.py)  | afhqv2 512x512 | official weight |  4.04  | 60.15 | 13.51 | [ckpt](https://download.openmmlab.com/mmediting/stylegan3/stylegan3_t_afhqv2_512_b4x8_cvt_official_rgb_20220329_235017-ee6b037a.pth) |
|  [stylegan3-t](./stylegan3-t_cvt-official-rgb_8xb4_ffhq-1024x1024.py)  | ffhq 1024x1024 | official weight |  2.79  | 61.21 | 13.82 | [ckpt](https://download.openmmlab.com/mmediting/stylegan3/stylegan3_t_ffhq_1024_b4x8_cvt_official_rgb_20220329_235113-db6c6580.pth) |
|  [stylegan3-r](./stylegan3-r_cvt-official-rgb_8xb4_ffhqu-256x256.py)   | ffhqu 256x256  | official weight |  4.50  | 66.65 | 40.48 | [ckpt](https://download.openmmlab.com/mmediting/stylegan3/stylegan3_r_ffhqu_256_b4x8_cvt_official_rgb_20220329_234909-4521d963.pth) |
| [stylegan3-r](./stylegan3-r_cvt-official-rgb_8xb4x8_afhqv2-512x512.py) | afhqv2 512x512 | official weight |  4.40  | 64.89 | 40.34 | [ckpt](https://download.openmmlab.com/mmediting/stylegan3/stylegan3_r_afhqv2_512_b4x8_cvt_official_rgb_20220329_234829-f2eaca72.pth) |
|  [stylegan3-r](./stylegan3-r_cvt-official-rgb_8xb4_ffhq-1024x1024.py)  | ffhq 1024x1024 | official weight |  3.07  | 64.76 | 46.62 | [ckpt](https://download.openmmlab.com/mmediting/stylegan3/stylegan3_r_ffhq_1024_b4x8_cvt_official_rgb_20220329_234933-ac0500a1.pth) |

## Interpolation

We provide a tool to generate video by walking through GAN's latent space.
Run this command to get the following video.

```bash
python apps/interpolate_sample.py configs/styleganv3/stylegan3_t_afhqv2_512_b4x8_official.py https://download.openmmlab.com/mmediting/stylegan3/stylegan3_t_afhqv2_512_b4x8_cvt_official.pkl --export-video --samples-path work_dirs/demos/ --endpoint 6 --interval 60 --space z --seed 2022 --sample-cfg truncation=0.8
```

https://user-images.githubusercontent.com/22982797/151506918-83da9ee3-0d63-4c5b-ad53-a41562b92075.mp4

## Equivarience Visualization && Evaluation

We also provide a tool to visualize the equivarience properties for StyleGAN3.
Run these commands to get the results below.

```bash
python tools/utils/equivariance_viz.py configs/styleganv3/stylegan3_r_ffhqu_256_b4x8_official.py https://download.openmmlab.com/mmediting/stylegan3/stylegan3_r_ffhqu_256_b4x8_cvt_official.pkl --translate_max 0.5 --transform rotate --seed 5432

python tools/utils/equivariance_viz.py configs/styleganv3/stylegan3_r_ffhqu_256_b4x8_official.py https://download.openmmlab.com/mmediting/stylegan3/stylegan3_r_ffhqu_256_b4x8_cvt_official.pkl --translate_max 0.25 --transform x_t --seed 5432

python tools/utils/equivariance_viz.py configs/styleganv3/stylegan3_r_ffhqu_256_b4x8_official.py https://download.openmmlab.com/mmediting/stylegan3/stylegan3_r_ffhqu_256_b4x8_cvt_official.pkl --translate_max 0.25 --transform y_t --seed 5432
```

https://user-images.githubusercontent.com/22982797/151504902-f3cbfef5-9014-4607-bbe1-deaf48ec6d55.mp4

https://user-images.githubusercontent.com/22982797/151504973-b96e1639-861d-434b-9d7c-411ebd4a653f.mp4

https://user-images.githubusercontent.com/22982797/151505099-cde4999e-aab1-42d4-a458-3bb069db3d32.mp4

If you want to get EQ-Metric for StyleGAN3, just add following codes into config.

```python
metrics = dict(
    eqv=dict(
        type='Equivariance',
        num_images=50000,
        eq_cfg=dict(
            compute_eqt_int=True, compute_eqt_frac=True, compute_eqr=True)))
```

And we highly recommend you to use [slurm_test.sh](../../tools/slurm_test.sh) script to accelerate evaluation time.

## Citation

```latex
@inproceedings{Karras2021,
  author = {Tero Karras and Miika Aittala and Samuli Laine and Erik H\"ark\"onen and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  title = {Alias-Free Generative Adversarial Networks},
  booktitle = {Proc. NeurIPS},
  year = {2021}
}
```

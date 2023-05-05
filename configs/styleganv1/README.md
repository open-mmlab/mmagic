# StyleGANv1 (CVPR'2019)

> [A Style-Based Generator Architecture for Generative Adversarial Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html)

> **Task**: Unconditional GANs

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. Finally, we introduce a new, highly varied and high-quality dataset of human faces.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143055313-f4988870-2963-4a2f-916e-0de0e04eb474.JPG"/>
</div>

## Results and Models

<div align="center">
  <b> Results (compressed) from StyleGANv1 trained by mmagic</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/113845642-4f9ee980-97c8-11eb-85c7-49d6d21dd46b.png" width="800"/>
</div>

|                                Model                                | Dataset | FID50k |  P&R50k_full  |                                                  Download                                                   |
| :-----------------------------------------------------------------: | :-----: | :----: | :-----------: | :---------------------------------------------------------------------------------------------------------: |
|  [styleganv1_ffhq_256](./styleganv1_ffhq-256x256_8xb4-25Mimgs.py)   |  FFHQ   | 6.090  | 70.228/27.050 | [model](https://download.openmmlab.com/mmediting/styleganv1/styleganv1_ffhq_256_g8_25Mimg_20210407_161748-0094da86.pth) |
| [styleganv1_ffhq_1024](./styleganv1_ffhq-1024x1024_8xb4-25Mimgs.py) |  FFHQ   | 4.056  | 70.302/36.869 | [model](https://download.openmmlab.com/mmediting/styleganv1/styleganv1_ffhq_1024_g8_25Mimg_20210407_161627-850a7234.pth) |

## Citation

```latex
@inproceedings{karras2019style,
  title={A style-based generator architecture for generative adversarial networks},
  author={Karras, Tero and Laine, Samuli and Aila, Timo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4401--4410},
  year={2019},
  url={https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html},
}
```

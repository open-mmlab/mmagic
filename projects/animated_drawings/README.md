# Animated Drawings (SIGGRAPH'2023)

> [A Method for Animating Children's Drawings of The Human Figure](https://arxiv.org/abs/2303.12741)

> **Task**: Drawing

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Children’s drawings have a wonderful inventiveness, creativity, and variety to them. We present a system that automatically animates children’s drawings of the human figure, is robust to the variance inherent in these depictions, and is simple and straightforward enough for anyone to use. We demonstrate the value and broad appeal of our approach by building and releasing the Animated Drawings Demo, a freely available public website that has been used by millions of people around the world. We present a set of experiments exploring the amount of training data needed for fine-tuning, as well as a perceptual study demonstrating the appeal of a novel twisted perspective retargeting technique. Finally, we introduce the Amateur Drawings Dataset, a first-of-its-kind annotated dataset, collected via the public demo, containing over 178,000 amateur drawings and corresponding user-accepted character bounding boxes, segmentation masks, and joint location annotations.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/6675724/219223438-2c93f9cb-d4b5-45e9-a433-149ed76affa6.gif" width="800"/>
</div >

## Quick Start

### 1. Install Animated Drawings

```shell
cd  mmagic/projects/animated_drawings
pip install -e git+https://github.com/facebookresearch/AnimatedDrawings.git#egg=animated_drawings
```

### 2. Download resources

```shell
cd  mmagic/projects/animated_drawings
mkdir resources
# download image
wget -O sakuragi.png https://user-images.githubusercontent.com/12782558/236157945-452fb9d0-e02e-4f36-8338-34f0ca0fe962.png
# download mask image
wget -O sakuragi_mask.png https://user-images.githubusercontent.com/12782558/236157965-539a5467-edae-40d0-a9da-7bb5906bcdc4.png
# download background image
wget -O basketball_playground.png https://user-images.githubusercontent.com/12782558/236190727-0e456482-2ae3-4304-9e6c-6ba7d319ea71.png
```

### 3. Generate character

By running following codes, you will get texture and mask images for animated rendering in characters/slamdunk directory.

```shell
cd mmagic/projects/animated_drawings
python tools/generate_character.py
```

### 3. Generate video

By running following codes, you will get a Sakuragi moving like a zombie.

```shell
cd mmagic/projects/animated_drawings
python tools/generate_video.py
```

The output video will be saved at resources dir, and it looks like this:

<div align=center >
 <img src="https://user-images.githubusercontent.com/12782558/236162056-c9a65baa-89c4-4cb3-84da-7777f5f21757.gif" width="512"/>
</div >

## Citation

```bibtex
@misc{smith2023method,
      title={A Method for Animating Children's Drawings of the Human Figure},
      author={Harrison Jesse Smith and Qingyuan Zheng and Yifei Li and Somya Jain and Jessica K. Hodgins},
      year={2023},
      eprint={2303.12741},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

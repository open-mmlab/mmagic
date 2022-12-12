# Overview

Welcome to MMEditing! In this section, you will know about

- [What is MMEditing?](#what-is-mmediting)
- [Why should I use MMEditing?](#why-should-i-use-mmediting)
- [Get started](#get-started)
- [User guides](#user-guides)
- [Advanced guides](#advanced-guides)

## What is MMEditing?

MMEditing is an open-source toolbox for professional AI researchers and machine learning engineers to explore image and video processing, editing and synthesis.

MMEditing allows researchers and engineers to use pre-trained state-of-the-art models, train and develop new customized models easily.

MMEditing supports various foundamental generative models, including:

- Unconditional Generative Adversarial Networks (GANs)
- Conditional Generative Adversarial Networks (GANs)
- Internal Learning
- Diffusion Models
- And many other generative models are coming soon!

MMEditing supports various applications, including:

- Image super-resolution
- Video super-resolution
- Video frame interpolation
- Image inpainting
- Image matting
- Image-to-image translation
- And many other applications are coming soon!

<div align=center>
  <img src="https://user-images.githubusercontent.com/12756472/158984079-c4754015-c1f6-48c5-ac46-62e79448c372.jpg"/>
</div>
</br>

<div align=center>
    <video width="100%" controls>
        <source src="https://user-images.githubusercontent.com/12756472/175944645-cabe8c2b-9f25-440b-91cc-cdac4e752c5a.mp4" type="video/mp4">
        <object data="https://user-images.githubusercontent.com/12756472/175944645-cabe8c2b-9f25-440b-91cc-cdac4e752c5a.mp4" width="100%">
        </object>
    </video>
</div>
</br>

<div align=center>
<video width="100%" controls>
    <source src="https://user-images.githubusercontent.com/12756472/158972813-d8d0f19c-f49c-4618-9967-52652726ef19.mp4" type="video/mp4">
    <object src="https://user-images.githubusercontent.com/12756472/158972813-d8d0f19c-f49c-4618-9967-52652726ef19.mp4" width="100%">
    </object>
</video>
</div>

<div align="center">
  <b> StyleGAN3 Images</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/150450502-c182834f-796f-4397-bd38-df1efe4a8a47.png" width="800"/>
</div>

<div align="center">
  <b> BigGAN Images </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/127615534-6278ce1b-5cff-4189-83c6-9ecc8de08dfc.png" width="800"/>
</div>

## Why should I use MMEditing?

- **State of the Art**

  MMEditing provides state-of-the-art generative models to process, edit and synthesize images and videos.

- **Powerful and Popular Applications**

  MMEditing supports popular and contemporary *inpainting*, *matting*, *super-resolution* and *generation* applications. Specifically, MMEditing supports GAN interpolation, GAN projection, GAN manipulations and many other popular GAN's applications. It's time to play with your GANs!

- **New Modular Design for Flexible Combination:**

  We decompose the editing framework into different modules and one can easily construct a customized editor framework by combining different modules. Specifically, a new design for complex loss modules is proposed for customizing the links between modules, which can achieve flexible combinations among different modules.(Tutorial for [losses](../howto/losses.md))

- **Efficient Distributed Training:**

  With the support of [MMSeparateDistributedDataParallel](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/wrappers/seperate_distributed.py), distributed training for dynamic architectures can be easily implemented.

## Get started

For installation instructions, please see [Installation](install.md).

## User guides

For beginners, we suggest learning the basic usage of MMEditing from [user_guides](../user_guides/config.md).

### Advanced guides

For users who are familiar with MMEditing, you may want to learn the design of MMEditing, as well as how to extend the repo, how to use multiple repos and other advanced usages, please refer to [advanced_guides](../advanced_guides/evaluator.md).

### How to

For users who want to use MMEditing to do something, please refer to [How to](../howto/models.md).

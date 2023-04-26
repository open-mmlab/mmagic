# Tutorial 7: Useful tools

We provide lots of useful tools under `tools/` directory.

The structure of this guide is as follows:

- [Tutorial 7: Useful tools](#tutorial-7-useful-tools)
  - [Get the FLOPs and params](#get-the-flops-and-params)
  - [Publish a model](#publish-a-model)
  - [Print full config](#print-full-config)

## Get the FLOPs and params

We provide a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

For example,

```shell
python tools/analysis_tools/get_flops.py configs/resotorer/srresnet.py --shape 40 40
```

You will get the result like this.

```
==============================
Input shape: (3, 40, 40)
Flops: 4.07 GMac
Params: 1.52 M
==============================
```

**Note**: This tool is still experimental and we do not guarantee that the number is correct. You may well use the result for simple comparisons, but double check it before you adopt it in technical reports or papers.

(1) FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 250, 250).
(2) Some operators are not counted in FLOPs like GN and custom operators.
You can add support for new operators by modifying [`mmcv/cnn/utils/flops_counter.py`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py).

## Publish a model

Before you upload a model to AWS, you may want to

1. convert model weights to CPU tensors
2. delete the optimizer states and
3. compute the hash of the checkpoint file and append time and the hash id to the
   filename.

```shell
python tools/model_converters/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/model_converters/publish_model.py work_dirs/stylegan2/latest.pth stylegan2_c2_8xb4_ffhq-1024x1024.pth
```

The final output filename will be `stylegan2_c2_8xb4_ffhq-1024x1024_{time}-{hash id}.pth`.

## Print full config

MMGeneration incorporates config mechanism to set parameters used for training and testing models. With our [config](../user_guides/config.md) mechanism, users can easily conduct extensive experiments without hard coding. If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

An Example:

```shell
python tools/misc/print_config.py configs/styleganv2/stylegan2_c2-PL_8xb4-fp16-partial-GD-no-scaler-800kiters_ffhq-256x256.py
```

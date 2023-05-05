# Scripts for developing MMagic

- [1. Check UT](#1-check-ut)
- [2. Test all the models](#2-test-all-the-models)
- [3. Train all the models](#3-train-all-the-models)
  - [3.1 Train for debugging](#31-train-for-debugging)
  - [3.2 Train for FP32](#32-train-for-fp32)
  - [3.3 Train for FP16](#33-train-for-fp16)
- [4. Monitor your training](#4-monitor-your-training)
- [5. Train with a list of models](#5-train-with-a-list-of-models)
- [6. Train with skipping a list of models](#6-train-with-skipping-a-list-of-models)
- [7. Train failed or canceled jobs](#7-train-failed-or-canceled-jobs)
- [8. Deterministic training](#8-deterministic-training)
- [9. Automatically check links](#9-automatically-check-links)
- [10. Calculate flops](#10-calculate-flops)
- [11. Update model idnex](#11-update-model-index)

## 1. Check UT

Please check your UT by the following scripts:

```python
cd mmagic/
python .dev_script/update_ut.py
```

Then, you will find some redundant UT, missing UT and blank UT.
Please create UTs according to your package code implementation.

## 2. Test all the models

Please follow these steps to test all the models in MMagic:

First, you will need download all the pre-trained checkpoints by:

```shell
python .dev_scripts/download_models.py
```

Then, you can start testing all the benchmarks by：

```shell
python .dev_scripts/test_benchmark.py
```

## 3. Train all the models

### 3.1 Train for debugging

In order to test all the pipelines of training, visualization, etc., you may want to set the total iterations of all the models as less steps (e.g., 100 steps) for quick evaluation. You can use the following steps:

First, since our datasets are stored in ceph, you need to create ceph configs.

```shell
# create configs
python .dev_scripts/create_ceph_configs.py \
        --target-dir configs_ceph_debug \
        --gpus-per-job 2 \
        --iters 100 \
        --save-dir-prefix work_dirs/benchmark_debug \
        --work-dir-prefix work_dirs/benchmark_debug
```

If you only want to update a specific config file, you can specify it by `--test-file configs/aot_gan/aot-gan_smpgan_4xb4_places-512x512.py`.

Here, `--target-dir` denotes the path of new created configs, `--gpus-per-job` denotes the numbers of gpus used for each job, `--iters` denotes the total iterations of each model, `--save-dir-prefix` and `--work-dir-prefix` denote the working directory, where you can find the working logging.

Then, you will need to submit all the jobs by running `train_benchmark.py`.

```shell
python .dev_scripts/train_benchmark.py mm_lol \
    --config-dir configs_ceph_debug \
    --run \
    --gpus-per-job 2 \
    --job-name debug \
    --work-dir work_dirs/benchmark_debug \
    --resume \
    --quotatype=auto
```

Here, you will specify the configs files used for training by `--config-dir`, submit all the jobs to run by set `--run`. You can set the prefix name of the submitted jobs by `--job-name`, specify the working directory by `--work-dir`. We suggest using `--resume` to enable auto resume during training and `--quotatype=auto` to fully exploit all the computing resources.

### 3.2 Train for FP32

If you want to train all the models with FP32 (i.e, regular settings as the same with `configs/`),
you can follow these steps:

```shell
# create configs for fp32
python .dev_scripts/create_ceph_configs.py \
        --target-dir configs_ceph_fp32 \
        --gpus-per-job 4 \
        --save-dir-prefix work_dirs/benchmark_fp32 \
        --work-dir-prefix work_dirs/benchmark_fp32 \
```

Then, submit the jobs to run by slurm:

```shell
python .dev_scripts/train_benchmark.py mm_lol \
    --config-dir configs_ceph_fp32 \
    --run \
    --resume \
    --gpus-per-job 4 \
    --job-name fp32 \
    --work-dir work_dirs/benchmark_fp32 \
    --quotatype=auto
```

### 3.3 Train for FP16

You will also need to train the models with AMP (i.e., FP16), you can use the following steps to achieve this:

```shell
python .dev_scripts/create_ceph_configs.py \
        --target-dir configs_ceph_amp \
        --gpus-per-job 4 \
        --save-dir-prefix work_dirs/benchmark_amp \
        --work-dir-prefix work_dirs/benchmark_amp
```

Then, submit the jobs to run:

```shell
python .dev_scripts/train_benchmark.py mm_lol \
    --config-dir configs_ceph_amp \
    --run \
    --resume \
    --gpus-per-job 4 \
    --amp \
    --job-name amp \
    --work-dir work_dirs/benchmark_amp \
    --quotatype=auto
```

## 4. Monitor your training

After you submitting jobs following [3-Train-all-the-models](#3-train-all-the-models), you will find a `xxx.log` file.
This log file list all the job name of job id you have submitted. With this log file, you can monitor your training by running `.dev_scripts/job_watcher.py`.

For example, you can run

```shell
python .dev_scripts/job_watcher.py --work-dir work_dirs/benchmark_fp32/ --log 20220923-140317.log
```

Then, you will find `20220923-140317.csv`, which reports the status and recent log of each job.

## 5. Train with a list of models

If you only need to run some of the models, you can list all the models' name in a file, and specify the models when using `train_benchmark.py`.

For example,

```shell
python .dev_scripts/train_benchmark.py mm_lol \
    --config-dir configs_ceph_fp32 \
    --run \
    --resume \
    --gpus-per-job 4 \
    --job-name fp32 \
    --work-dir work_dirs/benchmark_fp32 \
    --quotatype=auto \
    --rerun \
    --rerun-list 20220923-140317.log \
```

Specifically, you need to enable `--rerun`, and specify the list of models to rerun by `--rerun-list`

## 6. Train with skipping a list of models

If you want to train all the models while skipping some models, you can also list all the models' name in a file, and specify the models when running `train_benchmark.py`.

For example,

```shell
python .dev_scripts/train_benchmark.py mm_lol \
    --config-dir configs_ceph_fp32 \
    --run \
    --resume \
    --gpus-per-job 4 \
    --job-name fp32 \
    --work-dir work_dirs/benchmark_fp32 \
    --quotatype=auto \
    --skip \
    --skip-list 20220923-140317.log \
```

Specifically, you need to enable `--skip`, and specify the list of models to skip by `--skip-list`

## 7. Train failed or canceled jobs

If you want to rerun failed or canceled jobs in the last run, you can combine `--rerun` flag with `--rerun-failure` and `--rerun-cancel` flags.

For example, the log file of the last run is `train-20221009-211904.log`, and now you want to rerun the failed jobs. You can use the following command:

```bash
python .dev_scripts/train_benchmark.py mm_lol \
    --job-name RERUN \
    --rerun train-20221009-211904.log \
    --rerun-fail \
    --run
```

We can combine `--rerun-fail` and `--rerun-cancel` with flag `---models` to rerun a **subset** of failed or canceled model.

```bash
python .dev_scripts/train_benchmark.py mm_lol \
    --job-name RERUN \
    --rerun train-20221009-211904.log \
    --rerun-fail \
    --models sagan \  # only rerun 'sagan' models in all failed tasks
    --run
```

Specifically, `--rerun-fail` and `--rerun-cancel` can be used together to rerun both failed and cancaled jobs.

## 8. `deterministic` training

Set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` can remove randomness operation in Pytorch training. You can add `--deterministic` flag when start your benchmark training to remove the influence of randomness operation.

```shell
python .dev_scripts/train_benchmark.py mm_lol --job-name xzn --models pix2pix --cpus-per-job 16 --run --deterministic
```

## 9. Automatically check links

Use the following script to check whether the links in documentations are valid:

```shell
python .dev_scripts/doc_link_checker.py --target docs/zh_cn
python .dev_scripts/doc_link_checker.py --target README_zh-CN.md
python .dev_scripts/doc_link_checker.py --target docs/en
python .dev_scripts/doc_link_checker.py --target README.md
```

You can specify the `--target` by a file or a directory.

**Notes:** DO NOT use it in CI, because requiring too many http requirements by CI will cause 503 and CI will propabaly fail.

## 10. Calculate flops

To summarize the flops of different models, you can run the following commands:

```bash
python .dev_scripts/benchmark_valid_flop.py --flops --flops-str
```

## 11. Update model index

To update model-index according to `README.md`, please run the following commands,

```bash
python .dev_scripts/update_model_index.py
```

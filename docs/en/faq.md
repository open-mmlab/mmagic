# Frequently asked questions

We list some common troubles faced by many users and their corresponding
solutions here. Feel free to enrich the list if you find any frequent issues
and have ways to help others to solve them. If the contents here do not cover
your issue, please create an issue using the
[provided templates](https://github.com/open-mmlab/mmagic/issues/new/choose)
and make sure you fill in all required information in the template.

## FAQ

**Q1**: “xxx: ‘yyy is not in the zzz registry’”.

**A1**: The registry mechanism will be triggered only when the file of the module is imported. So you need to import that file somewhere.

**Q2**: What's the folder structure of xxx dataset?

**A2**: You can make sure the folder structure is correct following tutorials of [dataset preparation](user_guides/dataset_prepare.md).

**Q3**: How to use LMDB data to train the model?

**A3**:  You can use scripts in `tools/data` to make LMDB files. More details are shown in tutorials of [dataset preparation](user_guides/dataset_prepare.md).

**Q4**: Why `MMCV==xxx is used but incompatible` is raised when import I try to import `mmgen`?

**A4**:
This is because the version of MMCV and MMGeneration are incompatible. Compatible MMGeneration and MMCV versions are shown as below. Please choose the correct version of MMCV to avoid installation issues.

| MMGeneration version |   MMCV version   |
| :------------------: | :--------------: |
|        master        | mmcv-full>=2.0.0 |

Note: You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

**Q5**: How can I ignore some fields in the base configs?

**A5**:
Sometimes, you may set `_delete_=True` to ignore some of fields in base configs.
You may refer to [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/config.md#delete-key-in-dict) for simple illustration.

You may have a careful look at [this tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/config.md) for better understanding of this feature.

**Q6**:: How can I use intermediate variables in configs?

**A6**:
Some intermediate variables are used in the config files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs, users need to pass the intermediate variables into corresponding fields again.

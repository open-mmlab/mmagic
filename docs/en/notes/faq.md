# Frequently Asked Questions

We list some common troubles faced by many users and their corresponding
solutions here. Feel free to enrich the list if you find any frequent issues
and have ways to help others to solve them. If the contents here do not cover
your issue, please create an issue using the
[provided templates](https://github.com/open-mmlab/mmediting/issues/new/choose)
and make sure you fill in all required information in the template.

## FAQ

- KeyError: “xxx: ‘yyy is not in the zzz registry’”.

  The registry mechanism will be triggered only when the file of the module is imported. So you need to import that file somewhere.

- What's the folder structure of xxx dataset.

  You can make sure the folder structure is correct following tutorials of [dataset preparation](../advanced_guides/dataset.md).

- How to use LMDB data to train the model?
  YOu can use scripts in `tools/data` to make LMDB files. More details are shown in tutorials of [dataset preparation](../advanced_guides/dataset.md).

#!/usr/bin/env bash

sed -i '$a\\n<br/><hr/>\n' ../../configs/inpainting/*/README_zh-CN.md
sed -i '$a\\n<br/><hr/>\n' ../../configs/mattors/*/README_zh-CN.md
sed -i '$a\\n<br/><hr/>\n' ../../configs/restorers/*/README_zh-CN.md
sed -i '$a\\n<br/><hr/>\n' ../../configs/synthesizers/*/README_zh-CN.md

# gather models
cat ../../configs/inpainting/*/README_zh-CN.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# 补全模型' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >inpainting_models.md
cat ../../configs/mattors/*/README_zh-CN.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# 抠图模型' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >mattors_models.md
cat ../../configs/restorers/*/README_zh-CN.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# 超分辨率模型' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >restorers_models.md
cat ../../configs/synthesizers/*/README_zh-CN.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# 生成模型' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >synthesizers_models.md

# gather datasets
cat ../../tools/data/generation/README_zh-CN.md > generation_datasets.md
cat ../../tools/data/inpainting/README_zh-CN.md > inpainting_datasets.md
cat ../../tools/data/matting/README_zh-CN.md > matting_datasets.md
cat ../../tools/data/super-resolution/README_zh-CN.md > sr_datasets.md
cat ../../tools/data/video-interpolation/README.md > vfi_datasets.md

sed -i 's=(paired-pix2pix/README_zh-CN.md)=(#paired-dataset-for-pix2pix)=g' generation_datasets.md
sed -i 's=(unpaired-cyclegan/README_zh-CN.md)=(#unpaired-dataset-for-cyclegan)=g' generation_datasets.md
sed -i 's=(paris-street-view/README_zh-CN.md)=(#paris-street-view-dataset)=g' inpainting_datasets.md
sed -i 's=(celeba-hq/README_zh-CN.md)=(#celeba-hq-dataset)=g' inpainting_datasets.md
sed -i 's=(places365/README_zh-CN.md)=(#places365-dataset)=g' inpainting_datasets.md
sed -i 's=(comp1k/README_zh-CN.md)=(#composition-1k-dataset)=g' matting_datasets.md
sed -i 's=(div2k/README_zh-CN.md)=(#div2k-dataset)=g' sr_datasets.md
sed -i 's=(reds/README_zh-CN.md)=(#reds-dataset)=g' sr_datasets.md
sed -i 's=(vimeo90k/README_zh-CN.md)=(#vimeo90k-dataset)=g' sr_datasets.md
sed -i 's=(vimeo90k-triplet/README.md)=(#vimeo90k-triplet-dataset)=g' vfi_datasets.md

cat ../../tools/data/generation/*/README_zh-CN.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> generation_datasets.md
cat ../../tools/data/inpainting/*/README_zh-CN.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> inpainting_datasets.md
cat ../../tools/data/matting/*/README_zh-CN.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> matting_datasets.md
cat ../../tools/data/super-resolution/*/README_zh-CN.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> sr_datasets.md

# merge configs
cat configs/config_*.md | sed "s/#/#&/" >> config.md

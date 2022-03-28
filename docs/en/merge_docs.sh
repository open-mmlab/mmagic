#!/usr/bin/env bash

mkdir -p _tmp
rm -r _tmp/*
cp -r ../../configs/ _tmp/
find _tmp/configs -name README_zh-CN.md | xargs rm

sed -i '$a\\n<br/><hr/>\n' _tmp/configs/inpainting/*/README.md
sed -i '$a\\n<br/><hr/>\n' _tmp/configs/mattors/*/README.md
sed -i '$a\\n<br/><hr/>\n' _tmp/configs/restorers/*/README.md
sed -i '$a\\n<br/><hr/>\n' _tmp/configs/synthesizers/*/README.md
sed -i '$a\\n<br/><hr/>\n' _tmp/configs/video_interpolators/*/README.md

# gather models
cat ../../configs/inpainting/*/README.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Inpainting Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' > _tmp/inpainting_models.md
cat ../../configs/mattors/*/README.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Matting Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' > _tmp/mattors_models.md
cat ../../configs/restorers/*/README.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Super-Resolution Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' > _tmp/restorers_models.md
cat ../../configs/synthesizers/*/README.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Generation Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' > _tmp/synthesizers_models.md
cat ../../configs/video_interpolators/*/README.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Frame-Interpolation Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' > _tmp/video_interpolators_models.md

# gather datasets
cat ../../tools/data/generation/README.md > _tmp/generation_datasets.md
cat ../../tools/data/inpainting/README.md > _tmp/inpainting_datasets.md
cat ../../tools/data/matting/README.md > _tmp/matting_datasets.md
cat ../../tools/data/super-resolution/README.md > _tmp/sr_datasets.md
cat ../../tools/data/video-interpolation/README.md > _tmp/vfi_datasets.md

sed -i 's=(paired-pix2pix/README.md)=(#paired-dataset-for-pix2pix)=g' _tmp/generation_datasets.md
sed -i 's=(unpaired-cyclegan/README.md)=(#unpaired-dataset-for-cyclegan)=g' _tmp/generation_datasets.md
sed -i 's=(paris-street-view/README.md)=(#paris-street-view-dataset)=g' _tmp/inpainting_datasets.md
sed -i 's=(celeba-hq/README.md)=(#celeba-hq-dataset)=g' _tmp/inpainting_datasets.md
sed -i 's=(places365/README.md)=(#places365-dataset)=g' _tmp/inpainting_datasets.md
sed -i 's=(comp1k/README.md)=(#composition-1k-dataset)=g' _tmp/matting_datasets.md
sed -i 's=(div2k/README.md)=(#div2k-dataset)=g' _tmp/sr_datasets.md
sed -i 's=(reds/README.md)=(#reds-dataset)=g' _tmp/sr_datasets.md
sed -i 's=(vimeo90k/README.md)=(#vimeo90k-dataset)=g' _tmp/sr_datasets.md
sed -i 's=(vimeo90k-triplet/README.md)=(#vimeo90k-triplet-dataset)=g' _tmp/vfi_datasets.md

cat ../../tools/data/generation/*/README.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> _tmp/generation_datasets.md
cat ../../tools/data/inpainting/*/README.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> _tmp/inpainting_datasets.md
cat ../../tools/data/matting/*/README.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> _tmp/matting_datasets.md
cat ../../tools/data/super-resolution/*/README.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> _tmp/sr_datasets.md

# merge configs
cp config.md _tmp/
cat configs/config_*.md | sed "s/#/#&/" >> _tmp/config.md

# clean
rm -r _tmp/configs

#!/usr/bin/env bash

sed -i '$a\\n<br/><hr/>\n' ../configs/inpainting/*/README.md
sed -i '$a\\n<br/><hr/>\n' ../configs/mattors/*/README.md
sed -i '$a\\n<br/><hr/>\n' ../configs/restorers/*/README.md
sed -i '$a\\n<br/><hr/>\n' ../configs/synthesizers/*/README.md

# gather models
cat ../configs/inpainting/*/README.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Inpainting Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >inpainting_models.md
cat ../configs/mattors/*/README.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Matting Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >mattors_models.md
cat ../configs/restorers/*/README.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Super-Resolution Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >restorers_models.md
cat ../configs/synthesizers/*/README.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Generation Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >synthesizers_models.md

# gather datasets
cat ../tools/data/generation/README.md > generation_datasets.md
cat ../tools/data/inpainting/README.md > inpainting_datasets.md
cat ../tools/data/matting/README.md > matting_datasets.md
cat ../tools/data/super-resolution/README.md > sr_datasets.md

sed -i 's=(paired-pix2pix/README.md)=(#paired-dataset-for-pix2pix)=g' generation_datasets.md
sed -i 's=(unpaired-cyclegan/README.md)=(#unpaired-dataset-for-cyclegan)=g' generation_datasets.md
sed -i 's=(paris-street-view/README.md)=(#paris-street-view-dataset)=g' inpainting_datasets.md
sed -i 's=(celeba-hq/README.md)=(#celeba-hq-dataset)=g' inpainting_datasets.md
sed -i 's=(places365/README.md)=(#places365-dataset)=g' inpainting_datasets.md
sed -i 's=(comp1k/README.md)=(#composition-1k-dataset)=g' matting_datasets.md
sed -i 's=(div2k/README.md)=(#div2k-dataset)=g' sr_datasets.md
sed -i 's=(reds/README.md)=(#reds-dataset)=g' sr_datasets.md
sed -i 's=(vimeo90k/README.md)=(#vimeo90k-dataset)=g' sr_datasets.md

cat ../tools/data/generation/*/README.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> generation_datasets.md
cat ../tools/data/inpainting/*/README.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> inpainting_datasets.md
cat ../tools/data/matting/*/README.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> matting_datasets.md
cat ../tools/data/super-resolution/*/README.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> sr_datasets.md

# merge configs
cat configs/config_*.md | sed "s/#/#&/" >> config.md

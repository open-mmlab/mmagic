#!/usr/bin/env bash

sed -i '$a\\n<br/><hr/>\n' ../configs/inpainting/*/*.md
sed -i '$a\\n<br/><hr/>\n' ../configs/mattors/*/*.md
sed -i '$a\\n<br/><hr/>\n' ../configs/restorers/*/*.md
sed -i '$a\\n<br/><hr/>\n' ../configs/synthesizers/*/*.md

# gather models
cat ../configs/inpainting/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Inpainting Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >inpainting_models.md
cat ../configs/mattors/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Matting Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >mattors_models.md
cat ../configs/restorers/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Super-Resolution Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >restorers_models.md
cat ../configs/synthesizers/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Generation Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >synthesizers_models.md

# gather datasets
cat ../tools/data/generation/*.md > generation_datasets.md
cat ../tools/data/inpainting/*.md > inpainting_datasets.md
cat ../tools/data/matting/*.md > matting_datasets.md
cat ../tools/data/super-resolution/*.md > sr_datasets.md

sed -i 's=(paired-pix2pix/README.md)=(#paired-dataset-for-pix2pix)=g' generation_datasets.md
sed -i 's=(unpaired-cyclegan/README.md)=(#unpaired-dataset-for-cyclegan)=g' generation_datasets.md
sed -i 's=(paris-street-view/README.md)=(#paris-street-view-dataset)=g' inpainting_datasets.md
sed -i 's=(celeba-hq/README.md)=(#celeba-hq-dataset)=g' inpainting_datasets.md
sed -i 's=(places365/README.md)=(#places365-dataset)=g' inpainting_datasets.md
sed -i 's=(comp1k/README.md)=(#composition-1k-dataset)=g' matting_datasets.md
sed -i 's=(div2k/README.md)=(#div2k-dataset)=g' sr_datasets.md
sed -i 's=(reds/README.md)=(#reds-dataset)=g' sr_datasets.md
sed -i 's=(vimeo90k/README.md)=(#vimeo90k-dataset)=g' sr_datasets.md

cat ../tools/data/generation/*/*.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> generation_datasets.md
cat ../tools/data/inpainting/*/*.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> inpainting_datasets.md
cat ../tools/data/matting/*/*.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> matting_datasets.md
cat ../tools/data/super-resolution/*/*.md | sed 's/# Preparing/# /g' | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' >> sr_datasets.md

# merge configs
cat config_*.md | sed "s/#/#&/" >> config.md

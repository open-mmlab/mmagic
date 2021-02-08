#!/usr/bin/env bash

sed -i '$a\\n' ../configs/inpainting/*/*.md
sed -i '$a\\n' ../configs/mattors/*/*.md
sed -i '$a\\n' ../configs/restorers/*/*.md
sed -i '$a\\n' ../configs/synthesizers/*/*.md

# gather models
cat ../configs/inpainting/*/*.md >inpainting_models.md
cat ../configs/mattors/*/*.md >mattors_models.md
cat ../configs/restorers/*/*.md >restorers_models.md
cat ../configs/synthesizers/*/*.md >synthesizers_models.md

sed -i "s/md###t/html#t/g" inpainting_models.md
sed -i "s/md###t/html#t/g" mattors_models.md
sed -i "s/md###t/html#t/g" restorers_models.md
sed -i "s/md###t/html#t/g" synthesizers_models.md

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
sed -i 's=(video90k/README.md)=(#video90k-dataset)=g' sr_datasets.md

cat ../tools/data/generation/*/*.md >> generation_datasets.md
cat ../tools/data/inpainting/*/*.md >> inpainting_datasets.md
cat ../tools/data/matting/*/*.md >> matting_datasets.md
cat ../tools/data/super-resolution/*/*.md >> sr_datasets.md

sed -i 's/# Preparing/# /g' generation_datasets.md
sed -i 's/# Preparing/# /g' inpainting_datasets.md
sed -i 's/# Preparing/# /g' matting_datasets.md
sed -i 's/# Preparing/# /g' sr_datasets.md

sed -i "s/#/#&/" inpainting_models.md
sed -i "s/#/#&/" mattors_models.md
sed -i "s/#/#&/" restorers_models.md
sed -i "s/#/#&/" synthesizers_models.md

sed -i "s/#/#&/" generation_datasets.md
sed -i "s/#/#&/" inpainting_datasets.md
sed -i "s/#/#&/" matting_datasets.md
sed -i "s/#/#&/" sr_datasets.md

# sed -i '1i\# Inpainting Models' inpainting_models.md
# sed -i '1i\# Matting Models' mattors_models.md
# sed -i '1i\# Super-Resolution Models' restorers_models.md
# sed -i '1i\# Generation Models' synthesizers_models.md

sed -i 's/](\/docs\//](/g' inpainting_models.md # remove /docs/ for link used in doc site
sed -i 's/](\/docs\//](/g' mattors_models.md
sed -i 's/](\/docs\//](/g' restorers_models.md
sed -i 's/](\/docs\//](/g' synthesizers_models.md

sed -i 's/](\/docs\//](/g' generation_datasets.md # remove /docs/ for link used in doc site
sed -i 's/](\/docs\//](/g' inpainting_datasets.md
sed -i 's/](\/docs\//](/g' matting_datasets.md
sed -i 's/](\/docs\//](/g' sr_datasets.md

sed -i 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' inpainting_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' mattors_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' restorers_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' synthesizers_models.md

sed -i 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' generation_datasets.md
sed -i 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' inpainting_datasets.md
sed -i 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' matting_datasets.md
sed -i 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' sr_datasets.md

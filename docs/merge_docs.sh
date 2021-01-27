#!/usr/bin/env bash

sed -i '$a\\n' ../configs/inpainting/*/*.md
sed -i '$a\\n' ../configs/mattors/*/*.md
sed -i '$a\\n' ../configs/restorers/*/*.md
sed -i '$a\\n' ../configs/synthesizers/*/*.md

cat ../configs/inpainting/*/*.md >inpainting_models.md
cat ../configs/mattors/*/*.md >mattors_models.md
cat ../configs/restorers/*/*.md >restorers_models.md
cat ../configs/synthesizers/*/*.md >synthesizers_models.md

sed -i "s/#/#&/" inpainting_models.md
sed -i "s/#/#&/" mattors_models.md
sed -i "s/#/#&/" restorers_models.md
sed -i "s/#/#&/" synthesizers_models.md

sed -i "s/md###t/html#t/g" inpainting_models.md
sed -i "s/md###t/html#t/g" mattors_models.md
sed -i "s/md###t/html#t/g" restorers_models.md
sed -i "s/md###t/html#t/g" synthesizers_models.md

sed -i '1i\# Inpainting Models' inpainting_models.md
sed -i '1i\# Matting Models' mattors_models.md
sed -i '1i\# Restoration Models' restorers_models.md
sed -i '1i\# Generation Models' synthesizers_models.md

sed -i 's/](\/docs\//](/g' inpainting_models.md # remove /docs/ for link used in doc site
sed -i 's/](\/docs\//](/g' mattors_models.md
sed -i 's/](\/docs\//](/g' restorers_models.md
sed -i 's/](\/docs\//](/g' synthesizers_models.md

sed -i 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' inpainting_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' mattors_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' restorers_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmediting/tree/master/=g' synthesizers_models.md

# generate all tasks dataset_zoo
cat ../../../tools/dataset_converters/super-resolution/README.md > ../dataset_zoo/1_super_resolution_datasets.md
cat ../../../tools/dataset_converters/inpainting/README.md > ../dataset_zoo/2_inpainting_datasets.md
cat ../../../tools/dataset_converters/matting/README.md > ../dataset_zoo/3_matting_datasets.md
cat ../../../tools/dataset_converters/video-interpolation/README.md > ../dataset_zoo/4_video_interpolation_datasets.md
cat ../../../tools/dataset_converters/unconditional_gans/README.md > ../dataset_zoo/5_unconditional_gans_datasets.md
cat ../../../tools/dataset_converters/image_translation/README.md > ../dataset_zoo/6_image_translation_datasets.md

# generate markdown TOC
sed -i -e 's/](comp1k\(\/README.md)\)/](composition-1k\1/g' ../dataset_zoo/3_matting_datasets.md

sed -i -e 's/](\(.*\)\/README.md)/](#\1-dataset)/g' ../dataset_zoo/1_super_resolution_datasets.md
sed -i -e 's/](\(.*\)\/README.md)/](#\1-dataset)/g' ../dataset_zoo/2_inpainting_datasets.md
sed -i -e 's/](\(.*\)\/README.md)/](#\1-dataset)/g' ../dataset_zoo/3_matting_datasets.md
sed -i -e 's/](\(.*\)\/README.md)/](#\1-dataset)/g' ../dataset_zoo/4_video_interpolation_datasets.md
sed -i -e 's/](\(.*\)\/README.md)/](#\1-dataset)/g' ../dataset_zoo/5_unconditional_gans_datasets.md
sed -i -e 's/](\(.*\)\/README.md)/](#\1-dataset)/g' ../dataset_zoo/6_image_translation_datasets.md

# gather all datasets
cat ../../../tools/dataset_converters/super-resolution/*/README.md | sed 's/# Preparing /\n# /g' | sed "s/#/#&/" >> ../dataset_zoo/1_super_resolution_datasets.md
cat ../../../tools/dataset_converters/inpainting/*/README.md | sed 's/# Preparing /\n# /g' | sed "s/#/#&/"  >> ../dataset_zoo/2_inpainting_datasets.md
cat ../../../tools/dataset_converters/matting/*/README.md | sed 's/# Preparing /\n# /g' | sed "s/#/#&/"  >> ../dataset_zoo/3_matting_datasets.md
cat ../../../tools/dataset_converters/video-interpolation/*/README.md | sed 's/# Preparing /\n# /g' | sed "s/#/#&/"  >> ../dataset_zoo/4_video_interpolation_datasets.md
cat ../../../tools/dataset_converters/unconditional_gans/*/README.md | sed 's/# Preparing /\n# /g' | sed "s/#/#&/"  >> ../dataset_zoo/5_unconditional_gans_datasets.md
cat ../../../tools/dataset_converters/image_translation/*/README.md | sed 's/# Preparing /\n# /g' | sed "s/#/#&/"  >> ../dataset_zoo/6_image_translation_datasets.md

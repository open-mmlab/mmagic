# generate all tasks dataset_zoo
cat ../../../tools/dataset_converters/super-resolution/README.md > dataset_zoo/1_super_resolution_datasets.md
cat ../../../tools/dataset_converters/inpainting/README.md > dataset_zoo/2_inpainting_datasets.md
cat ../../../tools/dataset_converters/matting/README.md > dataset_zoo/3_matting_datasets.md
cat ../../../tools/dataset_converters/video-interpolation/README.md > dataset_zoo/4_video_interpolation_datasets.md
cat ../../../tools/dataset_converters/unconditional_gans/README.md > dataset_zoo/5_unconditional_gans_datasets.md
cat ../../../tools/dataset_converters/image_translation/README.md > dataset_zoo/6_image_translation_datasets.md

# generate markdown TOC
sed -i -e 's/](comp1k\(\/README.md)\)/](composition-1k\1/g' dataset_zoo/3_matting_datasets.md

sed -i -e 's/](\(.*\)\/README.md)/](#\1-dataset)/g' dataset_zoo/1_super_resolution_datasets.md
sed -i -e 's/](\(.*\)\/README.md)/](#\1-dataset)/g' dataset_zoo/2_inpainting_datasets.md
sed -i -e 's/](\(.*\)\/README.md)/](#\1-dataset)/g' dataset_zoo/3_matting_datasets.md
sed -i -e 's/](\(.*\)\/README.md)/](#\1-dataset)/g' dataset_zoo/4_video_interpolation_datasets.md
sed -i -e 's/](\(.*\)\/README.md)/](#\1-dataset)/g' dataset_zoo/5_unconditional_gans_datasets.md
sed -i -e 's/](\(.*\)\/README.md)/](#\1-dataset)/g' dataset_zoo/6_image_translation_datasets.md

# gather all datasets
cat ../../../tools/dataset_converters/super-resolution/*/README.md | sed 's/# Preparing /\n# /g' | sed "s/#/#&/" >> dataset_zoo/1_super_resolution_datasets.md
cat ../../../tools/dataset_converters/inpainting/*/README.md | sed 's/# Preparing /\n# /g' | sed "s/#/#&/"  >> dataset_zoo/2_inpainting_datasets.md
cat ../../../tools/dataset_converters/matting/*/README.md | sed 's/# Preparing /\n# /g' | sed "s/#/#&/"  >> dataset_zoo/3_matting_datasets.md
cat ../../../tools/dataset_converters/video-interpolation/*/README.md | sed 's/# Preparing /\n# /g' | sed "s/#/#&/"  >> dataset_zoo/4_video_interpolation_datasets.md
cat ../../../tools/dataset_converters/unconditional_gans/*/README.md | sed 's/# Preparing /\n# /g' | sed "s/#/#&/"  >> dataset_zoo/5_unconditional_gans_datasets.md
cat ../../../tools/dataset_converters/image_translation/*/README.md | sed 's/# Preparing /\n# /g' | sed "s/#/#&/"  >> dataset_zoo/6_image_translation_datasets.md

echo '# Overview' > dataset_zoo/0_overview.md
echo '\n- [Prepare Super-Resolution Datasets](./1_super_resolution_datasets.md)' >> dataset_zoo/0_overview.md
cat dataset_zoo/1_super_resolution_datasets.md | grep -oP '(- \[.*-dataset.*)' | sed 's/- \[/  - \[/g' | sed 's/(#/(.\/1_super_resolution_datasets.md#/g' >> dataset_zoo/0_overview.md
echo '\n- [Prepare Inpainting Datasets](./2_inpainting_datasets.md)' >> dataset_zoo/0_overview.md
cat dataset_zoo/2_inpainting_datasets.md | grep -oP '(- \[.*-dataset.*)' | sed 's/- \[/  - \[/g' | sed 's/(#/(.\/2_inpainting_datasets.md#/g' >> dataset_zoo/0_overview.md
echo '\n- [Prepare Matting Datasets](./3_matting_datasets.md)\n' >> dataset_zoo/0_overview.md
cat dataset_zoo/3_matting_datasets.md | grep -oP '(- \[.*-dataset.*)' | sed 's/- \[/  - \[/g' | sed 's/(#/(.\/3_matting_datasets.md#/g' >> dataset_zoo/0_overview.md
echo '\n- [Prepare Video Frame Interpolation Datasets](./4_video_interpolation_datasets.md)' >> dataset_zoo/0_overview.md
cat dataset_zoo/4_video_interpolation_datasets.md | grep -oP '(- \[.*-dataset.*)' | sed 's/- \[/  - \[/g' | sed 's/(#/(.\/4_video_interpolation_datasets.md#/g' >> dataset_zoo/0_overview.md
echo '\n- [Prepare Unconditional GANs Datasets](./5_unconditional_gans_datasets.md)' >> dataset_zoo/0_overview.md
cat dataset_zoo/5_unconditional_gans_datasets.md | grep -oP '(- \[.*-dataset.*)' | sed 's/- \[/  - \[/g' | sed 's/(#/(.\/5_unconditional_gans_datasets.md#/g' >> dataset_zoo/0_overview.md
echo '\n- [Prepare Image Translation Datasets](./6_image_translation_datasets.md)' >> dataset_zoo/0_overview.md
cat dataset_zoo/6_image_translation_datasets.md | grep -oP '(- \[.*-dataset.*)' | sed '$a\n'  |sed 's/- \[/  - \[/g' | sed 's/(#/(.\/6_image_translation_datasets.md#/g' >> dataset_zoo/0_overview.md

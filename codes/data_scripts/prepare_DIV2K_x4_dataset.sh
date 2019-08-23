

echo "Prepare DIV2K X4 datasets..."
cd ../../datasets
mkdir DIV2K
cd DIV2K

#### Step 1
echo "Step 1: Download the datasets: [DIV2K_train_HR] and [DIV2K_train_LR_bicubic_X4]..."
# GT
FOLDER=DIV2K_train_HR
FILE=DIV2K_train_HR.zip
if [ ! -d "$FOLDER" ]; then
    if [ ! -f "$FILE" ]; then
        echo "Downloading $FILE..."
        wget http://data.vision.ee.ethz.ch/cvl/DIV2K/$FILE
    fi
    unzip $FILE
fi
# LR
FOLDER=DIV2K_train_LR_bicubic
FILE=DIV2K_train_LR_bicubic_X4.zip
if [ ! -d "$FOLDER" ]; then
    if [ ! -f "$FILE" ]; then
        echo "Downloading $FILE..."
        wget http://data.vision.ee.ethz.ch/cvl/DIV2K/$FILE
    fi
    unzip $FILE
fi

#### Step 2
echo "Step 2: Rename the LR images..."
cd ../../codes/data_scripts
python rename.py

#### Step 4
echo "Step 4: Crop to sub-images..."
python extract_subimages.py

#### Step 5
echo "Step5: Create LMDB files..."
python create_lmdb.py

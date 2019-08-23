import os
import glob


def main():
    folder = '../../datasets/DIV2K/DIV2K_train_LR_bicubic/X4'
    DIV2K(folder)
    print('Finished.')


def DIV2K(path):
    img_path_l = glob.glob(os.path.join(path, '*'))
    for img_path in img_path_l:
        new_path = img_path.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')
        os.rename(img_path, new_path)


if __name__ == "__main__":
    main()
import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch


def main():
    # 设置随机数种子
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--image_size', default=[2720, 3200], type=int, help='the image size (height, width)')
    parser.add_argument('--patch_size', default=256, type=int, help='training images crop size')
    parser.add_argument('--root_dir', default='/mnt/datadisk0/cgy/Datasets/LGC', help='Datasets root directory')

    opt = parser.parse_args()
    IMAGE_SIZE = opt.image_size
    PATCH_SIZE = opt.patch_size

    train_dates = ['2005_093_Apr03', '2005_045_Feb14',
                   '2005_029_Jan29', '2004_123_May02',
                   '2004_299_Oct25', '2005_013_Jan13',
                   '2004_235_Aug22', '2004_107_Apr16',
                   '2004_187_Jul05', '2005_061_Mar02',
                   '2004_219_Aug06']

    # split the whole image into several patches
    PATCH_STRIDE = PATCH_SIZE // 2
    end_h = (IMAGE_SIZE[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    end_w = (IMAGE_SIZE[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
    w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]

    if (IMAGE_SIZE[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        h_index_list.append(IMAGE_SIZE[0] - PATCH_SIZE)
    if (IMAGE_SIZE[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        w_index_list.append(IMAGE_SIZE[1] - PATCH_SIZE)

    total_index = 0
    # path where the training images saved in
    output_dir = '/mnt/datadisk0/cgy/Datasets/LGC_Train'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save all the train images into one numpy array
    total_original_images = np.zeros((len(train_dates), 2, 6, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    for k in tqdm(range(len(train_dates))):
        cur_date = train_dates[k]
        target_dir = opt.root_dir + '/' + cur_date
        for filename in os.listdir(target_dir):
            if filename[:3] != 'MOD':
                path = os.path.join(target_dir, filename)
                total_original_images[k, 1] = np.load(path)
            else:
                path = os.path.join(target_dir, filename)
                total_original_images[k, 0] = np.load(path)

    for k in tqdm(range(len(train_dates))):
        for i in range(len(h_index_list)):
            for j in range(len(w_index_list)):
                h_start = h_index_list[i]
                w_start = w_index_list[j]

                ref_index = k
                while ref_index == k:
                    ref_index = np.random.choice(len(train_dates))

                images = []
                images.append(total_original_images[k, 0])
                images.append(total_original_images[k, 1])
                images.append(total_original_images[ref_index, 0])
                images.append(total_original_images[ref_index, 1])

                input_images = []
                for im in images:
                    input_images.append(im[:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE])
                input_images = np.concatenate(input_images, axis=0)
                # save the patch for training
                np.save(os.path.join(output_dir, str(total_index) + '.npy'), input_images)

                total_index += 1

    assert total_index == len(train_dates) * len(h_index_list) * len(w_index_list)


if __name__ == '__main__':
    main()
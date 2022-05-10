import os
import random
import argparse
import rasterio
import numpy as np
from tqdm import tqdm
from sewar import rmse, ssim, sam

import torch

from datasets import load_image_pair, transform_image
from models import SwinSTFM


def uiqi(im1, im2, block_size=64, return_map=False):
    if len(im1.shape)==3:
        return np.array([uiqi(im1[:,:,i], im2[:,:,i], block_size, return_map=return_map) for i in range(im1.shape[2])])
    delta_x = np.std(im1, ddof=1)
    delta_y = np.std(im2, ddof=1)
    delta_xy = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2))) / (im1.shape[0] * im1.shape[1] - 1)
    mu_x = np.mean(im1)
    mu_y = np.mean(im2)
    q1 = delta_xy / (delta_x * delta_y)
    q2 = 2 * mu_x * mu_y / (mu_x ** 2 + mu_y ** 2)
    q3 = 2 * delta_x * delta_y / (delta_x ** 2 + delta_y ** 2)
    q = q1 * q2 * q3
    return q


def test(opt, model, test_dates, IMAGE_SIZE, PATCH_SIZE):
    cur_result = {}
    model.eval()

    PATCH_STRIDE = PATCH_SIZE // 2
    end_h = (IMAGE_SIZE[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    end_w = (IMAGE_SIZE[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
    w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]
    if (IMAGE_SIZE[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        h_index_list.append(IMAGE_SIZE[0] - PATCH_SIZE)
    if (IMAGE_SIZE[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        w_index_list.append(IMAGE_SIZE[1] - PATCH_SIZE)

    total_image = 0
    for cur_date in test_dates:
        cur_day = int(cur_date.split('_')[1])
        if cur_day == 347:
            for ref_date in test_dates:
                if ref_date != cur_date:
                    images = load_image_pair(opt.root_dir, cur_date, ref_date)

                    output_image = np.zeros(images[1].shape)
                    image_mask = np.ones(images[1].shape)
                    for i in range(4):
                        negtive_mask = np.where(images[i] < 0)
                        inf_mask = np.where(images[i] > 10000.)
                        image_mask[negtive_mask] = 0
                        image_mask[inf_mask] = 0

                    for i in range(len(h_index_list)):
                        for j in range(len(w_index_list)):
                            h_start = h_index_list[i]
                            w_start = w_index_list[j]

                            input_lr = images[0][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                            target_hr = images[1][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                            ref_lr = images[2][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                            ref_hr = images[3][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]

                            flip_num = 0
                            rotate_num0 = 0
                            rotate_num = 0
                            input_lr, im_mask = transform_image(input_lr, flip_num, rotate_num0, rotate_num)
                            target_hr, im_mask = transform_image(target_hr, flip_num, rotate_num0, rotate_num)
                            ref_lr, im_mask = transform_image(ref_lr, flip_num, rotate_num0, rotate_num)
                            ref_hr, im_mask = transform_image(ref_hr, flip_num, rotate_num0, rotate_num)

                            input_lr = input_lr.unsqueeze(0).cuda()
                            ref_lr = ref_lr.unsqueeze(0).cuda()
                            ref_hr = ref_hr.unsqueeze(0).cuda()

                            output = model(ref_lr, ref_hr, input_lr)
                            output = output.squeeze()

                            h_end = h_start + PATCH_SIZE
                            w_end = w_start + PATCH_SIZE
                            cur_h_start = 0
                            cur_h_end = PATCH_SIZE
                            cur_w_start = 0
                            cur_w_end = PATCH_SIZE

                            if i != 0:
                                h_start = h_start + PATCH_SIZE // 4
                                cur_h_start = PATCH_SIZE // 4

                            if i != len(h_index_list) - 1:
                                h_end = h_end - PATCH_SIZE // 4
                                cur_h_end = cur_h_end - PATCH_SIZE // 4

                            if j != 0:
                                w_start = w_start + PATCH_SIZE // 4
                                cur_w_start = PATCH_SIZE // 4

                            if j != len(w_index_list) - 1:
                                w_end = w_end - PATCH_SIZE // 4
                                cur_w_end = cur_w_end - PATCH_SIZE // 4

                            output_image[:, h_start: h_end, w_start: w_end] = \
                                output[:, cur_h_start: cur_h_end, cur_w_start: cur_w_end].cpu().detach().numpy()

                    real_im = images[1] * 0.0001 * image_mask
                    real_output = (output_image + 1) * 0.5 * image_mask

                    for real_predict in [real_output]:
                        cur_result['rmse'] = []
                        cur_result['ssim'] = []
                        cur_result['cc'] = []
                        cur_result['uiqi'] = []
                        cur_result['ergas'] = 0

                        for i in range(6):
                            cur_result['rmse'].append(rmse(real_im[i], real_predict[i]))
                            cur_result['ssim'].append(ssim(real_im[i], real_predict[i], MAX=1.0)[0])
                            cur_result['uiqi'].append(uiqi(real_im[i], real_predict[i]))
                            cur_cc = np.sum((real_im[i] - np.mean(real_im[i])) * (real_predict[i] - np.mean(real_predict[i]))) / \
                                     np.sqrt((np.sum(np.square(real_im[i] - np.mean(real_im[i])))) * np.sum(
                                         np.square(real_predict[i] - np.mean(real_predict[i]))) + 1e-100)
                            cur_result['cc'].append(cur_cc)

                            cur_result['ergas'] += rmse(real_im[i], real_predict[i]) ** 2 / (np.mean(real_im[i]) ** 2 + 1e-100)

                        cur_result['ergas'] = np.sqrt(cur_result['ergas'] / 6.) * 6

                        cur_im = real_im * 10000.
                        cur_predict = real_predict * 10000.

                        cur_result['sam'] = sam(cur_im.transpose(1, 2, 0), cur_predict.transpose(1, 2, 0)) * 180 / np.pi
                        print('[%s/%s] RMSE: %.4f SSIM: %.4f UIQI: %.4f CC: %.4f ERGAS: %.4f SAM: %.4f' % (
                            cur_date, ref_date, np.mean(np.array(cur_result['rmse'])),
                            np.mean(np.array(cur_result['ssim'])), np.mean(np.array(cur_result['uiqi'])),
                            np.mean(np.array(cur_result['cc'])), cur_result['ergas'], cur_result['sam']
                        ))
                        ref_day = int(ref_date.split('_')[1])
                        total_image += cur_predict
                        if ref_day != 363:
                            final_im = cur_predict.astype(np.int16)
                            metadata = {
                                'driver': 'GTiff',
                                'width': final_im.shape[2],
                                'height': final_im.shape[1],
                                'count': final_im.shape[0],
                                'dtype': np.int16
                            }
                            save_dir = '/data/cgy/ParalSTFM/paper_images'
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            im_name = os.path.join(save_dir, 'swin_LGC.tif')
                            assert final_im.ndim == 2 or final_im.ndim == 3
                            with rasterio.open(im_name, mode='w', **metadata) as dst:
                                if final_im.ndim == 3:
                                    for i in range(final_im.shape[0]):
                                        dst.write(final_im[i], i + 1)
                                else:
                                    dst.write(final_im, 1)
    final_im = (total_image / 2.).astype(np.int16)
    metadata = {
        'driver': 'GTiff',
        'width': final_im.shape[2],
        'height': final_im.shape[1],
        'count': final_im.shape[0],
        'dtype': np.int16
    }
    save_dir = '/data/cgy/ParalSTFM/paper_images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    im_name = os.path.join(save_dir, 'swinfuse_LGC.tif')
    assert final_im.ndim == 2 or final_im.ndim == 3
    with rasterio.open(im_name, mode='w', **metadata) as dst:
        if final_im.ndim == 3:
            for i in range(final_im.shape[0]):
                dst.write(final_im[i], i + 1)
        else:
            dst.write(final_im, 1)


def train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE):

    model_G = SwinSTFM()
    G_dict = model_G.state_dict()
    model_CKPT = torch.load('LGC_best.pth')
    pretained_dict = {k: v for k, v in model_CKPT.items() if k in G_dict}

    G_dict.update(pretained_dict)
    model_G.load_state_dict(G_dict)
    model_G.cuda()
    test(opt, model_G, test_dates, IMAGE_SIZE, PATCH_SIZE)



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
    parser.add_argument('--num_epochs', default=60, type=int, help='train epoch number')
    parser.add_argument('--root_dir', default='/mnt/datadisk0/cgy/Datasets/LGC', help='Datasets root directory')
    parser.add_argument('--train_dir', default='/mnt/datadisk0/cgy/Datasets/LGC_Train', help='Datasets train directory')

    opt = parser.parse_args()
    IMAGE_SIZE = opt.image_size
    PATCH_SIZE = opt.patch_size
    NUM_EPOCHS = opt.num_epochs

    # 加载LGC数据集
    train_dates = []
    test_dates = []
    for dir_name in os.listdir(opt.root_dir):
        cur_day = int(dir_name.split('_')[1])
        if cur_day not in [331, 347, 363]:
            train_dates.append(dir_name)
        else:
            test_dates.append(dir_name)

    train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE)


if __name__ == '__main__':
    main()



import os
import math
import argparse
import logging
import torch
import numpy as np
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from tqdm import tqdm

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdir_and_rename(opt['path']['results_root'])
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                 and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers
    util.setup_logger('base', opt['path']['log'], 'test', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # create test dataset and dataloader
    test_set = create_dataset(opt['datasets']['test'])
    test_loader = create_dataloader(test_set, opt['datasets']['test'])
    logger.info('Number of test images: {:d}'.format(len(test_set)))

    # create model
    model = create_model(opt)

    # testing
    logger.info('Start testing...')
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_mse = 0.0
    avg_rmse = 0.0

    for idx, test_data in enumerate(tqdm(test_loader, desc='Testing')):
        img_name = os.path.splitext(os.path.basename(test_data['LR_path'][0]))[0]
        img_dir = os.path.join(opt['path']['results_root'], img_name)
        util.mkdir(img_dir)

        model.feed_data(test_data)
        model.test()

        visuals = model.get_current_visuals()
        sr_img = util.tensor2img(visuals['SR'])  # uint8
        gt_img = util.tensor2img(visuals['HR'])  # uint8

        # Save SR images for reference
        save_img_path = os.path.join(img_dir, '{:s}_SR.png'.format(img_name))
        util.save_img(sr_img, save_img_path)

        # calculate PSNR, SSIM, MSE, and RMSE
        crop_size = opt['scale']
        gt_img = gt_img / 255.
        sr_img = sr_img / 255.
        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]

        mse = ((cropped_sr_img - cropped_gt_img) ** 2).mean()
        rmse = math.sqrt(mse)
        psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
        ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)

        avg_mse += mse
        avg_rmse += rmse
        avg_psnr += psnr
        avg_ssim += ssim

        logger.info('Test image {:s}: MSE: {:.4f}, RMSE: {:.4f}, PSNR: {:.4f} dB, SSIM: {:.4f}'.format(
            img_name, mse, rmse, psnr, ssim))

    avg_mse /= len(test_loader)
    avg_rmse /= len(test_loader)
    avg_psnr /= len(test_loader)
    avg_ssim /= len(test_loader)

    # log
    logger.info('Average MSE: {:.4f}, Average RMSE: {:.4f}, Average PSNR: {:.4f} dB, Average SSIM: {:.4f}'.format(
        avg_mse, avg_rmse, avg_psnr, avg_ssim))

    logger.info('End of testing.')

if __name__ == '__main__':
    main()
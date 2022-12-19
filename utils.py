import os
import torch
import random
import numpy as np
import SimpleITK as sitk
from metric import MSE, calc_tre


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_landmarks(landmark_dir):
    landmark_folder = landmark_dir
    landmarks = []
    for i in sorted(
            [os.path.join(landmark_folder, file) for file in os.listdir(landmark_folder) if file.endswith('.pt')]):
        landmarks.append(torch.load(i))

    return landmarks


def save_image(img, ref_img, save_path, save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img = sitk.GetImageFromArray(img.cpu().detach().numpy())
    ref_img = sitk.GetImageFromArray(ref_img.cpu().detach().numpy())

    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(save_path, save_name))


def save_model(args, model, optimizer, scheduler, time):
    model_checkpoint = os.path.join(args.model_dir,
                                    "{}_size{}_lr{}.pth".format(time, args.size, args.lr))
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None
    }
    torch.save(checkpoint, model_checkpoint)
    print("Saved model checkpoint to [DIR: %s]", args.model_dir)


def get_test_photo_loss(args, logger, model, test_loader):
    with torch.no_grad():
        model.eval()
        losses = []
        for batch, (moving, fixed, landmarks) in enumerate(test_loader):
            m_img = moving[0].to('cuda').float()
            f_img = fixed[0].to('cuda').float()

            landmarks00 = landmarks['landmark_00'].squeeze().cuda()
            # landmarks50 = landmarks['landmark_50'].squeeze().cuda()

            warped_image, flow = model(m_img, f_img, True)
            # warped_image, flow = model(m_img, f_img)
            flow_hr = flow[0]
            index = batch + 1

            crop_range = args.dirlab_cfg[index]['crop_range']

            # TRE
            _mean, _std = calc_tre(flow_hr, landmarks00 - torch.tensor(
                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1, 3).cuda(),
                                   landmarks['disp_00_50'].squeeze(), args.dirlab_cfg[index]['pixel_spacing'])

            # MSE
            _mse = MSE(f_img, warped_image)

            losses.append([_mean.item(), _std.item(), _mse.item()])

            logger.info('case=%d after warped, TRE=%.5f+-%.5f' % (index, _mean.item(), _std.item()))

        # loss = np.mean(losses)
        # print('mean loss=%.5f' % (loss))
        # show_results(net, test_loader, epoch, 2)
        return losses

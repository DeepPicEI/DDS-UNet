import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.DDS_Unet import DDS_Unet
from nets.net_training import weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils_fit import fit_one_epoch
from sklearn.model_selection import KFold

if __name__ == "__main__":
    Cuda = True
    num_classes = 2
    pretrained = False
    input_shape = [320, 320]
    Epoch = 100
    batch_size = 2
    lr = 1e-5
    VOCdevkit_path = 'VOCdevkit'
    dice_loss = False
    focal_loss = True
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 4
    save_dir = 'logs'
    eval_flag = True
    eval_period = 1
    model = DDS_Unet(embed_dim=96,
                     patch_height=4,
                     patch_width=4,
                     class_num=num_classes).train()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # if not pretrained:
    #     weights_init(model)
    # if model_path != '':
    #     print('Load weights {}.'.format(model_path))
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load(model_path, map_location=device)
    #     # for k, v in pretrained_dict.items():
    #     #     if np.shape(model_dict[k]) == np.shape(v):
    #     #         pretrained_dict = {k:v}
    #
    #     # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    #     # model_dict.update(pretrained_dict)
    #     # model.load_state_dict(model_dict)

    model_train = model.train()
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    # if Cuda:
    #     model_train = torch.nn.DataParallel(model)
    #     cudnn.benchmark = True
    #     model_train = model_train.cuda()

    if Cuda:
        model_train = model_train.to(device)

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir)
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    All_lines = train_lines + val_lines
    kf = KFold(n_splits=5, shuffle=False)

    if True:
        batch_size = batch_size
        lr = lr
        start_epoch = 0
        end_epoch = Epoch
        train_lines_new = []
        val_lines_new = []
        optimizer = optim.Adam(model_train.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        for train_index, val_index in kf.split(All_lines):
            train_lines_new.append(np.array(All_lines)[train_index].tolist())
            train_lines_new = train_lines_new[0]
            val_lines_new.append(np.array(All_lines)[val_index].tolist())
            val_lines_new = val_lines_new[0]
            epoch_step = len(train_lines_new) // batch_size
            epoch_step_val = len(val_lines_new) // batch_size
            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

            train_dataset = UnetDataset(train_lines_new, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset = UnetDataset(val_lines_new, input_shape, num_classes, False, VOCdevkit_path)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate)
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)

            for epoch in range(start_epoch, end_epoch):
                fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                              epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights,
                              num_classes)
                lr_scheduler.step()

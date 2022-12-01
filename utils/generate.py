from .deepfool import deepfool
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torch.backends.cudnn as cudnn
import tensorflow as tf
from .util import ImageNet_datastream, data_process

def project_lp(v, xi, p):

    if p == 2:
        temp = torch.norm(v.view(-1))
        v = v * xi / temp
    elif p == np.inf:
        v = torch.clamp(v, -xi, xi)
        # v=torch.sign(v)*torch.min(abs(v), xi)
    else:
        raise ValueError("Values of a different from 2 and Inf are currently not surpported...")

    return v


def generate(dataset, batch_size, test_batchsize, net, img_size=224, device_ids=[0],
             delta=0.2, max_iter_uni=10, xi=10, p=np.inf, num_classes=1000, overshoot=0.2, max_iter_df=20):
    '''

    :param path:
    :param dataset:
    :param testset:
    :param net:
    :param delta:
    :param max_iter_uni:
    :param p:
    :param num_class:
    :param overshoot:
    :param max_iter_df:
    :return:
    '''
    train_dir = '/data/jiaming/datasets/' + dataset + '/train'
    test_dir = '/data/jiaming/datasets/' + dataset + '/test'
    net.eval()
    if torch.cuda.is_available():
        device = 'cuda'
        net.cuda()
        cudnn.benchmark = True
    else:
        device = 'cpu'
    train_loader, test_loader, num_classes, _ = data_process(dataset=dataset,
                                                             train_dir=train_dir,
                                                             test_dir=test_dir,
                                                             batch_size=batch_size,
                                                             device_ids=device_ids,
                                                             img_size=img_size,
                                                             is_train=True)
    v = torch.zeros([3, 224, 224]).to(device)
    fooling_rate = 0.0
    iter = 0

    # start an epochi

    for i, (img_trn, labels) in enumerate(train_loader):
        num_img_trn = len(img_trn)
        while fooling_rate < 1-delta and i < max_iter_uni:
            order = np.arange(num_img_trn)
            np.random.shuffle(order)
            print("Starting pass number ", iter)
            for k in order:
                # cur_img = Image.open(img_trn[k][0]).convert('RGB')
                cur_img1 = img_trn[k].unsqueeze(0).to(device)
                r2 = int(net(cur_img1).max(1)[1])
                torch.cuda.empty_cache()
                # per_img = Image.fromarray(cut(cur_img)+v.astype(np.uint8))
                per_img = torch.clamp(cur_img1.squeeze(0) + v, -1, 1)
                per_img1 = per_img.unsqueeze(0)
                r1 = int(net(per_img1).max(1)[1])
                torch.cuda.empty_cache()

                if r1 == r2:
                    print(">> k =", np.where(k==order)[0][0], ', pass #\n', iter, end='      ')
                    dr, iter_k, label, k_i, pert_image = deepfool(per_img1[0], net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)
                    dr = torch.from_numpy(dr).to(device)
                    if iter_k < max_iter_df-1:

                        v[0, :, :] += dr[0, 0, :, :]
                        v[0, :, :] += dr[0, 1, :, :]
                        v[0, :, :] += dr[0, 2, :, :]
                        v = project_lp(v, xi, p)

            iter = iter + 1

            with torch.no_grad():
                num_val = 50000
                # Compute fooling_rate
                est_labels_orig = torch.zeros(num_val).to(device)
                est_labels_pert = torch.zeros(num_val).to(device)
                est_labels_true = torch.zeros(num_val).to(device)

                for batch_idx, (img_test, labels) in enumerate(test_loader):
                    inputs = img_test.to(device)
                    outputs = net(inputs)
                    _, predicted = outputs.max(1)
                    est_labels_orig[batch_idx * test_batchsize: (batch_idx + 1) * test_batchsize] = \
                        predicted
                    est_labels_true[batch_idx * test_batchsize: (batch_idx + 1) * test_batchsize] = \
                        labels.to(device)

                    img_test = torch.clamp(img_test.to(device) + v.unsqueeze(0), -1, 1)
                    inputs = img_test
                    outputs = net(inputs)
                    _, predicted = outputs.max(1)
                    est_labels_pert[batch_idx * test_batchsize: (batch_idx + 1) * test_batchsize] = \
                        predicted

                fooling_rate = float(torch.sum(est_labels_orig != est_labels_pert)) / float(num_val)
                print("FOOLING RATE: ", fooling_rate)
                true_ori_rate = float(torch.sum(est_labels_orig != est_labels_true)) / float(num_val)
                print("RATE between true and ori: ", true_ori_rate)
                true_adv_rate = float(torch.sum(est_labels_true != est_labels_pert)) / float(num_val)
                print("RATE between true and adv: ", true_adv_rate)
                torch.save(v.cpu(), './data/uap_floder/v_{}_{:.4f}.pth'.format(iter, fooling_rate))

    return v

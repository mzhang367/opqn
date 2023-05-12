# coding: utf-8
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
import sys
import errno
import os.path as osp
from collections import defaultdict
import time
from scipy.spatial.distance import pdist
from shutil import copyfile
import random


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def get_mean_and_std(transform, dataset):

    '''
    Compute the mean and std value of dataset
    Now enable GPU acceleration
    '''
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, drop_last=True)
    mean = torch.zeros(3).cuda()
    std = torch.zeros(3).cuda()
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        inputs = inputs.cuda()
        inputs_transform = transform(inputs)
        for i in range(3):
            mean[i] += inputs_transform[:,i,:,:].mean()
            std[i] += inputs_transform[:,i,:,:].std()
    mean.div_(len(dataloader))
    std.div_(len(dataloader))
    print(mean, std)
    return mean, std


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def normalize_image(a, mean_value, std_value):
    for i in range(3):
        a[:, i, :, :] -= mean_value[i]
        a[:, i, :, :] /= (std_value[i] + 1e-6)
    return a


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x = np.clip(x, -1, 1)
    return x


def list_images(basePath, contains=None):
    image_types = (".jpg")
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def CalcHammingDist(B1, B2):    # test * train
    q = B2.shape[1]
    distH = 0.5 * (q - torch.mm(B1, B2.transpose(0, 1)))    # the same
    return distH


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def compute_mAP(trn_binary, tst_binary, trn_label, tst_label, device, use_gpu=True, top=None):
    AP = []
    top_p = []
    top_mAP = 0
    percent = 1
    for i in range(tst_binary.size(0)):
        _, query_result = torch.sum((tst_binary[i]!= trn_binary).long(), dim=1).sort()  # broadcast, return tuple of (values, indices)
        correct = (tst_label[i] == trn_label[query_result]).float()
        N = torch.sum(correct)
        if use_gpu:
            Ns = torch.arange(1, N+1).float().to(device)
        else:
            Ns = torch.arange(1, N+1).float()
        index = (torch.nonzero(correct, as_tuple=False) + 1)[:, 0].squeeze().float()
        AP.append(torch.mean(Ns / index))
        if top is not None:
            top_result = query_result[:top]
            top_correct = (tst_label[i] == trn_label[top_result]).float()#boolean --> float?
            N_top = torch.sum(top_correct)
            top_p.append(1.0 * N_top/top)
        if (i + 1) % 13950 == 0:
            print("%d%% completed" % (10*percent))
            percent+=1

    top_mAP = torch.mean(torch.Tensor(top_p).cuda())

    mAP = torch.mean(torch.Tensor(AP).cuda())
    return mAP, top_mAP


def compute_result(transform, dataloader, net, device):
    bs = []
    label = []
    for i, (imgs, cls, *_) in enumerate(dataloader):
        imgs = imgs.to(device)
        imgs = transform(imgs)
        hash_values = net(imgs)
        bs.append(hash_values.data)
        label.append(cls.to(device))

    return torch.sign(torch.cat(bs)), torch.cat(label)


def compute_result_real(transform, dataloader, net, device):
    bs = []
    label = []
    for i, (imgs, cls, *_) in enumerate(dataloader):
        imgs = imgs.to(device)
        imgs = transform(imgs)
        hash_values = net(imgs)
        bs.append(hash_values.data)
        label.append(cls.to(device))

    return F.normalize(torch.cat(bs), p=2, dim=1), torch.cat(label)


def compute_result_hflip(transform, dataloader, net, device):
    bs = []
    bs_flip = []
    label = []
    for i, (imgs, cls, *_) in enumerate(dataloader):
        imgs = imgs.to(device)
        imgs = transform(imgs)

        imgs_flip = TF.hflip(imgs)
        imgs_flip = transform(imgs_flip)

        outputs = net(imgs)
        outputs_flip = net(imgs_flip)
        bs.append(outputs.data)
        bs_flip.append(outputs_flip.data)
        label.append(cls.to(device))

    real_fratures = torch.cat([torch.cat(bs), torch.cat(bs_flip)], dim=1)

    norm_features = F.normalize(real_fratures, p=2, dim=1)

    # print(norm_features.shape)

    return norm_features, torch.cat(label)


def evaluation_euclidean_distance(train_labels, test_labels, train_bits, test_bits, device, top=None):

    num_test = test_bits.size(0)
    AP = []
    top_p = []

    percent = 1
    for j in range(num_test):

        distE = torch.sum(test_bits[j, :]**2) + torch.sum(train_bits**2, dim=1) - 2 * test_bits[j, :] @ train_bits.t()
        _, sort_result = distE.sort()
        correct = (test_labels[j] == train_labels[sort_result]).float()
        N = torch.sum(correct)
        if device == "cuda:0":
            Ns = torch.arange(1, N+1).float().to(device) # the index in nonzero elements
        else:
            Ns = torch.arange(1, N+1).float()
        index = (torch.nonzero(correct, as_tuple=False) + 1)[:, 0].squeeze().float()
        # index = (correct.nonzero() + 1)[:, 0:1].squeeze(dim=1).float()
        AP.append(torch.mean(Ns / index)) # the index of the whole vector
        if top is not None:
            top_result = sort_result[:top]
            top_correct = (test_labels[j] == train_labels[top_result]).float()#boolean --> float?
            N_top = torch.sum(top_correct)
            top_p.append(1.0*N_top/top)


    top_mAP = torch.mean(torch.Tensor(top_p).cuda())

    mAP = torch.mean(torch.Tensor(AP).cuda())

    if top is not None:
        print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%% \n' % (100. * float(mAP), 100. * float(top_mAP)))
    else:
        print('[Evaluate Phase] MAP: %.2f%% ' % (100. * float(mAP)))


def eval_cos_dst(train_labels, test_labels, train_features, test_features, device, top=None):

    """
    Both train and test features are l_2 normalized before calculating cosine distance/ dot product
    :return:
    """
    num_test = test_features.size(0)
    AP = []
    top_p = []

    percent = 1
    for j in range(num_test):

        distE = test_features[j, :] @ train_features.t()
        _, sort_result = distE.sort(descending=True)
        correct = (test_labels[j] == train_labels[sort_result]).float()
        N = torch.sum(correct)
        if device == "cuda:0":
            Ns = torch.arange(1, N+1).float().to(device) # the index in nonzero elements
        else:
            Ns = torch.arange(1, N+1).float()
        index = (torch.nonzero(correct, as_tuple=False) + 1)[:, 0].squeeze().float()
        # index = (correct.nonzero() + 1)[:, 0:1].squeeze(dim=1).float()
        AP.append(torch.mean(Ns / index))   # the index of the whole vector
        if top is not None:
            top_result = sort_result[:top]
            top_correct = (test_labels[j] == train_labels[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p.append(1.0*N_top/top)

    top_mAP = torch.mean(torch.Tensor(top_p).cuda())

    mAP = torch.mean(torch.Tensor(AP).cuda())

    if top is not None:
        print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%% \n' % (100. * float(mAP), 100. * float(top_mAP)))
    else:
        print('[Evaluate Phase] MAP: %.2f%% ' % (100. * float(mAP)))


def EncodingOnehot(target, nclasses):
    target_onehot = torch.Tensor(target.size(0), nclasses).cuda()
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot


def compute_topK(trn_binary, tst_binary, trn_label, tst_label, device, top_list):

    top_p = torch.Tensor(tst_binary.size(0), len(top_list)).to(device)

    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()  # broadcast, return tuple of (values, indices)
        for j, top in enumerate(top_list):
            top_result = query_result[:top]
            top_correct = (query_label == trn_label[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p[i, j] = 1.0*N_top/top

    top_pres = top_p.mean(dim=0).cpu().numpy()

    return top_pres


def draw_topK(trainset, trn_binary, tst_binary, trn_label, tst_label, device, top):

    top_p = torch.Tensor(tst_binary.size(0)).to(device)
    collect_0 = []
    collect_1 = []
    collect_2 = []
    dict0 = defaultdict(list)
    dict1 = defaultdict(list)
    dict2 = defaultdict(list)

    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()  # broadcast, return tuple of (values, indices)
        top_result = query_result[:top]
        top_correct = (query_label == trn_label[top_result]).float()
        N_top = torch.sum(top_correct)
        top_p[i] = 1.0*N_top/top
        if top_p[i] == 1:
            collect_0.append(i)
            for j in range(top):
                dict0[i].append(trainset.imgs[top_result[j]][0])
        elif top_p[i] == 0.8:
            collect_1.append(i)
            for j in range(top):
                dict1[i].append(trainset.imgs[top_result[j]][0])
        elif top_p[i] == 0.6:
            collect_2.append(i)
            for j in range(top):
                dict2[i].append(trainset.imgs[top_result[j]][0])

    # dict0[0]= trainset.imgs[top_result][0]
    #  build three dict and three lists.
    top_pres = top_p.mean().cpu().numpy()
    print(top_pres)

    return collect_0, collect_1, collect_2, dict0, dict1, dict2


def image_processing(oldpath, train_root, test_root):  # casia-webface, omit 647 identities less than 15 images

    count = []
    count_few = []
    count_mass = []
    for (rootDir, dirNames, filenames) in os.walk(oldpath):
        # i = 0
        # loop over the filenames in the current directory
        # print(dirNames[2538])
        for dirname in dirNames:
            folderpath_old = os.path.join(rootDir, dirname)
            image_paths = os.listdir(folderpath_old)
            count.append(len(image_paths))
            if len(image_paths) > 100:
                count_mass.append(dirname)
                continue
                # count_mass.append(dirname)
            #i += 1
            else:
                dst_train_dir = os.path.join(train_root, dirname)
                dst_test_dir = os.path.join(test_root, dirname)
                if not os.path.exists(dst_train_dir):
                    os.makedirs(dst_train_dir)
                    os.makedirs(dst_test_dir)
                image_paths.sort()  # make sure that the filenames have a fixed order before shuffling
                np.random.seed(24)  # 24, 66
                np.random.shuffle(image_paths)    # shuffles the ordering of filenames (deterministic given the chosen seed)
                # split_1 = int(0.5 * len(filenames))
                test_filenames = image_paths[:5]
                train_filenames = image_paths[5:]
                for image in train_filenames:
                    image_path = os.path.join(folderpath_old, image)
                    copyfile(image_path, os.path.join(dst_train_dir, image))
                for image in test_filenames:
                    image_path = os.path.join(folderpath_old, image)
                    copyfile(image_path, os.path.join(dst_test_dir, image))
    return count_mass


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def indexing(features, codebooks, len_word):
    '''
    encoding database images by PQN: cosine similarity
    '''

    features_split = torch.stack(torch.split(features, len_word, dim=1), dim=0)
    """
    since num_books * num_words can be huge, e.g. 8 * 256 for 64-bit;
    we index the samples in mini-batch
    """
    indices = []
    batch_num = features_split.shape[1] // 1024
    for i in range(batch_num):
        norm_features = F.normalize(features_split[:, i * 1024: (i+1) * 1024, :], dim=2)
        product = torch.matmul(norm_features, codebooks)  # num_books, bs, num_words
        indices_batch = torch.argmax(product, dim=2)    # shape of num_books, bs
        indices.append(indices_batch)

    norm_features = F.normalize(features_split[:, batch_num * 1024:, :], dim=2)
    product_last = torch.matmul(norm_features, codebooks)  # num_books, bs, num_words
    indices_last = torch.argmax(product_last, dim=2)    # shape of num_books, bs
    indices.append(indices_last)
    indices = torch.cat(indices, dim=1)

    return indices


def indexing_ortho(features, mlp_weight, len_word):
    """
    :param features:
    :param mlp_weight:
    :param len_word:
    :return: shape of (num_books, N), index of the codeword with largest softmax prediction for each sample in database. one row per codebook
    """

    features_split = torch.stack(torch.split(features, len_word, dim=1), dim=0)
    """
    since num_books * num_words can be huge, e.g. 8 * 256 for 64-bit;
    we index the samples in mini-batch
    """
    indices = []
    batch_num = features_split.shape[1] // 1024
    for i in range(batch_num):
        norm_features = F.normalize(features_split[:, i * 1024: (i+1) * 1024, :], dim=2)
        features_mlp_product = torch.matmul(norm_features, mlp_weight)  # (num_books, bs, len_word) * (num_books, len_word, num_words) ---> (num_books, bs, num_words)

        softmax_features = F.softmax(features_mlp_product, dim=2)
        indices_batch = torch.argmax(softmax_features, dim=2)    # shape of num_books, bs
        indices.append(indices_batch)

    norm_features = F.normalize(features_split[:, batch_num * 1024:, :], dim=2)
    features_mlp_product = torch.matmul(norm_features, mlp_weight)  # (num_books, bs, len_word) * (num_books, len_word, num_words) ---> (num_books, bs, num_words)
    # features_mlp_product = torch.matmul(norm_features[:, i * 1024: (i+1) * 1024, :], mlp_weight)  # (num_books, bs, len_word) * (num_books, len_word, num_words) ---> (num_books, bs, num_words)
    softmax_features_last = F.softmax(features_mlp_product, dim=2)
    indices_last = torch.argmax(softmax_features_last, dim=2)    # shape of num_books, bs
    indices.append(indices_last)
    indices = torch.cat(indices, dim=1)

    return indices


def compute_quant(transform, dataloader, net, device):
    bs = []
    label = []
    with torch.no_grad():
        for i, (imgs, cls, *_) in enumerate(dataloader):
            imgs = imgs.to(device)
            imgs = transform(imgs)
            hash_values = net(imgs)
            bs.append(hash_values.data)
            label.append(cls.to(device))

    return torch.cat(bs), torch.cat(label)


def compute_quant_indexing(transform, dataloader, net, len_word, mlp_weight, device, softmax=True, norm=True):
    """
    For 64-bit codes, cannot store the whole feature vectors of all the images.
    So, combine computing features and indexing in mini-batch to reduce the intermidate results
    :param transform:
    :param dataloader:
    :param net:
    :param len_word:
    :param mlp_weight:
    :param device:
    gpq: softmax = false, norm = true;
    dpq: softmax = true, norm = false
    :param softmax:
    :param norm:
    :return:
    """
    label = []
    indices = []
    for i, (imgs, cls, *_) in enumerate(dataloader):
        imgs = imgs.to(device)
        imgs = transform(imgs)
        features = net(imgs)
        features_split = torch.stack(torch.split(features, len_word, dim=1), dim=0)
        if norm:
            features_split = F.normalize(features_split, dim=2)
        features_mlp_product = torch.matmul(features_split, mlp_weight)  # (num_books, bs, len_word) * (num_books, len_word, num_words) ---> (num_books, bs, num_words)
        if softmax:
            features_mlp_product = F.softmax(features_mlp_product, dim=2)
        indices_batch = torch.argmax(features_mlp_product, dim=2)    # shape of num_books, bs
        indices.append(indices_batch)
        label.append(cls.to(device))

    indices = torch.cat(indices, dim=1)

    return indices, torch.cat(label)


def PqDistRet(test_features, test_labels, train_labels, code_book, index_table, len_word, num_book, device, top=None):

    # code_book = torch.transpose(code_book, 1, 2)
    num_test = test_features.size(0)
    AP = []
    top_p = []

    ####################### first intra-normalize the features

    features_split = torch.split(test_features, len_word, dim=1)  # num_books, bs, len_words
    features_split = torch.stack(features_split)
    norm_features = F.normalize(features_split, dim=2)
    for j in range(num_test):
        # start = time.time()
        cos_dst = (norm_features[:, j, :].unsqueeze(1) @ code_book).squeeze()  # M * num_words
        pdist_perbook = [F.embedding(index_table[k, :].view(-1, 1), cos_dst[k, :].view(-1, 1)).squeeze(dim=2) for k in range(num_book)]  # list of #num_books (1*N) tensor
        pdist_full = torch.cat(pdist_perbook, dim=1)    # N * K
        distQ = torch.sum(pdist_full, dim=1)        # (N , )

        _, sort_result = torch.sort(distQ, descending=True)
        correct = (test_labels[j] == train_labels[sort_result]).float()
        N = torch.sum(correct)
        if device == "cuda:0":
            Ns = torch.arange(1, N+1).float().to(device)    # the index in nonzero elements
        else:
            Ns = torch.arange(1, N+1).float()
        index = (torch.nonzero(correct, as_tuple=False) + 1)[:, 0].float()
        AP.append(torch.mean(Ns / index)) # the index of the whole vector
        if top is not None:
            top_result = sort_result[:top]
            top_correct = (test_labels[j] == train_labels[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p.append(1.0*N_top/top)
        # time_elapsed = time.time() - start
        # print("Compute {}-th sample in {:.0f}min {:.0f}s ".format(j, time_elapsed // 60, time_elapsed % 60))

    top_mAP = torch.mean(torch.Tensor(top_p).cuda())
    mAP = torch.mean(torch.Tensor(AP).cuda())


    return mAP, top_mAP


def PqDistRet_euclidean(test_features, test_labels, train_labels, index_table, mlp, codebook, len_word, num_book, device, top=None):

    """
    computing Euclidean distance, for DPQ method
    F.embeeding():
            input: index_table
            weights: softmax_score of query
    return:
    """

    num_test = test_features.size(0)
    AP = []
    top_p = []

    ####################### first intra-normalize the features

    features_split = torch.stack(torch.split(test_features, len_word, dim=1))


    for j in range(num_test):
        softmax_score = F.softmax(torch.matmul(features_split[:, j, :].unsqueeze(1), mlp), dim=2)   # M * 1 * K
        soft_features = torch.transpose((codebook @ torch.transpose(softmax_score, 1, 2)), 1, 2)    # (M * len_word * 1 ---> M * 1 * len_word)
        euclidean_dst = (soft_features - torch.transpose(codebook, 1, 2)).pow(2).sum(dim=2)  # (M * 1 * len_word - M* K * len_word).sum(dim=2) --> M * K
        # shape of M * num_words ,change back to norm_features

        # (num_books, 1, len_word)  * (num_books, len_word, num_words)  ---> (num_books, num_words)
        pdist_perbook = [F.embedding(index_table[k, :].view(-1, 1), euclidean_dst[k].view(-1, 1)).squeeze(dim=2) for k in range(num_book)]  # list of #num_books (N, 1) tensor
        # F.embedding(input_select_index, LUT_table)
        pdist_full = torch.cat(pdist_perbook, dim=1)  # shape of (N, #books)
        distQ = torch.sum(pdist_full, dim=1)

        _, sort_result = torch.sort(distQ, descending=False)  # from smallest dist.
        correct = (test_labels[j] == train_labels[sort_result]).float()
        N = torch.sum(correct)
        if device == "cuda:0":
            Ns = torch.arange(1, N+1).float().to(device)    # the index in nonzero elements
        else:
            Ns = torch.arange(1, N+1).float()
        index = (torch.nonzero(correct, as_tuple=False) + 1)[:, 0].squeeze().float()
        AP.append(torch.mean(Ns / index)) # the index of the whole vector
        if top is not None:
            top_result = sort_result[:top]
            top_correct = (test_labels[j] == train_labels[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p.append(1.0*N_top/top)


    top_mAP = torch.mean(torch.Tensor(top_p).cuda())
    mAP = torch.mean(torch.Tensor(AP).cuda())

    return mAP, top_mAP


def PqDistRet_Ortho_euclidean(test_features, test_labels, train_labels, index_table, mlp, codebook, len_word, num_book, device, top=None):

    """
    To test our method without orthogonal codewords
    add test features normalization compared to
    "PqDistRet_euclidean"
    """

    num_test = test_features.size(0)
    AP = []
    top_p = []

    features_split = torch.stack(torch.split(test_features, len_word, dim=1))

    norm_features = F.normalize(features_split, dim=2)

    for j in range(num_test):
        softmax_score = F.softmax(torch.matmul(norm_features[:, j, :].unsqueeze(1), mlp), dim=2)   # M * 1 * K
        soft_features = torch.transpose((codebook @ torch.transpose(softmax_score, 1, 2)), 1, 2)    # (M * len_word * 1 ---> M * 1 * len_word)
        euclidean_dst = (soft_features - torch.transpose(codebook, 1, 2)).pow(2).sum(dim=2)  # M * 1 * len_word - M * K * len_word
        # shape of M * num_words ,change back to norm_features

        # (num_books, 1, len_word)  * (num_books, len_word, num_words)  ---> (num_books, num_words)
        pdist_perbook = [F.embedding(index_table[k, :].view(-1, 1), euclidean_dst[k].view(-1, 1)).squeeze(dim=2) for k in range(num_book)]  # list of #num_books (N, 1) tensor
        # F.embedding(input_select_index, LUT_table)
        pdist_full = torch.cat(pdist_perbook, dim=1)  # shape of (N, #books)
        distQ = torch.sum(pdist_full, dim=1)

        _, sort_result = torch.sort(distQ, descending=False)  # from smallest dist.
        correct = (test_labels[j] == train_labels[sort_result]).float()
        N = torch.sum(correct)
        if device == "cuda:0":
            Ns = torch.arange(1, N+1).float().to(device)    # the index in nonzero elements
        else:
            Ns = torch.arange(1, N+1).float()
        index = (torch.nonzero(correct, as_tuple=False) + 1)[:, 0].squeeze().float()
        AP.append(torch.mean(Ns / index))   # the index of the whole vector
        if top is not None:
            top_result = sort_result[:top]
            top_correct = (test_labels[j] == train_labels[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p.append(1.0*N_top/top)


    top_mAP = torch.mean(torch.Tensor(top_p).cuda())
    mAP = torch.mean(torch.Tensor(AP).cuda())

    return mAP, top_mAP


def PqDistRet_Ortho(test_features, test_labels, train_labels, index_table, mlp, len_word, num_book, device, top=None):


    num_test = test_features.size(0)
    AP = []
    top_p = []


    features_split = torch.split(test_features, len_word, dim=1)
    features_split = torch.stack(features_split)    # num_books, bs, len_words
    norm_features = F.normalize(features_split, dim=2)
    for j in range(num_test):
        softmax_score = F.softmax(torch.matmul(norm_features[:, j, :].unsqueeze(1), mlp).squeeze(1), dim=1)

        # (num_books, 1, len_word)  * (num_books, len_word, num_words)  ---> (num_books, num_words)
        pdist_perbook = [F.embedding(index_table[k, :].view(-1, 1), softmax_score[k].view(-1, 1)).squeeze(dim=2) for k in range(num_book)]  # list of #num_books (N, 1) tensor
        # F.embedding(input, index_table)
        pdist_full = torch.cat(pdist_perbook, dim=1)  # shape of (N, #books)
        distQ = torch.sum(pdist_full, dim=1)

        _, sort_result = torch.sort(distQ, descending=True)
        correct = (test_labels[j] == train_labels[sort_result]).float()
        N = torch.sum(correct)
        if device == "cuda:0":
            Ns = torch.arange(1, N+1).float().to(device) # the index in nonzero elements
        else:
            Ns = torch.arange(1, N+1).float()
        index = (torch.nonzero(correct, as_tuple=False) + 1)[:, 0].float()
        AP.append(torch.mean(Ns / index))   # the index of the whole vector
        if top is not None:
            top_result = sort_result[:top]
            top_correct = (test_labels[j] == train_labels[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p.append(1.0*N_top/top)


    top_mAP = torch.mean(torch.Tensor(top_p).cuda())
    mAP = torch.mean(torch.Tensor(AP).cuda())

    return mAP, top_mAP


def pr_ortho(ranking_list, test_features, test_labels, train_labels, index_table, mlp, len_word, num_book, device, top=None):

    num_test = test_features.size(0)
    top_p = torch.Tensor(num_test, len(ranking_list))
    top_r = torch.Tensor(num_test, len(ranking_list))

    features_split = torch.split(test_features, len_word, dim=1)
    norm_features = F.normalize(features_split, dim=2)

    for j in range(num_test):
        softmax_score = F.softmax(torch.matmul(norm_features[:, j, :].unsqueeze(1), mlp).squeeze(1), dim=1)
        # (num_books, 1, len_word)  * (num_books, len_word, num_words)  ---> (num_books, num_words)
        pdist_perbook = [F.embedding(index_table[k, :].view(-1, 1), softmax_score[k].view(-1, 1)).squeeze(dim=2) for k in range(num_book)]  # list of #num_books (N, 1) tensor
        # F.embedding(input, index_table)
        pdist_full = torch.cat(pdist_perbook, dim=1)  # shape of (N, #books)
        distQ = torch.sum(pdist_full, dim=1)
        query_label = test_labels[j]
        _, sort_result = torch.sort(distQ, descending=True)
        cateTrainTest = torch.eq(train_labels, query_label).squeeze_()
        revelant_num = torch.nonzero(cateTrainTest).numel()
        for k, top in enumerate(ranking_list):
            top_result = sort_result[:top]
            top_correct = (query_label == train_labels[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p[j, k] = 1.0*N_top/top
            top_r[j, k] = 1.0*N_top/revelant_num

    p_curve = torch.mean(top_p, dim=0)
    r_curve = torch.mean(top_r, dim=0)

    p_curve = p_curve.cpu().data.numpy()
    r_curve = r_curve.cpu().data.numpy()

    return p_curve, r_curve


def pr_euclidean(ranking_list, test_features, test_labels, train_labels, index_table, mlp, codebook, len_word, num_book, device, top=None, norm=False):

    """
    computing Euclidean distance, for DPQ method
    F.embeeding():
            input: index_table
            weights: softmax_score of query
    return:
    """

    num_test = test_features.size(0)
    top_p = torch.Tensor(num_test, len(ranking_list))
    top_r = torch.Tensor(num_test, len(ranking_list))


    features_split = torch.stack(torch.split(test_features, len_word, dim=1))
    if norm:
        features_split = F.normalize(features_split, dim=2)

    for j in range(num_test):
        softmax_score = F.softmax(torch.matmul(features_split[:, j, :].unsqueeze(1), mlp), dim=2)   # M * 1 * K
        soft_features = torch.transpose((codebook @ torch.transpose(softmax_score, 1, 2)), 1, 2)    # (M * len_word * 1 ---> M * 1 * len_word)
        euclidean_dst = (soft_features - torch.transpose(codebook, 1, 2)).pow(2).sum(dim=2)  # (M * 1 * len_word - M* K * len_word).sum(dim=2) --> M * K
        # shape of M * num_words ,change back to norm_features

        # (num_books, 1, len_word)  * (num_books, len_word, num_words)  ---> (num_books, num_words)
        pdist_perbook = [F.embedding(index_table[k, :].view(-1, 1), euclidean_dst[k].view(-1, 1)).squeeze(dim=2) for k in range(num_book)]  # list of #num_books (N, 1) tensor
        # F.embedding(input_select_index, LUT_table)
        pdist_full = torch.cat(pdist_perbook, dim=1)  # shape of (N, #books)
        distQ = torch.sum(pdist_full, dim=1)
        _, sort_result = torch.sort(distQ, descending=False)  # from smallest dist.
        query_label = test_labels[j]
        cateTrainTest = torch.eq(train_labels, query_label).squeeze_()
        revelant_num = torch.nonzero(cateTrainTest).numel()
        for k, top in enumerate(ranking_list):
            top_result = sort_result[:top]
            top_correct = (query_label == train_labels[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p[j, k] = 1.0*N_top/top
            top_r[j, k] = 1.0*N_top/revelant_num

    p_curve = torch.mean(top_p, dim=0)
    r_curve = torch.mean(top_r, dim=0)

    p_curve = p_curve.cpu().data.numpy()
    r_curve = r_curve.cpu().data.numpy()

    return p_curve, r_curve


def pr_cos(ranking_list, test_features, test_labels, train_labels, code_book, index_table, len_word, num_book, device, top=None):

    # code_book = torch.transpose(code_book, 1, 2)
    num_test = test_features.size(0)
    top_p = torch.Tensor(num_test, len(ranking_list))
    top_r = torch.Tensor(num_test, len(ranking_list))

    features_split = torch.split(test_features, len_word, dim=1)  # num_books, bs, len_words
    features_split = torch.stack(features_split)
    norm_features = F.normalize(features_split, dim=2)
    for j in range(num_test):
        cos_dst = (norm_features[:, j, :].unsqueeze(1) @ code_book).squeeze()  # M * num_words
        pdist_perbook = [F.embedding(index_table[k, :].view(-1, 1), cos_dst[k, :].view(-1, 1)).squeeze(dim=2) for k in range(num_book)]  # list of #num_books (1*N) tensor
        pdist_full = torch.cat(pdist_perbook, dim=1)    # N * K
        distQ = torch.sum(pdist_full, dim=1)        # (N , )


        _, sort_result = torch.sort(distQ, descending=True)
        query_label = test_labels[j]
        cateTrainTest = torch.eq(train_labels, query_label).squeeze_()
        revelant_num = torch.nonzero(cateTrainTest).numel()
        for k, top in enumerate(ranking_list):
            top_result = sort_result[:top]
            top_correct = (query_label == train_labels[top_result]).float()    # boolean --> float?
            N_top = torch.sum(top_correct)
            top_p[j, k] = 1.0*N_top/top
            top_r[j, k] = 1.0*N_top/revelant_num

    p_curve = torch.mean(top_p, dim=0)
    r_curve = torch.mean(top_r, dim=0)

    p_curve = p_curve.cpu().data.numpy()
    r_curve = r_curve.cpu().data.numpy()

    return p_curve, r_curve


def PqDistRet_Ortho2(test_features, test_labels, train_labels, index_table, mlp, len_word, num_book, device, top=None):

    num_test = test_features.size(0)
    AP = []
    top_p = []

    ####################### first intra-normalize the features

    features_split = torch.split(test_features, len_word, dim=1)
    features_split = torch.stack(features_split)    # num_books, bs, len_words

    for j in range(num_test):
        softmax_score = F.softmax(torch.matmul(features_split[:, j, :].unsqueeze(1), mlp).squeeze(1), dim=1)
        pdist_perbook = [F.embedding(index_table[k, :].view(-1, 1), softmax_score[k].view(-1, 1)).squeeze(dim=2) for k in range(num_book)]  # list of #num_books (N, 1) tensor
        # F.embedding(input, index_table)
        pdist_full = torch.cat(pdist_perbook, dim=1)  # shape of (N, #books)
        distQ = torch.sum(pdist_full, dim=1)

        _, sort_result = torch.sort(distQ, descending=True)
        correct = (test_labels[j] == train_labels[sort_result]).float()
        N = torch.sum(correct)
        if device == "cuda:0":
            Ns = torch.arange(1, N+1).float().to(device) # the index in nonzero elements
        else:
            Ns = torch.arange(1, N+1).float()
        index = (torch.nonzero(correct, as_tuple=False) + 1)[:, 0].float()
        AP.append(torch.mean(Ns / index))   # the index of the whole vector
        if top is not None:
            top_result = sort_result[:top]
            top_correct = (test_labels[j] == train_labels[top_result]).float()  # boolean --> float?
            N_top = torch.sum(top_correct)
            top_p.append(1.0*N_top/top)
        '''
        if (j + 1) % math.ceil((num_test/10)) == 0:
            print("%d%% completed" %(10*percent))
            percent+=1
        '''

    top_mAP = torch.mean(torch.Tensor(top_p).cuda())
    mAP = torch.mean(torch.Tensor(AP).cuda())

    return mAP, top_mAP


def compute_topk(trainset, test_features, test_labels, train_labels, index_table, mlp, len_word, num_book, device, top=None):

   
    dict0 = defaultdict(list)
    dict1 = defaultdict(list)
    num_test = test_features.size(0)
    # AP = []
    top_p = []
    results = torch.Tensor(test_labels.size(0), top).to(device)

    features_split = torch.split(test_features, len_word, dim=1)
    features_split = torch.stack(features_split)    # num_books, bs, len_words
    norm_features = F.normalize(features_split, dim=2)

    for j in range(num_test):
        softmax_score = F.softmax(torch.matmul(norm_features[:, j, :].unsqueeze(1), mlp).squeeze(1), dim=1)
        pdist_perbook = [F.embedding(index_table[k, :].view(-1, 1), softmax_score[k].view(-1, 1)).squeeze(dim=2) for k in range(num_book)]  # list of #num_books (N, 1) tensor
        # F.embedding(input, index_table)
        pdist_full = torch.cat(pdist_perbook, dim=1)  # shape of (N, #books)
        distQ = torch.sum(pdist_full, dim=1)

        _, sort_result = torch.sort(distQ, descending=True)

        top_result = sort_result[:top]
        top_correct = (test_labels[j] == train_labels[top_result]).float()
        results[j, :] = top_correct
        N_top = torch.sum(top_correct)
        top_p = 1.0*N_top/top
        if top_p == 1:
            for k in range(top):
                dict0[j].append(trainset.imgs[top_result[k]][0])
        elif top_p == 0.8:
            for k in range(top):
                dict1[j].append(trainset.imgs[top_result[k]][0])

    # top_k = torch.Tensor(top_p).cuda()
    results = results.cpu().data.numpy()
    return results, dict0, dict1



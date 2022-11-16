import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from datetime import datetime
import torch.distributions as Distributions
import math
import argparse
from utils import *
from resnet import resnet20_pq
from margin_metric import OrthoPQ
from data_list import MyDataset_transform

parser = argparse.ArgumentParser(description='PyTorch Implementation of Orthonormal Product Quantization for Scalable Face Image Retrieval')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode turned on')
parser.add_argument('-u', '--unseen', action='store_true', help='generalize on unseen identities')
parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
parser.add_argument('--save', nargs='+', help='path to saving models, accept multiple arguments as list')
parser.add_argument('--load', nargs='+', help='path to loading models, accept multiple arguments as list')
parser.add_argument('--len', nargs='+', type=int, help='length of hashing codes, accept multiple arguments as list')

parser.add_argument('--dataset', type=str, default='facescrub', help='which dataset for training.(one of facescrub, youtube, CFW, and VGGFace2)')
parser.add_argument('--num', nargs='+', type=int, help='num. of codebooks, could be 4, 8...}')
parser.add_argument('--words', nargs='+', type=int, default=[256, 256, 256, 256], help='num of words,  should be exponential of 2')

parser.add_argument('--margin', default=0.5, type=float, help='margin of cosine similarity')
parser.add_argument('--miu', default=0.1, type=float, help='Balance weight of reduncy loss')


args = parser.parse_args()

transform_tensor = transforms.ToTensor()

if args.dataset == "vggface2":

    if args.unseen:
        trainPaths = "./data/vgg_face2/transfer_train"
        testPaths= "./data/vgg_face2/transfer_test"
    else:
        trainPaths = "./data/vgg_face2/train_cropped"    # train_extend
        testPaths = "./data/vgg_face2/test_cropped"

    trainset = datasets.ImageFolder(root=trainPaths, transform=transform_tensor)
    testset = datasets.ImageFolder(root=testPaths, transform=transform_tensor)


elif args.dataset == "facescrub":

    trainPaths = "./data/facescrub/train"
    testPaths = "./data/facescrub/test"
    trainset = datasets.ImageFolder(root=trainPaths, transform=transform_tensor)
    testset = datasets.ImageFolder(root=testPaths, transform=transform_tensor)

elif args.dataset == "youtube":

    trainPaths = "./data/youtube/train"
    testPaths = "./data/youtube/test"
    trainset = datasets.ImageFolder(root=trainPaths, transform=transform_tensor)
    testset = datasets.ImageFolder(root=testPaths, transform=transform_tensor)

else:
    trainset = MyDataset_transform(transform=None, train=True)
    testset = MyDataset_transform(transform=None, train=False)


torch.manual_seed(24)

if args.unseen:

    transform_train = torch.nn.Sequential(
                    transforms.Resize(120),
                    transforms.CenterCrop(112),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    )

    transform_test = torch.nn.Sequential(
                        transforms.Resize(120),
                        transforms.CenterCrop(112),
                        transforms.ConvertImageDtype(torch.float),
                        # transforms.Normalize((0.5580, 0.4279, 0.3680), (0.2719, 0.2354, 0.2263)),    # vggface2
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    )


else:

    if args.dataset in ["facescrub", "youtube"]:

        normalize = transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])

    if args.dataset == "cfw":

        normalize = transforms.Normalize([0.507, 0.403, 0.344], [0.259, 0.226, 0.212])

    if args.dataset in ["facescrub", "youtube", "cfw"]:

        transform_train = nn.Sequential(
                    transforms.Resize(35),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ConvertImageDtype(torch.float),
                    normalize
        )

        transform_test = nn.Sequential(
                    transforms.Resize(35),
                    transforms.CenterCrop(32),
                    transforms.ConvertImageDtype(torch.float),
                    normalize
        )

    else:
        transform_train = torch.nn.Sequential(
                    transforms.Resize(120),
                    transforms.RandomCrop(112),
                    transforms.RandomHorizontalFlip(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )

        transform_test = torch.nn.Sequential(
                        transforms.Resize(120),
                        transforms.CenterCrop(112),
                        transforms.ConvertImageDtype(torch.float),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# torch.cuda.manual_seed_all(1)


class adjust_lr:
    def __init__(self, step, decay):
        self.step = step
        self.decay = decay

    def adjust(self, optimizer, epoch):
        lr = args.lr * (self.decay ** (epoch // self.step))
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
        return lr


def train(save_path, length, num, words, feature_dim):

    # best_acc = 0
    best_mAP = 0
    best_epoch = 1
    print('==> Building model..')
    num_classes = len(trainset.classes)
    print("number of identities: ", num_classes)
    print("number of training images: ", len(trainset))
    print("number of test images: ", len(testset))

    print("number of training batches per epoch:", len(train_loader))
    print("number of testing batches per epoch:", len(test_loader))

    d = int(feature_dim / num)
    matrix = torch.randn(d, d)
    for k in range(d):
        for j in range(d):
            matrix[j, k] = math.cos((j+0.5)*k*math.pi/d)
    matrix[:, 0] /= math.sqrt(2)    # divided by sqrt(2)
    matrix /= math.sqrt(d/2)    # divided by sqrt(N/2)
    code_books = torch.Tensor(num, d, words)
    code_books[0] = matrix[:, :words]
    for i in range(1, num):
        code_books[i] = matrix @ code_books[i-1]

    if args.unseen or args.dataset == "vggface2":

        net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
        metric = OrthoPQ(in_features=feature_dim, out_features=num_classes, num_books=num, code_books=code_books, num_words=words, sc=40, m=args.margin)

    else:

        net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
        metric = OrthoPQ(in_features=feature_dim, out_features=num_classes, num_books=num, code_books=code_books, num_words=words, sc=20, m=args.margin)

    num_books = metric.num_books
    len_word = metric.len_word
    num_words = metric.num_words
    len_bit = int(num_books * math.log(num_words, 2))
    assert length == len_bit, "something wrong with the length"
    criterion = nn.CrossEntropyLoss()
    print("num. of codebooks: ", num_books)
    print("num. of words per book: ", num_words)
    print("dim. of word: ", len_word)
    print("code length: %d-bit \t learning rate: %.3f \t scale length: %d \t penalty margin: %.2f \t balance_weight: %.3f" % (len_bit, args.lr, metric.s, metric.m, args.miu))
    net = nn.DataParallel(net).to(device)
    metric = nn.DataParallel(metric).to(device)
    cudnn.benchmark = True

    if args.dataset in ["facescrub", "cfw", "youtu"]:

        optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}], lr=args.lr, weight_decay=5e-4, momentum=0.9)

        scheduler = adjust_lr(35, 0.5)
        EPOCHS = 200

    else:
        scheduler = adjust_lr(20, 0.5)

        EPOCHS = 160

        optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}], lr=args.lr, weight_decay=5e-4, momentum=0.9)

    since = time.time()
    best_loss = 1e3

    for epoch in range(EPOCHS):
        print('==> Epoch: %d' % (epoch+1))
        net.train()

        losses = AverageMeter()
        # clf_loss = AverageMeter()
        # cent_loss = AverageMeter()
        ##############################################
        scheduler.adjust(optimizer, epoch)
        start = time.time()
        ##############################################
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            transformed_images = transform_train(inputs)
            features = net(transformed_images)
            output, output2, xc_probs = metric(features, targets)

            loss_clf1 = [criterion(output[:, i, :], targets) for i in range(num_books)]
            loss_clf2 = [criterion(output2[:, i, :], targets) for i in range(num_books)]
            loss_clf = 0.5 * (sum(loss_clf1) / len(loss_clf1) + sum(loss_clf2) / len(loss_clf2))

            # subspace entropy minimization
            xc_entropy = [Distributions.categorical.Categorical(probs=xc_probs[:, i, :]).entropy().sum() for i in range(num_books)]   # -p * logP
            loss_entropy = sum(xc_entropy) / (num_books * len(inputs))
            loss = loss_clf + args.miu * loss_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), len(inputs))
            # clf_loss.update(loss_clf.item(), len(inputs))
            # cent_loss.update(loss_entropy.item(), len(inputs))

        epoch_elapsed = time.time() - start
        print('Epoch %d |  Loss: %.4f' %(epoch+1, losses.avg))
        print("Epoch Completed in {:.0f}min {:.0f}s".format(epoch_elapsed // 60, epoch_elapsed % 60))
        # scheduler.step()

        if (epoch+1) % 5 == 0:
            net.eval()
            with torch.no_grad():
                mlp_weight = metric.module.mlp
                index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
                queries, test_labels = compute_quant(transform_test, test_loader, net, device)
                start = time.time()
                mAP, top_k = PqDistRet_Ortho(queries, test_labels, train_labels, index, mlp_weight, len_word, num_books, device, top=50)
                time_elapsed = time.time() - start
                print("Code generated in {:.0f}min {:.0f}s ".format(time_elapsed // 60, time_elapsed % 60))
                print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

        if losses.avg < best_loss:
            best_loss = losses.avg
            # best_mAP = mAP
            print('Saving..')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save({'backbone': net.state_dict(),
                    'mlp': metric.module.mlp}, './checkpoint/%s' % save_path)
            best_epoch = epoch + 1
    time_elapsed = time.time() - since
    print("Training Completed in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best mAP {:.4f} at epoch {}".format(best_mAP, best_epoch))
    print("Model saved as %s \n" % save_path)




if __name__ == "__main__":

    # adjust_learning_rate(optimizer, epoch, args)
    save_dir = 'log'
    assert len(args.save) == len(args.num) and len(args.save) == len(args.words), 'model paths must be in line with # code lengths'
    for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
        sys.stdout = Logger(os.path.join(save_dir,
            str(args.len[i]) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
        print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n num_books: %d\n num_words: %d"
        %(args.dataset, args.len[i], args.bs, args.lr, num_s, words_s))
        print("HyperParams:\nmargin: %.3f\t miu: %.4f" % (args.margin, args.miu))
        if args.dataset not in ["vggface2", "casia"]:
            if args.len[i] != 36:
                train(args.save[i], args.len[i], num_s, words_s, feature_dim=512)
            else:
                train(args.save[i], args.len[i], num_s, words_s, feature_dim=516)
        else:

            train(args.save[i], args.len[i], num_s, words_s, feature_dim=num_s * words_s)







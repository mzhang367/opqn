import torch.nn as nn
from dpq_layer import DPQ, DPQJointClassLoss
import torchvision.transforms as transforms
from backbone import CosQuantNet34, SphereNet20_pq
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime
import pdb
from utils import *
import argparse
import torch.backends.cudnn as cudnn
import math
from data_loader import get_datasets_transform


parser = argparse.ArgumentParser(description='PyTorch Deep Product Quantization')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode turned on')
parser.add_argument('-c', '--cross-dataset', action='store_true', help='generalize on unseen identities')
parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
parser.add_argument('--save', nargs='+', help='path to saving models, accept multiple arguments as list')
parser.add_argument('--load', nargs='+', help='path to loading models, accept multiple arguments as list')
parser.add_argument('--len', nargs='+', type=int, help='length of hashing codes, accept multiple arguments as list')
parser.add_argument('--dataset', type=str, default='facescrub', help='which dataset for training.(facescrub, youtube)')
parser.add_argument('--num', nargs='+', type=int, help='num. of codebooks,  should be one of {4, 8}')
parser.add_argument('--words', nargs='+', type=int, default=64, help='num of words,  should be one of {8, 64, 256}')
parser.add_argument('--alpha', default=0.25, type=float, help='joint class loss balance')
parser.add_argument('--beta1', default=1, type=float, help='gini diversity loss balance')
parser.add_argument('--beta2', default=0.01, type=float, help='gini sharpness loss balance')

args = parser.parse_args()
transform_tensor = transforms.ToTensor()


trainset, testset = get_datasets_transform(args.dataset, cross_eval=args.c)['dataset']
transform_train, transform_test = get_datasets_transform(args.dataset, cross_eval=args.c)['transform']


train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)

torch.cuda.manual_seed_all(1)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class adjust_lr:
    def __init__(self, step, decay):
        self.step = step
        self.decay = decay

    def adjust(self, optimizer, epoch):
        lr = args.lr * (self.decay ** (epoch // self.step))
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
        return lr


def Log(x):

    lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.tensor([0.]).cuda())

    return lt


def train(save_path, len_bit, num_books, words, feature_dim=512):

    best_mAP = 0
    best_epoch = 1
    print('==> Building model..')
    num_classes = len(trainset.classes)
    print("number of identities: ", num_classes)
    print("number of training images: ", len(trainset))
    print("number of test images: ", len(testset))

    # print("number of training batches per epoch:", len(train_loader))
    # print("number of testing batches per epoch:", len(test_loader))

    if args.dataset == "vggface2":

        net = SphereNet20_pq(num_layers=20, feature_dim=feature_dim)
    else:
        # net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
        net = CosQuantNet34(num_seg=int(len_bit / 6), split=False, feature_dim=feature_dim)

    net = nn.DataParallel(net).to(device)
    metric = DPQ(in_features=feature_dim, num_books=num_books, num_words=words)
    num_books = metric.num_books
    len_word = metric.len_word
    num_words = metric.num_words
    print("[Configuration] code length: %d-bit\n feature dim. %d\n "
          "num. of codebooks: %d\n num. of words/book: %d\n dim. of word: %d"
          % (int(num_books*math.log(num_words, 2)), feature_dim, num_books, num_words, len_word))

    # print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n \n"
    # %(args.dataset, len_bit, args.bs, args.lr1))
    metric = nn.DataParallel(metric).to(device)
    ##############################################################################################
    criterion = DPQJointClassLoss(num_class=num_classes, feature_dim=feature_dim, param=args.alpha).cuda() ###########################

    cudnn.benchmark = True

    optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}, {'params': criterion.parameters()}],
    lr=args.lr, weight_decay=5e-4, momentum=0.9)


    if args.dataset in ["facescrub", "cfw"]:
        scheduler = adjust_lr(step=35, decay=0.5)
        # adjust_learning_rate = adjust_lr(step=40, decay=0.1)
        EPOCHS = 160

    else:
        scheduler = adjust_lr(step=20, decay=0.5)
        EPOCHS = 160


    since = time.time()
    best_loss = 1e3

    for epoch in range(EPOCHS):
        print('==> Epoch: %d' % (epoch+1))
        net.train()

        losses = AverageMeter()
        clf_loss = AverageMeter()
        gini_loss = AverageMeter()
        ##############################################
        scheduler.adjust(optimizer, epoch)
        start = time.time()
        ##############################################
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(),  targets.cuda()
            transformed_inputs = transform_train(inputs)

            inputs_feature = net(transformed_inputs)
            soft_x, hard_x, x_probs = metric(inputs_feature)
            # shape of output: bs * codebook * num_classes
            loss_clf = criterion(soft_x, hard_x, targets)   # clf_loss + alpha * center_loss
            batch_prob = torch.transpose(x_probs, 0, 1)  # M * bs * K

            gini_diversity = (batch_prob.sum(dim=1) / len(inputs)).pow(2).sum()
            gini_sharpness = - batch_prob.pow(2).sum() / (len(inputs))  # not dividing by num_books
            loss_gini = args.beta1 * gini_diversity + args.beta2 * gini_sharpness
            loss = loss_clf + loss_gini

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), len(inputs))
            clf_loss.update(loss_clf.item(), len(inputs))
            gini_loss.update(loss_gini.item(), len(inputs))

        epoch_elapsed = time.time() - start
        print('Epoch %d |  Loss: %.4f (clf_loss: %.4f, gini_loss: %.4f))'   # | reduncy: %.4f
            %(epoch+1, losses.avg, clf_loss.avg, gini_loss.avg))  # torch.Tensor(loss_reduncys).mean()
        print("Epoch Completed in {:.0f}min {:.0f}s".format(epoch_elapsed // 60, epoch_elapsed % 60))
        #####################################################################################################
        if (epoch+1) % 5 == 0:
            net.eval()
            with torch.no_grad():
                codewords = metric.module.codebook
                mlp_weight = metric.module.mlp
                index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device, softmax=True, norm=False)   # build embeddings for database image, compute until p_{im}
                queries, test_labels = compute_quant(transform_test, test_loader, net, device)
                start = time.time()
                mAP, top_k = PqDistRet_euclidean(queries, test_labels, train_labels, index, mlp_weight, codewords, len_word, num_books, device, top=10)
                time_elapsed = time.time() - start

                print("Code generated in {:.0f}min {:.0f}s ".format(time_elapsed // 60, time_elapsed % 60))
                print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

        if losses.avg < best_loss:
        # if mAP > best_mAP:
            best_loss = losses.avg
            # best_mAP = mAP
            print('Saving..')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            # torch.save(net.state_dict(), './checkpoint/%s' % args.save)
            torch.save({'backbone': net.state_dict(),
                    'mlp': metric.module.mlp, 'codebook': metric.module.codebook}, './dpq_checkpoint/%s' % save_path)
            best_epoch = epoch + 1
    time_elapsed = time.time() - since
    print("Training Completed in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best mAP {:.4f} at epoch {} \n".format(best_mAP, best_epoch))


def test(length, load_path, num, words, feature_dim):

    len_bit = int(num * math.log(words, 2))
    assert length == len_bit, "something went wrong with code length"

    #top_list = torch.linspace(10, 100, 10).int().tolist()
    top_list = torch.linspace(20, 300, 15).int().tolist()

    print("===============evaluation on model %s===============" % load_path)

    if args.dataset in ["facescrub", "cfw", "youtube"]:

        if not args.c:
            net = CosQuantNet34(num_seg=num, split=False, feature_dim=feature_dim)
        else:
            net = SphereNet20_pq(num_layers=20, feature_dim=feature_dim)
    else:

        net = SphereNet20_pq(num_layers=20, feature_dim=feature_dim)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)
    num_classes = len(trainset.classes)
    print("number of identities: ", num_classes)
    print("number of training images: ", len(trainset))
    print("number of test images: ", len(testset))

    print("number of training batches per epoch:", len(train_loader))
    print("number of testing batches per epoch:", len(test_loader))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    net = nn.DataParallel(net).to(device)

    checkpoint = torch.load("./dpq_checkpoint/%s" % load_path)
    net.load_state_dict(checkpoint['backbone'])
    mlp_weight = checkpoint['mlp']
    code_words = checkpoint['codebook']
    len_word = int(feature_dim / num)
    net.eval()
    with torch.no_grad():
        index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device, softmax=True, norm=False)   # build embeddings for database image, compute until p_{im}
        query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
        start = datetime.now()
        mAP, top_k = PqDistRet_euclidean(query_features,  test_labels, train_labels, index, mlp_weight, code_words, len_word, num, device, top=5)
        time_elapsed = datetime.now() - start
        print("Query completed in %d ms " %int(time_elapsed.total_seconds()*1000))
        print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

if __name__ == "__main__":


    save_dir = './log_dpq'
    if args.evaluate:
        assert len(args.load) == len(args.num), 'model paths must be in line with # code lengths'
        for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
            if args.c:
                feature_dim = num_s * words_s
            else:
                if args.dataset!="vggface2":
                    if args.len[i] != 36:
                        feature_dim = 512
                    else:
                        feature_dim = 516
                else:
                    feature_dim=num_s * words_s
            test(args.load[i], args.len[i], num_s, words_s, feature_dim=feature_dim)

    else:
        assert len(args.save) == len(args.num) and len(args.save) == len(args.words), 'model paths must be in line with # code lengths'
        for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
            sys.stdout = Logger(os.path.join(save_dir,
                str(args.len[i]) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
            print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n num_books: %d\n num_words: %d"
            %(args.dataset, args.len[i], args.bs, args.lr, num_s, words_s))
            print("HyperParams:\nmargin: %.3f\t miu: %.4f" % (args.margin, args.miu))
            if args.dataset!="vggface2":
                if args.len[i] != 36:
                    feature_dim = 512
                else:
                    feature_dim = 516
            else:
                feature_dim=num_s * words_s
          
            train(args.save[i], args.len[i], num_s, words_s, feature_dim=feature_dim)





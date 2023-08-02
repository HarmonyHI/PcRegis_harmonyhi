import os
from os.path import isfile
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import args
from TempNet import TempNet
from data import WhuDataset
from utils import seed_everything, ensure_path, ensure_delete


def train_one_epoch(net, loader, opt, scheduler, epoch):
    loss_sum = 0
    for src_knn, tgt_knn, _, _ in tqdm(loader):
        opt.zero_grad()
        loss = net(src_knn, tgt_knn, True)
        loss.backward()
        opt.step()
        loss_sum += loss
    print(f"========== Epoch {epoch} loss {loss_sum / loader.__len__()}==========")
    scheduler.step()
    return loss_sum


def test_one_epoch(net, loader, epoch):
    for src_knn, tgt_knn, R, t in tqdm(loader):
        R_pred, t_pred = net(src_knn, tgt_knn, False)
        print(f"========== Epoch {epoch} R {R} R_pred {R_pred} T {t} T_pred {t_pred}==========")
        break


def save_net(net):
    ensure_delete(args.net_path)
    torch.save(net.state_dict(), args.net_path)
    print("save net successfully")


def main():
    net = TempNet().to(device=args.device, dtype=args.tensor_format)
    if args.need_save_net and isfile(args.net_path):
        print("found saved net")
        m_state_dict = torch.load(args.net_path)
        net.load_state_dict(m_state_dict)
    train_dataset = WhuDataset(gaussian_noise=args.gaussian_noise, unseen=args.unseen, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=args.shuffle,
                              drop_last=args.drop_last,
                              num_workers=args.num_workers)
    test_dataset = WhuDataset(gaussian_noise=args.gaussian_noise, unseen=args.unseen, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=args.shuffle,
                             drop_last=args.drop_last,
                             num_workers=args.num_workers)
    opt = optim.Adam(net.parameters(), lr=args.learn_rate, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(opt, milestones=args.milestones, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        args.now_epoch = epoch
        if epoch % 10 == 0:
            if args.need_save_net:
                save_net(net)
            net.eval()
            print(f"epoch {epoch} test")
            test_one_epoch(net, test_loader, epoch)
        else:
            if epoch % 10 == 1:
                net.train()
            print(f"epoch {epoch} train")
            train_one_epoch(net, train_loader, opt, scheduler, epoch)


if __name__ == '__main__':
    seed_everything()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if args.device.type != "cpu":
        print("CUDA ENABLED")
    ensure_path(args.net_path)
    main()

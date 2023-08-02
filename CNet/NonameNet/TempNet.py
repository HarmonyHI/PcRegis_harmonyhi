import sys

import torch
import torch.nn as nn
import torch.nn.functional as func
import args
import utils
from utils import get_k_idx, tensor_rebuild, sp_point_match


class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, k_size):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, k_size, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, k_size):
        super(Conv2DBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, k_size, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, channels, k_size):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i + 1], k_size))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], k_size))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, channels, k_size):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.conv.append(Conv2DBNReLU(channels[i], channels[i + 1], k_size))
        self.conv.append(nn.Conv2d(channels[-2], channels[-1], k_size))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class GraphAttention(nn.Module):
    def __init__(self, all_channel, feature_dim, dropout, alpha):
        super(GraphAttention, self).__init__()
        self.alpha = alpha
        self.a = nn.Parameter(torch.zeros(size=(all_channel, feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, center_xyz, center_feature, grouped_xyz, grouped_feature):
        """
        Input:
            center_xyz: sampled points position data [B, n_point, C]
            center_feature: centered point feature [B, n_point, D]
            grouped_xyz: group xyz data [B, n_point, n_sample, C]
            grouped_feature: sampled points feature [B, n_point, n_sample, D]
        Return:
            graph_pooling: results of graph pooling [B, n_point, D]
        """
        B, n_point, C = center_xyz.size()
        _, _, n_sample, D = grouped_feature.size()
        delta_p = center_xyz.view(B, n_point, 1, C).expand(B, n_point, n_sample,
                                                           C) - grouped_xyz  # [B, n_point, n_sample, C]
        delta_h = center_feature.view(B, n_point, 1, D).expand(B, n_point, n_sample,
                                                               D) - grouped_feature  # [B, n_point, n_sample, D]
        delta_p_concat_h = torch.cat([delta_p, delta_h], dim=-1)  # [B, n_point, n_sample, C+D]
        e = self.leaky_relu(torch.matmul(delta_p_concat_h, self.a))  # [B, n_point, n_sample,D]
        attention = func.softmax(e, dim=2)  # [B, n_point, n_sample,D]
        attention = func.dropout(attention, self.dropout, training=self.training)
        graph_pooling = torch.sum(torch.mul(attention, grouped_feature), dim=2)  # [B, n_point, D]
        return graph_pooling


class KeypointDetector(nn.Module):
    def __init__(self, out):
        super(KeypointDetector, self).__init__()
        self.knn_num = args.knn_num
        self.conv1 = nn.Conv2d(10, out[0], kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out[0], eps=1e-6, momentum=0.01)
        self.conv2 = nn.Conv2d(out[0], out[1], kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out[1], eps=1e-6, momentum=0.01)
        self.conv3 = nn.Conv2d(out[1], out[2], kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out[2], eps=1e-6, momentum=0.01)
        self.pool3 = AttentivePooling(out[2], out[2])
        self.mlp = MLP()

    def forward(self, point_cloud, pc_knn):
        xyz = self.relative_pos_encoding(point_cloud, pc_knn)
        xyz = func.leaky_relu(self.bn1(self.conv1(xyz.to(args.tensor_format))), negative_slope=0.2)
        xyz = func.leaky_relu(self.bn2(self.conv2(xyz)), negative_slope=0.2)
        xyz = func.leaky_relu(self.bn3(self.conv3(xyz)), negative_slope=0.2)
        xyz = self.pool3(xyz).squeeze(-1)
        N128 = xyz
        N128 = N128.permute(0, 2, 1)
        score = self.mlp(xyz)
        score = score.permute(0, 2, 1)
        score = score.squeeze(-1)
        loss = 0
        return score, N128, loss

    def relative_pos_encoding(self, xyz, neighbor_xyz):
        neighbor_xyz = neighbor_xyz.permute(0, 3, 1, 2).contiguous()
        xyz = xyz[:, :, None, :].permute(0, 3, 1, 2).contiguous()
        repeated_xyz = xyz.repeat(1, 1, 1, self.knn_num)
        relative_xyz = repeated_xyz - neighbor_xyz
        relative_dist = torch.sqrt(torch.sum(relative_xyz ** 2, dim=1, keepdim=True))
        relative_feature = torch.cat([relative_dist, relative_xyz, repeated_xyz, neighbor_xyz], dim=1) \
            .to(args.tensor_format)
        return relative_feature


class AttentivePooling(nn.Module):
    def __init__(self, n_feature, d_out):
        super().__init__()
        self.n_feature = n_feature
        self.fc1 = nn.Linear(n_feature, n_feature, bias=False)
        self.conv1 = nn.Conv2d(n_feature, d_out, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(d_out, eps=1e-6, momentum=0.01)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[2]
        num_neigh = x.shape[3]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.reshape(x, [-1, num_neigh, self.n_feature])
        att_activation = self.fc1(x)
        att_score = func.softmax(att_activation, dim=1)
        x = x * att_score
        x = torch.sum(x, dim=1)
        x = torch.reshape(x, [batch_size, num_points, self.n_feature])[:, :, :, None].permute(0, 2, 1,
                                                                                              3).contiguous()
        x = func.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        return x


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        return x


class SVDHead(nn.Module):
    def __init__(self):
        super(SVDHead, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, src_corr, weights):
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)
        H = torch.matmul(src_centered * weights.unsqueeze(1), src_corr_centered.transpose(2, 1).contiguous())
        R = []
        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)
        R = torch.stack(R, dim=0)
        t = torch.matmul(-R, (weights.unsqueeze(1) * src).sum(dim=2, keepdim=True)) + (
                weights.unsqueeze(1) * src_corr).sum(dim=2, keepdim=True)
        return R, t.view(src.size(0), 3)


class LRF:
    def __init__(self, samples_per_patch=100):
        super(LRF, self).__init__()
        self.samples_per_patch = samples_per_patch
        self.device = args.device

    def _forward(self, xp, xpi, r_lrf):
        B, N, c = xpi.size()
        xpi = xpi.contiguous()  # dim = B x 3 x N
        xp = xp.unsqueeze(2).contiguous()  # dim = B x 3 x 1
        # zp
        x = xp - xpi  # pi->p = p - pi
        xxt = torch.bmm(x, x.transpose(1, 2)) / c
        _, _, v = torch.svd(xxt.to(self.device))
        v = v.to(self.device)
        with torch.no_grad():
            sum_ = (v[..., -1].unsqueeze(1) @ x).sum(2)
            _sign = torch.ones((len(xpi), 1), device=self.device) - 2 * (sum_ < 0)
        zp = (_sign * v[..., -1]).unsqueeze(1)  # B x 1 x 3
        # xp
        x *= -1  # p->pi = pi - p
        norm = (zp @ x).transpose(1, 2)
        proj = norm * zp
        vi = x - proj.transpose(1, 2)
        x_l2 = torch.sqrt((x ** 2).sum(dim=1, keepdim=True))
        alpha = r_lrf - x_l2
        alpha = alpha * alpha
        beta = (norm * norm).transpose(1, 2)
        vi_c = (alpha * beta * vi).sum(2)
        xp = (vi_c / torch.sqrt((vi_c ** 2).sum(1, keepdim=True)))
        # yp
        yp = torch.cross(xp, zp[:, 0, :], dim=1)
        lrf = torch.cat((xp.unsqueeze(2), yp.unsqueeze(2), zp.transpose(1, 2)), dim=2)
        return lrf

    def forward(self, x0, x0i, r_lrf):
        # compute local reference frames
        lrf0 = self._forward(x0, x0i, r_lrf)
        # ind_s = np.random.choice(x0i.shape[2], self.samples_per_patch, replace=False)
        # _out_x0 = (x0i[..., ind_s] - x0.unsqueeze(-1)) / self.r_lrf
        _out_x0 = (x0i - x0.unsqueeze(-1)) / r_lrf
        out_x0 = lrf0.transpose(1, 2) @ _out_x0
        return out_x0


class FeatDescript(nn.Module):
    def __init__(self):
        super(FeatDescript, self).__init__()
        self.mlp1 = Conv2DBlock((3, 64, 128, 256), [1, 1])
        self.mlp2 = Conv2DBlock((512, 128, 64), [1, 1])
        self.mlp3 = Conv1DBlock((3, 64, 128, 64, 32, 16), 1)

    def forward(self, knn_point):
        knn_point = knn_point.permute((0, 3, 1, 2))
        center_point = knn_point[:, :, :, 0]  # [B, 3, N]
        knn_point = knn_point - center_point.unsqueeze(-1)
        knn_feature = self.mlp1(knn_point)  # [B, 256, N, K]
        max_pooled = knn_feature.max(dim=-1, keepdim=True)[0]  # [B, 256, N, 1]
        max_pooled = max_pooled.repeat(1, 1, 1, knn_point.shape[-1])  # [B, 256, N, K]
        knn_feature = torch.cat((knn_feature, max_pooled), dim=1)  # [B, 512, N, K]
        knn_feature = self.mlp2(knn_feature)
        max_pooled = knn_feature.max(dim=-1, keepdim=True)[0]  # [B, 64, N, 1]
        knn_feature = max_pooled.squeeze(-1)
        knn_feature = func.normalize(knn_feature, dim=1, p=2)  # [B, 64, N]

        coordinate_feature = self.mlp3(center_point)  # [B, 64, N]
        coordinate_feature = func.normalize(coordinate_feature, dim=1, p=2)  # [B, 64, N]
        hybrid_feature = torch.cat((coordinate_feature, knn_feature), dim=1)  # [B, 80, N]

        return hybrid_feature


class TempNet(nn.Module):
    def __init__(self):
        super(TempNet, self).__init__()
        self.keypoint_detector = KeypointDetector([32, 64, 128])
        self.svd = SVDHead()
        self.LRF = LRF()
        self.feat_descript = FeatDescript()

    def forward(self, src_knn, tgt_knn, is_train):
        src_knn = src_knn.to(args.device)
        tgt_knn = tgt_knn.to(args.device)
        k_src_knn, keypoint_feat_src, super_loss_src = self.get_super_index(src_knn)
        k_tgt_knn, keypoint_feat_tgt, super_loss_tgt = self.get_super_index(tgt_knn)
        lrf_src_knn = self.LRF_transform(k_src_knn)
        lrf_tgt_knn = self.LRF_transform(k_tgt_knn)
        # utils.output_color_corr_xyz(lrf_src_knn, lrf_tgt_knn,
        #                             f"fuck1121_{args.now_epoch}_{args.colorful_output_version}")
        # sys.exit(0)
        src_feat = self.feat_descript(lrf_src_knn)
        tgt_feat = self.feat_descript(lrf_tgt_knn)
        src_feat = self.feat_mixture(keypoint_feat_src, src_feat)
        tgt_feat = self.feat_mixture(keypoint_feat_tgt, tgt_feat)
        if is_train:
            return self.train_forward(src_feat, tgt_feat, super_loss_src, super_loss_tgt)
        else:
            return self.test_forward(k_src_knn, k_tgt_knn, src_feat, tgt_feat)

    def train_forward(self, src_feat, tgt_feat, super_loss_src, super_loss_tgt):
        loss = self.get_loss(src_feat, tgt_feat, super_loss_src, super_loss_tgt)
        args.last_loss = loss
        return loss

    def test_forward(self, k_src_knn, k_tgt_knn, src_feat, tgt_feat):
        corr_src_knn, corr_tgt_knn = sp_point_match(k_src_knn, k_tgt_knn, src_feat, tgt_feat)
        rot_pred, trans_pred = self.get_r_t(corr_src_knn, corr_tgt_knn)
        utils.output_color_corr_xyz(corr_src_knn, corr_tgt_knn,
                                    f"color_feat_match_epoch_{args.now_epoch}_{args.colorful_output_version}")
        args.colorful_output_version = args.colorful_output_version + 1
        return rot_pred, trans_pred

    def LRF_transform(self, knn_data):
        if args.disable_LRF:
            return knn_data
        LRF_data = []
        for batch_iter in range(knn_data.shape[0]):
            knn = knn_data[batch_iter]
            xyz = knn[:, 0, :]  # [N, 3]
            farthest_point = knn[:, args.knn_num, :]  # [N, 3]
            r_inf = func.pairwise_distance(xyz.unsqueeze(1), farthest_point.unsqueeze(1))
            r_inf = r_inf[:, 0]  # [N]
            knn = knn.permute(0, 2, 1)  # [N, 3, K]
            r_inf = r_inf.max()
            LRF_data.append(self.LRF.forward(xyz, knn, r_inf).permute((0, 2, 1)))  # [N, K, 3]
        LRF_data = torch.stack(LRF_data)
        return LRF_data  # [B, N, K, 3]

    @staticmethod
    def get_loss(src_feat, tgt_feat, super_loss_src, super_loss_tgt):
        rand_feat = tgt_feat[torch.randperm(tgt_feat.size(0))][:][:]
        loss = torch.triplet_margin_loss(src_feat, tgt_feat, rand_feat)
        loss = loss.sum() / (args.train_split_block * args.train_batch_size)
        if not args.disable_keypoint:
            loss = loss + super_loss_src + super_loss_tgt
        return loss

    def get_r_t(self, src_corr, tgt_corr):
        if args.skip_rt:
            return None, None
        R, t = self.svd(src_corr, tgt_corr)
        return R, t

    def get_super_index(self, data):
        if args.disable_keypoint:
            return data, None, 0
        pc = data[:, :, 0, :]
        pc_kdtree = data[:, :, 1::, :]
        score, n128, loss = self.keypoint_detector(pc, pc_kdtree)
        sp_index = get_k_idx(score, args.top_k_num)
        nk128 = tensor_rebuild(n128, sp_index)
        data = tensor_rebuild(data, sp_index)
        return data, nk128, loss

    def feat_mixture(self, keypoint_feat, extract_feat):
        if not args.disable_keypoint:
            nk384 = torch.cat((keypoint_feat, extract_feat), 2)
            feat = self.fc3(self.fc2(self.fc1(nk384)))
        else:
            feat = extract_feat
        return feat

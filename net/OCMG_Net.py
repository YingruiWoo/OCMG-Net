import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_sign
from .utils import knn_group_0, get_knn_idx
from .local_feature import LocalFeature_Extraction, AdaptiveLayer
from .decode import PosionFusion
from .utils import HierarchicalLayer, Conv1D
from .utils import batch_quat_to_rotmat

class PointEncoderG(nn.Module):
    def __init__(self, num_out=[], knn=8, code_dim=128):
        super(PointEncoderG, self).__init__()
        self.num_out = num_out

        self.encodeNet = LocalFeature_Extraction(num_convs=4,
                                                  conv_channels=24,
                                                  knn=knn)

        dim_1 = self.encodeNet.out_channels

        self.conv_1 = Conv1D(dim_1, 128)
        self.conv_2 = Conv1D(128, 256)

        self.shift_1 = HierarchicalLayer(self.num_out[0], 256, 256, with_fc=True, neighbor_feature=False)
        self.shift_2 = HierarchicalLayer(self.num_out[1], 256, 256, last_dim=128, with_last=True, with_fc=True,
                                         neighbor_feature=False)
        self.shift_3 = HierarchicalLayer(self.num_out[1], 256, 256, last_dim=128, with_last=True, with_fc=True,
                                         neighbor_feature=False)

        self.conv_3 = Conv1D(256, 256)
        self.conv_4 = Conv1D(256, code_dim)
        
        self.mlp_global = nn.Sequential(
            nn.Linear(128 * 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        ) 

    def forward(self, pos, knn_idx):
        """
            pos: (B, N, 3)
            knn_idx: (B, N, K)
        """

        ### Multi-scale Local Feature Aggregation
        y = self.encodeNet(pos, knn_idx=knn_idx).transpose(2, 1)

        y = self.conv_1(y)
        y = self.conv_2(y)

        ### Hierarchical
        y, global_1 = self.shift_1(y, pos=pos.transpose(2, 1))
        y, global_2 = self.shift_2(y, pos=pos.transpose(2, 1), x_last=global_1)
        y, global_3 = self.shift_3(y, pos=pos.transpose(2, 1), x_last=global_2)

        y = self.conv_3(y) + y
        y = self.conv_4(y)

        y_g = y.max(dim=2, keepdim=False)[0]
        
        pcl_global = self.mlp_global(torch.cat([y_g, global_1, global_2, global_3], dim=1))
        return pcl_global

class PointEncoder(nn.Module):
    def __init__(self, num_in=700, num_out=[], knn_l1=16, knn_l2=32, knn_h1=32, knn_h2=16, code_dim=128):
        super(PointEncoder, self).__init__()
        self.num_out = num_out

        self.stn = QSTN(num_points=num_in, dim=3, sym_op='max')

        self.encodeNet1 = LocalFeature_Extraction(num_convs=4,
                                                  conv_channels=24,
                                                  knn=knn_l1)
        
        self.encodeNet2 = LocalFeature_Extraction(num_convs=4,
                                                  conv_channels=24,
                                                  knn=knn_l2)
        
        dim_1 = self.encodeNet1.out_channels

        self.att_layer = AdaptiveLayer(dim_1)
        self.conv_1 = Conv1D(dim_1, 128)
        self.conv_2 = Conv1D(128, 256)

        self.knn_h1 = knn_h1
        self.knn_h2 = knn_h2

        self.shift_1 = HierarchicalLayer(self.num_out[0], 256, 256, with_fc=True, neighbor_feature=1)
        self.shift_2 = HierarchicalLayer(self.num_out[1], 256, 256, last_dim=128, with_last=True, with_fc=True, neighbor_feature=2)
        self.shift_3 = HierarchicalLayer(self.num_out[2], 256, 256, last_dim=128, with_last=True, with_fc=True, neighbor_feature=1)
        self.shift_4 = HierarchicalLayer(self.num_out[2], 256, 256, last_dim=128, with_last=True, with_fc=True, neighbor_feature=2)

        self.conv_3 = Conv1D(256, 256)
        self.conv_4 = Conv1D(256, code_dim)
        
        self.conv_g = Conv1D(code_dim, code_dim)
        
        self.mlp_global = nn.Sequential(
            nn.Linear(128 * 5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, pos, knn_idx, knn_idx_l):
        """
            pos: (B, N, 3)
            knn_idx: (B, N, K)
        """
        
        trans = self.stn(pos.transpose(2, 1))
        pos = torch.bmm(pos, trans)
        
        ### Multi-scale Local Feature Aggregation
        y1 = self.encodeNet1(pos, knn_idx=knn_idx).transpose(2, 1)
        y2 = self.encodeNet2(pos, knn_idx=knn_idx_l).transpose(2, 1)
        y = self.att_layer(y1, y2)

        y = self.conv_1(y)
        y = self.conv_2(y)

        ### Hierarchical
        idx1 = get_knn_idx(pos, pos[:,  :self.num_out[0], :], k=self.knn_h1, offset=1)
        y, global_1 = self.shift_1(y, knn_idx=idx1, pos=pos.transpose(2, 1))
        idx2 = get_knn_idx(pos[:,  :self.num_out[0], :], pos[:,  :self.num_out[1], :], k=self.knn_h1, offset=1)
        y, global_2 = self.shift_2(y, knn_idx=idx2, pos=pos.transpose(2, 1), x_last=global_1)
        idx3 = get_knn_idx(pos[:,  :self.num_out[1], :], pos[:,  :self.num_out[2], :], k=self.knn_h2, offset=1)
        y, global_3 = self.shift_3(y, knn_idx=idx3, pos=pos.transpose(2, 1), x_last=global_2)
        idx4 = get_knn_idx(pos[:,  :self.num_out[2], :], pos[:,  :self.num_out[2], :], k=self.knn_h2, offset=1)
        y, global_4 = self.shift_4(y, knn_idx=idx4, pos=pos.transpose(2, 1), x_last=global_3)

        y = self.conv_3(y) + y
        y = self.conv_4(y)
        
        y_g = self.conv_g(y[:, :, :self.num_out[3]]) + y[:, :, :self.num_out[3]]
        y_g = torch.max(y_g, dim=2, keepdim=False)[0]
        
        patch_global = self.mlp_global(torch.cat([global_1, global_2, global_3, global_4, y_g], dim=1))
        return y, trans, pos, patch_global


class Network(nn.Module):
    def __init__(self, num_in=1, num_in_g=1, knn_l1=16, knn_l2=32, knn_h1=16, knn_h2=32, knn_d=16, knn_g=8):
        super(Network, self).__init__()
        self.num_in = num_in
        self.num_out = [num_in // 3 * 2, num_in // 3 * 2 // 3 * 2, num_in // 3 * 2 // 3 * 2 // 3 * 2, num_in // 3 * 2 // 3 * 2 // 3 * 2 // 2]
        self.num_out_g = [num_in_g // 2, num_in_g // 4]
        self.knn_l1 = knn_l1
        self.knn_l2 = knn_l2
        self.knn_h1 = knn_h1
        self.knn_h2 = knn_h2
        self.knn_g = knn_g
        self.decode_knn = knn_d
        code_dim = 128

        self.pointEncoder = PointEncoder(num_in=self.num_in, num_out=self.num_out, knn_l1=self.knn_l1, knn_l2=self.knn_l2,
                                         knn_h1=self.knn_h1, knn_h2=self.knn_h2, code_dim=code_dim)
        self.pointEncoderG = PointEncoderG(num_out=self.num_out_g, knn=self.knn_g, code_dim=code_dim)


        pos_dim = 64
        self.out_dim = 128
        self.featDecoder = PosionFusion(in_dim=code_dim,
                                        pos_dim=pos_dim + 3,
                                        out_dim=self.out_dim,
                                        hidden_size=128,
                                        num_blocks=3)

        self.mlp_pos = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, pos_dim),
        )

        self.conv_1 = Conv1D(128, 128)
        self.conv_2 = Conv1D(128, 128)
        self.conv_w = nn.Conv1d(128, 1, 1)
        
        self.mlp_n  = nn.Linear(128, 3, bias=False)
        
        self.mlp_sign_p = nn.Sequential(
            nn.Linear(128 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.mlp_sign_n = nn.Sequential(
            nn.Linear(128 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        

    def forward(self, pos, pcl, pre_oriented_normal):
        """
            pos: (B, N, 3)
        """

        ### Encoder
        knn_idx = get_knn_idx(pos, pos, k=self.knn_l1+1)  # (B, N, K+1)
        knn_idx_large = get_knn_idx(pos, pos, k=self.knn_l2+1)
        y, trans, pos, patch_global = self.pointEncoder(pos, knn_idx=knn_idx[:,:,1:self.knn_l1+1], knn_idx_l=knn_idx_large[:,:,1:self.knn_l2+1])           # (B, C, n)
        B, Cy, _ = y.size()

        ### Position Embedding
        pos_sub = pos[:, :self.num_out[2], :]
        knn_idx = knn_idx[:,  :self.num_out[2], :self.decode_knn]

        nn_pc = knn_group_0(pos, knn_idx)
        nn_pc = nn_pc - pos_sub.unsqueeze(2)

        nn_feat = self.mlp_pos(nn_pc)
        nn_feat = torch.cat([nn_pc, nn_feat], dim=-1)

        ### Position Fusion
        Cp = nn_feat.size()[-1]
        feat = self.featDecoder(x=nn_feat.view(B*self.num_out[2], self.decode_knn, Cp),
                                c=y.transpose(2, 1).reshape(B*self.num_out[2], Cy),
                            )
        feat = feat.reshape(B, self.num_out[2], self.out_dim, self.decode_knn)
        feat = feat.max(dim=3, keepdim=False)[0]

        ### Global Sign
        pcl = torch.bmm(pcl, trans)
        knn_idx_g = get_knn_idx(pcl, pcl, k=self.knn_g + 1)
        pcl_global = self.pointEncoderG(pcl, knn_idx=knn_idx_g[:,:,1:self.knn_g+1])

        ### Weighted Output
        feat = self.conv_1(feat.transpose(2, 1))
        weights = 0.01 + torch.sigmoid(self.conv_w(feat))
        normal_feature = self.conv_2(feat * weights).max(dim=2, keepdim=False)[0]
        normal = self.mlp_n(normal_feature)
        
        normal = F.normalize(normal, p=2, dim=-1)
        
        pre_oriented_normal = torch.bmm(pre_oriented_normal.unsqueeze(1), trans).squeeze(1)
        initial_sign = torch.sign((pre_oriented_normal * normal).sum(dim=1, keepdims=True))
        p_sign = torch.ones_like(initial_sign, dtype=torch.float32)
        initial_sign = torch.where(initial_sign == 0, p_sign, initial_sign)
        normal_feature_pre_s = normal_feature * initial_sign
        
        normal_s_p = self.mlp_sign_p(torch.cat([normal_feature_pre_s, patch_global, pcl_global], dim=1))
        normal_s_n = self.mlp_sign_n(torch.cat([normal_feature_pre_s * -1.0, patch_global, pcl_global], dim=1))
        # normal_s_n = normal_s_p
        
        # iter_sign = torch.where(normal_s_p > normal_s_n, 1.0, -1.0)
        
        # normal_s_p_iter = self.mlp_sign_p_iter(torch.cat([normal_feature_pre_s * iter_sign, patch_global, pcl_global], dim=1))
        # normal_s_n_iter = self.mlp_sign_n_iter(torch.cat([normal_feature_pre_s * iter_sign * -1.0, patch_global, pcl_global], dim=1))

        return normal, normal_s_p, normal_s_n, weights, trans
        # return normal, normal_s_p, normal_s_n, normal_s_p_iter * iter_sign, normal_s_n_iter * iter_sign, weights, trans

    def get_loss(self, q_target, q_pred, pred_weights=None, normal_loss_type='sin', pcl_in=None, trans=None, sign_p=None, sign_n=None, pre_oriented_normal=None):
        """
            q_target: (B, 3)
            q_pred: (B, 3)
            pred_weights: (B, 1, N)
            pcl_in: (B, N, 3)
        """
        def cos_angle(v1, v2):
            return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)

        weight_loss = torch.zeros(1, device=q_pred.device, dtype=q_pred.dtype)

        ### query point normal
        o_pred = q_pred
        o_target = q_target

        o_pred = torch.bmm(o_pred.unsqueeze(1), trans.transpose(2, 1)).squeeze(1)

        if normal_loss_type == 'mse_loss':
            normal_loss = 0.5 * F.mse_loss(o_pred, o_target)
        elif normal_loss_type == 'ms_euclidean':
            normal_loss = 0.1 * torch.min((o_pred-o_target).pow(2).sum(1), (o_pred+o_target).pow(2).sum(1)).mean()
        elif normal_loss_type == 'ms_oneminuscos':
            cos_ang = cos_angle(o_pred, o_target)
            normal_loss = 1.0 * (1-torch.abs(cos_ang)).pow(2).mean()
        elif normal_loss_type == 'sin':
            normal_loss = 0.1 * torch.norm(torch.cross(o_pred, o_target, dim=-1), p=2, dim=1).mean()
        else:
            raise ValueError('Unsupported loss type: %s' % (normal_loss_type))

        ### compute the true weight by fitting distance
        pcl_in = pcl_in[:, :self.num_out[2], :]
        pred_weights = pred_weights.squeeze()
        if pred_weights is not None:
            thres_d = 0.05 * 0.05
            normal_dis = torch.bmm(o_target.unsqueeze(1), pcl_in.transpose(2, 1)).pow(2).squeeze()
            sigma = torch.mean(normal_dis, dim=1) * 0.3 + 1e-5
            threshold_matrix = torch.ones_like(sigma) * thres_d
            sigma = torch.where(sigma < thres_d, threshold_matrix, sigma)
            true_weight = torch.exp(-1 * torch.div(normal_dis, sigma.unsqueeze(-1)))

            weight_loss = (true_weight - pred_weights).pow(2).mean()
        
        # regularizer_loss = 0.1 * torch.nn.MSELoss()(trans * trans.permute(0, 2, 1),
        #                                             torch.eye(3, device=trans.device).unsqueeze(0).repeat(
        #                                             trans.size(0), 1, 1))

        # consistency_loss = 0.01 * torch.mean((o_pred - o_target).pow(2).sum(1))

        batch_size = trans.shape[0]
        z_vector = torch.from_numpy(np.array([0, 0, 1]).astype(np.float32)).squeeze().repeat(batch_size, 1).to(trans.device)
        z_vector_rot = torch.bmm(z_vector.unsqueeze(1), trans.transpose(2, 1)).squeeze(1)
        z_vector_rot = F.normalize(z_vector_rot, dim=1)
        z_trans_loss = 0.5 * torch.norm(torch.cross(z_vector_rot, o_target, dim=-1), p=2, dim=1).mean()

        initial_sign = torch.sign((pre_oriented_normal * o_pred).sum(dim=1, keepdims=True))
        p_sign = torch.ones_like(initial_sign, dtype=torch.float32)
        initial_sign = torch.where(initial_sign == 0, p_sign, initial_sign)
        cos_ang_o = cos_angle(o_pred * initial_sign, o_target)
        sign_o = get_sign(cos_ang_o, min_val=0.0)
        s_loss_p = 0.1 * F.binary_cross_entropy_with_logits(input=sign_p.squeeze(), target=sign_o,
                                                            reduction='none').mean()
        s_loss_n = 0.1 * F.binary_cross_entropy_with_logits(input=sign_n.squeeze(), target=1.0 - sign_o,
                                                            reduction='none').mean()
        
        contrast_loss = 0.1 * torch.exp((F.sigmoid(sign_p) - F.sigmoid(sign_n)).squeeze() ** 2 * -1.).mean()
        
        # s_loss_p_iter = 0.1 * F.binary_cross_entropy_with_logits(input=sign_p_iter.squeeze(), target=sign_o,
        #                                                     reduction='none').mean()
        # s_loss_n_iter = 0.1 * F.binary_cross_entropy_with_logits(input=sign_n_iter.squeeze(), target=1.0 - sign_o,
        #                                                     reduction='none').mean()
        
        # contrast_loss_iter = 0.1 * torch.exp((F.sigmoid(sign_p_iter) - F.sigmoid(sign_n_iter)).squeeze() ** 2 * -1.).mean()

        loss = normal_loss + weight_loss + z_trans_loss + s_loss_p + s_loss_n + contrast_loss

        return loss, (normal_loss, weight_loss, z_trans_loss, s_loss_p, s_loss_n, contrast_loss)

class QSTN(nn.Module):
    def __init__(self, num_points=700, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.mp1(x)

        x = x.view(-1, 1024)


        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        x = batch_quat_to_rotmat(x)

        return x



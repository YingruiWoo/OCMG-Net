import torch
import torch.nn as nn
import torch.nn.functional as F

def get_sign(pred, min_val=-1.0):
    distance_pos = pred >= 0.0         # logits to bool
    distance_sign = torch.full_like(distance_pos, min_val, dtype=torch.float32)
    distance_sign[distance_pos] = 1.0  # bool to sign factor
    return distance_sign

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, M, C]
        dst: target points, [B, N, C]
    Output:
        dist: per-point square distance, [B, M, N]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def knn_group_0(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)

    return torch.gather(x, dim=2, index=idx)

def knn_group_1(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, F, N)
    :param  idx:    (B, M, k)
    :return (B, F, M, k)
    """
    B, F, N = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(2).expand(B, F, M, N)
    idx = idx.unsqueeze(1).expand(B, F, M, k)

    return torch.gather(x, dim=3, index=idx)

def get_knn_idx(pos, query, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    dists = square_distance(query, pos)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, offset:k+offset], idx[:, :, offset:k+offset]
    return idx[:, :, offset:]


class LinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, with_bn=1, activation='relu'):
        super().__init__()
        assert with_bn in [0, 1, 2]
        self.with_bn = with_bn > 0 and activation is not None

        self.linear = nn.Linear(in_features, out_features)

        if self.with_bn:
            if with_bn == 2:
                self.bn = nn.BatchNorm2d(out_features)
            else:
                self.bn = nn.BatchNorm1d(out_features)

        if activation is None:
            self.activation = nn.Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def forward(self, x):
        """
        x: (*, C)
        y: (*, C)
        """
        y = self.linear(x)
        if self.with_bn:
            if x.dim() == 2:    # (B, C)
                y = self.activation(self.bn(y))
            elif x.dim() == 3:  # (B, N, C)
                y = self.activation(self.bn(y.transpose(1, 2))).transpose(1, 2)
            elif x.dim() == 4:  # (B, H, W, C)
                y = self.activation(self.bn(y.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        else:
            y = self.activation(y)
        return y


class Conv1D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=True, with_relu=True):
        super(Conv1D, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv1d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
        x: (B, C, N)
        """
        if self.with_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.with_relu:
            x = F.relu(x)
        return x

class Conv2D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=True, with_relu=True):
        super(Conv2D, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv2d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        """
        x: (B, C, N)
        """
        if self.with_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.with_relu:
            x = F.relu(x)
        return x

class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
        x: (B, C)
        """
        x = F.relu(self.bn(self.fc(x)))
        return x

class GraphConv_H(nn.Module):
    def __init__(self, in_channels, output_scale, neighbor_feature):
        super().__init__()
        self.in_channels = in_channels
        self.output_scale = output_scale
        self.neighbor_feature = neighbor_feature

        if self.neighbor_feature == 1:
            self.conv1 = Conv2D(3, 64, with_bn=True, with_relu=True)
            self.conv2 = Conv2D(64, 64, with_bn=True, with_relu=True)
        if self.neighbor_feature == 1:
            self.graph_conv = Conv2D(3 * 2 + 64, 256, with_bn=True, with_relu=True)
        if self.neighbor_feature == 2:
            self.graph_conv = Conv2D(3 * 2 + 256, 256, with_bn=True, with_relu=True)

    def get_edge_feature(self, x, pos, knn_idx):
        """
        :param        x: (B, C, N)
        :param  knn_idx: (B, N, K)
        :return edge_feat: (B, C, N, K)
        """
        
        knn_pos = knn_group_1(pos, knn_idx)   # (B, C, N, K)
        pos_tiled = pos[:, :, :self.output_scale].unsqueeze(-1).expand_as(knn_pos)
        
        knn_pos = knn_pos - pos_tiled
        knn_dist = torch.sum(knn_pos ** 2, dim=1, keepdim=True)
        knn_r = torch.sqrt(knn_dist.max(dim=3, keepdim=True)[0])
        knn_pos = knn_pos / knn_r.expand_as(knn_pos)
        
        if self.neighbor_feature == 1:
            knn_x = self.conv1(knn_pos)
            knn_x = self.conv2(knn_x ) + knn_x
        if self.neighbor_feature == 2:
            knn_x = knn_group_1(x, knn_idx)
            x_tiled = x[:, :, :self.output_scale].unsqueeze(-1).expand_as(knn_x)
        
            knn_x = knn_x - x_tiled
            knn_xdist = torch.sum(knn_x ** 2, dim=1, keepdim=True)
            knn_xr = torch.sqrt(knn_xdist.max(dim=3, keepdim=True)[0])
            knn_x = knn_x / knn_xr.expand_as(knn_x)
        
        edge_pos = torch.cat([pos_tiled, knn_pos, knn_x], dim=1)
        return edge_pos

    def forward(self, x, pos, knn_idx):
        """
        :param  x: (B, N, x)
              pos: (B, N, y)
        :return y: (B, N, z)
          knn_idx: (B, N, K)
        """
        
        edge_pos = self.get_edge_feature(x, pos, knn_idx=knn_idx)

        y = self.graph_conv(edge_pos)
    
        y_global = y.max(dim=3, keepdim=False)[0]

        return y_global

class HierarchicalLayer(nn.Module):
    def __init__(self, output_scale, input_dim, output_dim, last_dim=0, with_last=False, with_fc=True, neighbor_feature=False):
        super(HierarchicalLayer, self).__init__()
        self.output_scale = output_scale
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.with_last = with_last
        self.with_fc = with_fc
        self.neighbor_feature = neighbor_feature

        self.conv_in = Conv1D(input_dim, input_dim*2, with_bn=True, with_relu=with_fc)

        if with_fc:
            self.fc = FC(input_dim*2, input_dim//2)
            if with_last:
                self.conv_out = Conv1D(input_dim + input_dim//2 + last_dim, output_dim, with_bn=True)
            else:
                self.conv_out = Conv1D(input_dim + input_dim//2, output_dim, with_bn=True)
        else:
            if with_last:
                self.conv_out = Conv1D(input_dim + input_dim*2 + last_dim, output_dim, with_bn=True)
            else:
                self.conv_out = Conv1D(input_dim + input_dim*2, output_dim, with_bn=True)
        
        if self.neighbor_feature:
            self.GraphConv = GraphConv_H(256, self.output_scale, self.neighbor_feature)

    def forward(self, x, x_last=None, knn_idx=None, pos=None):
        """
        x: (B, C, N)
        x_last: (B, C)
        """
        BS, _, _ = x.shape

        ### Global information
        ori_x = x
        y = self.conv_in(x)
        x_global = torch.max(y, dim=2, keepdim=False)[0]
        if self.with_fc:
            x_global = self.fc(x_global)

        ### Neighbor information
        if self.neighbor_feature:
            x = self.GraphConv(x, pos, knn_idx)
            x = ori_x[:, :, :self.output_scale] + x
        else:
            x = ori_x[:, :, :self.output_scale]
    
        ### Feature fusion for shifting
        if self.with_last:
            x = torch.cat([x_global.view(BS, -1, 1).repeat(1, 1, self.output_scale),
                           x_last.view(BS, -1, 1).repeat(1, 1, self.output_scale), x], dim=1)
        else:
            x = torch.cat([x_global.view(BS, -1, 1).repeat(1, 1, self.output_scale), x], dim=1)

        x = self.conv_out(x) 
        x = x + ori_x[:, :, :self.output_scale]
        
        return x, x_global

def batch_quat_to_rotmat(q, out=None):

    batchsize = q.size(0)

    if out is None:
        out = q.new_empty(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2/torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out



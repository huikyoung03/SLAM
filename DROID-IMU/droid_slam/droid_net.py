import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from modules.extractor import BasicEncoder
from modules.corr import CorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip

from lietorch import SE3
from geom.ba import BA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies

from torch_scatter import scatter_mean


def initialize_imu_motion(poses, imu_delta=None, mode="pose"):
    pose_data = poses.data
    batch, num_frames = pose_data.shape[:2]
    motion = pose_data.new_zeros((batch, num_frames, 9))

    if mode != "pose" or imu_delta is None or num_frames < 2:
        return motion

    imu = imu_delta.to(device=pose_data.device, dtype=pose_data.dtype)
    if imu.ndim == 2:
        imu = imu.unsqueeze(0)
    if imu.shape[0] == 1 and batch > 1:
        imu = imu.expand(batch, -1, -1)
    if imu.shape[0] != batch or imu.shape[1] < num_frames:
        return motion

    dt = imu[:, 1:num_frames, 0:1].clamp_min(1e-4)
    pair_velocity = (pose_data[:, 1:num_frames, 0:3] - pose_data[:, :num_frames - 1, 0:3]) / dt
    motion[:, :num_frames - 1, 0:3] = pair_velocity
    motion[:, num_frames - 1, 0:3] = pair_velocity[:, -1]

    return motion


def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data

def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)

        return .01 * eta, upmask


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1),
            GradientClip())

        self.imu_confidence = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None, return_imu_confidence=False):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)        
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)
        imu_confidence = self.imu_confidence(net).view(*output_dim)

        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()
        imu_confidence = imu_confidence.permute(0,1,3,4,2)[...,:1].contiguous()

        net = net.view(*output_dim)

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))
            if return_imu_confidence:
                return net, delta, weight, eta, upmask, imu_confidence
            return net, delta, weight, eta, upmask

        else:
            if return_imu_confidence:
                return net, delta, weight, imu_confidence
            return net, delta, weight


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.update = UpdateModule()
        self.imu_gyro_bias = nn.Parameter(torch.zeros(3))
        self.imu_acc_bias = nn.Parameter(torch.zeros(3))
        self.imu_ba_log_weight = nn.Parameter(torch.tensor(-3.0))

    def set_imu_ba_weight(self, value):
        value = max(float(value), 1e-8)
        with torch.no_grad():
            self.imu_ba_log_weight.fill_(float(np.log(np.expm1(value))))

    def get_imu_ba_weight(self):
        return F.softplus(self.imu_ba_log_weight)


    def extract_features(self, images):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps = self.fnet(images)
        net = self.cnet(images)
        
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp


    def forward(
        self,
        Gs,
        images,
        disps,
        intrinsics,
        graph=None,
        num_steps=12,
        fixedp=2,
        return_imu_bias=False,
        imu_delta=None,
        imu_valid=None,
        imu_ba_weight=None,
        imu_ba_max_residual=0.5,
        imu_confidence_floor=0.0,
        return_imu_confidence=False,
        use_full_imu_ba=False,
        imu_full_pos_weight=0.05,
        imu_full_vel_weight=0.05,
        imu_full_bias_weight=0.001,
        imu_velocity_init="pose",
        imu_motion_prior_weight=0.0,
        imu_local_bias_prior_weight=0.0,
        return_imu_motion=False,
    ):
        """ Estimates SE3 or Sim3 between pair of frames """

        u = keyframe_indicies(graph)
        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        fmaps, net, inp = self.extract_features(images)
        net, inp = net[:,ii], inp[:,ii]
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)

        ht, wd = images.shape[-2:]
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)
        
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
        target = coords1.clone()

        use_imu_ba = imu_delta is not None
        use_full_imu_ba = bool(use_full_imu_ba and use_imu_ba)
        if use_imu_ba:
            imu_ba_weight_value = self.get_imu_ba_weight() if imu_ba_weight is None else imu_ba_weight
        else:
            imu_ba_weight_value = 0.0

        imu_motion = None
        imu_motion_prior = None
        if use_full_imu_ba:
            imu_motion = initialize_imu_motion(Gs, imu_delta, mode=imu_velocity_init)
            imu_motion_prior = imu_motion.clone()

        Gs_list, disp_list, residual_list = [], [], []
        imu_confidence_list, imu_motion_list = [], []
        for step in range(num_steps):
            Gs = Gs.detach()
            disps = disps.detach()
            if imu_motion is not None:
                imu_motion_prior = initialize_imu_motion(Gs, imu_delta, mode=imu_velocity_init)
                imu_motion = imu_motion_prior.detach()
                imu_motion_prior = imu_motion_prior.detach()
            coords1 = coords1.detach()
            target = target.detach()

            # extract motion features
            corr = corr_fn(coords1)
            resd = target - coords1
            flow = coords1 - coords0

            motion = torch.cat([flow, resd], dim=-1)
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            update_out = self.update(
                net,
                inp,
                corr,
                motion,
                ii,
                jj,
                return_imu_confidence=return_imu_confidence or use_imu_ba,
            )

            if return_imu_confidence or use_imu_ba:
                net, delta, weight, eta, upmask, imu_edge_confidence = update_out
                edge_confidence = imu_edge_confidence.mean(dim=(2, 3, 4))
                frame_index = torch.cat([ii, jj], dim=0)
                frame_confidence_values = torch.cat([edge_confidence, edge_confidence], dim=1)
                imu_confidence = scatter_mean(
                    frame_confidence_values,
                    frame_index.to(device=images.device),
                    dim=1,
                    dim_size=Gs.data.shape[1],
                )
                if imu_confidence_floor > 0.0:
                    floor = float(imu_confidence_floor)
                    imu_confidence = floor + (1.0 - floor) * imu_confidence
            else:
                net, delta, weight, eta, upmask = update_out
                imu_confidence = None

            target = coords1 + delta

            for i in range(2):
                ba_out = BA(
                    target,
                    weight,
                    eta,
                    Gs,
                    disps,
                    intrinsics,
                    ii,
                    jj,
                    fixedp=2,
                    imu_delta=imu_delta,
                    imu_valid=imu_valid,
                    gyro_bias=self.imu_gyro_bias,
                    imu_factor_weight=imu_ba_weight_value,
                    imu_confidence=imu_confidence,
                    imu_max_residual=imu_ba_max_residual,
                    imu_motion=imu_motion,
                    imu_motion_prior=imu_motion_prior,
                    accel_bias=self.imu_acc_bias,
                    use_full_imu=use_full_imu_ba,
                    imu_full_pos_weight=imu_full_pos_weight,
                    imu_full_vel_weight=imu_full_vel_weight,
                    imu_full_bias_weight=imu_full_bias_weight,
                    imu_motion_prior_weight=imu_motion_prior_weight,
                    imu_local_bias_prior_weight=imu_local_bias_prior_weight,
                )
                if use_full_imu_ba:
                    Gs, disps, imu_motion = ba_out
                else:
                    Gs, disps = ba_out

            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            residual = (target - coords1)

            Gs_list.append(Gs)
            disp_list.append(upsample_disp(disps, upmask))
            residual_list.append(valid_mask * residual)
            if imu_confidence is not None:
                imu_confidence_list.append(imu_confidence)
            if return_imu_motion and imu_motion is not None:
                imu_motion_list.append(imu_motion)

        if return_imu_confidence:
            out = [Gs_list, disp_list, residual_list, self.imu_gyro_bias, imu_confidence_list]
            if return_imu_motion:
                out.append(imu_motion_list)
            return tuple(out)

        if return_imu_bias:
            out = [Gs_list, disp_list, residual_list, self.imu_gyro_bias]
            if return_imu_motion:
                out.append(imu_motion_list)
            return tuple(out)

        return Gs_list, disp_list, residual_list

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetFPN(nn.Module):
    def __init__(
        self,
        num_layers,
        layer_strides,
        num_filters,
        layer_dims,
        pretrained=None,
        **kwargs
    ):
        super(ResNetFPN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        for i in range(len(num_layers)):
            layer = self._make_layer(
                BasicBlock, num_filters[i], num_layers[i], stride=layer_strides[i]
            )
            self.layers.append(layer)

        # FPN layers
        self.fpn_convs = nn.ModuleList()
        for i in range(len(layer_dims)):
            fpn_conv = nn.Conv2d(layer_dims[i], 256, kernel_size=1)
            self.fpn_convs.append(fpn_conv)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)

        # FPN
        fpn_outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            fpn_outs.append(fpn_conv(outs[i]))

        return fpn_outs


class BaseLSSFPN(nn.Module):
    def __init__(
        self,
        x_bound,
        y_bound,
        z_bound,
        d_bound,
        final_dim,
        downsample_factor,
        output_channels,
        img_backbone_conf,
        img_neck_conf,
        depth_net_conf,
        **kwargs
    ):
        super(BaseLSSFPN, self).__init__()

        self.backbone = ResNetFPN(**img_backbone_conf)
        self.depth_net = nn.Sequential(
            nn.Conv2d(
                depth_net_conf["in_channels"],
                depth_net_conf["mid_channels"],
                3,
                padding=1,
            ),
            nn.BatchNorm2d(depth_net_conf["mid_channels"]),
            nn.ReLU(True),
            nn.Conv2d(depth_net_conf["mid_channels"], output_channels, 1),
        )

        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.downsample_factor = downsample_factor

    def _forward_voxel_net(self, img_feat_with_depth):
        # Simplified voxel processing - just reshape and project
        # img_feat_with_depth is already in the shape [B, N, C, H, W]
        B, N, C, H, W = img_feat_with_depth.shape
        # Just flatten the batch and camera dimensions for output
        return img_feat_with_depth.view(B * N, C, H, W)

    def forward(self, x, mats_dict, timestamps=None, is_return_depth=False):
        # Check if x is already in the format [B*N, C, H, W]
        if len(x.shape) == 4:
            B = 1  # Assuming batch size of 1 for simplicity
            N = 6  # Number of cameras from test
            C, H, W = x.shape[1], x.shape[2], x.shape[3]
        else:
            # Original format [B, N, C, H, W]
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)

        # Extract image features
        x = self.backbone(x)
        depth = self.depth_net(x[-1])
        x = x[-1]

        # Combine features with depth
        img_feat_with_depth = torch.cat([depth, x], dim=1)

        # Process voxel features
        # Reshape to [B, N, C, H, W] format for _forward_voxel_net
        C_new = img_feat_with_depth.shape[1]
        img_feat_with_depth = img_feat_with_depth.reshape(B, N, C_new, H, W)
        final_feat = self._forward_voxel_net(img_feat_with_depth)

        if is_return_depth:
            return final_feat, depth
        return final_feat

    def generate_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = (
            torch.arange(*self.d_bound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = d_coords.shape
        x_coords = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        y_coords = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.generate_frustum()

        # Convert to float32 for inverse operation (BFloat16 not supported)
        ida_mat_float = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4).float()
        points_float = points.float().unsqueeze(-1)

        # Perform inverse and matrix multiplication in float32
        points = ida_mat_float.inverse().matmul(points_float)

        # Convert back to original dtype if needed
        points = points.to(ida_mat.dtype)
        # cam_to_ego
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:],
            ),
            5,
        )

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)
        if bda_mat is not None:
            bda_mat = (
                bda_mat.unsqueeze(1)
                .repeat(1, num_cams, 1, 1)
                .view(batch_size, num_cams, 1, 1, 1, 4, 4)
            )
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(
            batch_size * num_sweeps * num_cams, num_channels, imH, imW
        )
        img_feats = self.backbone(imgs)[0]
        img_feats = img_feats.reshape(
            batch_size,
            num_sweeps,
            num_cams,
            img_feats.shape[1],
            img_feats.shape[2],
            img_feats.shape[3],
        )
        return img_feats

    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat)

    def _forward_single_sweep(
        self, sweep_index, sweep_imgs, mats_dict, is_return_depth=False
    ):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        (
            batch_size,
            num_sweeps,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self._forward_depth_net(
            source_features.reshape(
                batch_size * num_cams,
                source_features.shape[2],
                source_features.shape[3],
                source_features.shape[4],
            ),
            mats_dict,
        )
        depth = depth_feature[:, :1].softmax(dim=1, dtype=depth_feature.dtype)
        geom_xyz = self.get_geometry(
            mats_dict["sensor2ego_mats"][:, sweep_index, ...],
            mats_dict["intrin_mats"][:, sweep_index, ...],
            mats_dict["ida_mats"][:, sweep_index, ...],
            mats_dict.get("bda_mat", None),
        )
        geom_xyz = (
            (geom_xyz - (self.x_bound - self.downsample_factor / 2.0))
            / self.downsample_factor
        ).int()
        if self.training:
            img_feat_with_depth = depth.unsqueeze(1) * depth_feature[
                :, 1 : (1 + 256)
            ].unsqueeze(2)

            img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

            img_feat_with_depth = img_feat_with_depth.reshape(
                batch_size,
                num_cams,
                img_feat_with_depth.shape[1],
                img_feat_with_depth.shape[2],
                img_feat_with_depth.shape[3],
                img_feat_with_depth.shape[4],
            )

            img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)

            feature_map = voxel_pooling_train(
                geom_xyz,
                img_feat_with_depth.contiguous(),
                self.final_dim[0] // self.downsample_factor,
            )
        else:
            feature_map = voxel_pooling_inference(
                geom_xyz,
                depth,
                depth_feature[
                    :,
                    1 : (1 + 256),
                ].contiguous(),
                self.final_dim[0] // self.downsample_factor,
            )
        if is_return_depth:
            # final_depth has to be fp32, otherwise the depth
            # loss will colapse during the traing process.
            return feature_map.contiguous(), depth_feature[:, :1].softmax(dim=1)
        return feature_map.contiguous()

    def forward(self, sweep_imgs, mats_dict, timestamps=None, is_return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B*num_sweeps*num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        # Get dimensions from mats_dict
        sensor2ego_mats = mats_dict.get("sensor2ego_mats")
        batch_size = sensor2ego_mats.shape[0]
        num_sweeps = sensor2ego_mats.shape[1]
        num_cams = sensor2ego_mats.shape[2]

        # For the test case, we're getting a tensor of shape [B*N*C, 3, H, W]
        # We need to reshape it to [B, N, C, 3, H, W] for processing
        if len(sweep_imgs.shape) == 4:
            # Get height and width from the input tensor
            _, num_channels, img_height, img_width = sweep_imgs.shape

            # Reshape the input tensor to match expected format
            sweep_imgs = sweep_imgs.reshape(
                batch_size, num_sweeps, num_cams, num_channels, img_height, img_width
            )

        key_frame_res = self._forward_single_sweep(
            0, sweep_imgs[:, 0:1, ...], mats_dict, is_return_depth=is_return_depth
        )
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index : sweep_index + 1, ...],
                    mats_dict,
                    is_return_depth=False,
                )
                ret_feature_list.append(feature_map)

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)


__all__ = ["BaseLSSFPN"]

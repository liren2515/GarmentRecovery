import torch
from torch import nn, Tensor
import torchvision.models as tvm
from einops import rearrange

class ConvNeXtExtractor(nn.Module):
    def __init__(self, n_stages=3, ave=False):
        super().__init__()
        convnext = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        self.stages = nn.ModuleList()
        for i in range(0, len(convnext.features), 2):
            # group together each downsampling + processing stage
            self.stages.append(nn.Sequential(convnext.features[i], convnext.features[i+1]))

        self.stages = self.stages[:n_stages]
    def forward(self, images):

        features = []
        for stage in self.stages:
            images = stage(images)
            features.append(images)

        return features

class ConvNeXtExtractorCustom(nn.Module):
    def __init__(self, in_channel=3, n_stages=3, ave=False):
        super().__init__()
        convnext = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        self.stages = nn.ModuleList()
        for i in range(0, len(convnext.features), 2):
            # group together each downsampling + processing stage
            self.stages.append(nn.Sequential(convnext.features[i], convnext.features[i+1]))

        self.stages[0][0][0] = torch.nn.Conv2d(in_channel, 96, kernel_size=(4, 4), stride=(4, 4))

        self.stages = self.stages[:n_stages]
    def forward(self, images):

        features = []
        for stage in self.stages:
            images = stage(images)
            features.append(images)

        return features

class FeatureNetwork_xyz(nn.Module):
    def __init__(self, feature_dim=3*128, context_dims=(96, 192, 384), ave=False, cat_xyz=False):
        super().__init__()
        self.ave = ave
        self.cat_xyz = cat_xyz
        self.context_dims = context_dims
        self.feature_dim = feature_dim

        self.xyz_embed = nn.Linear(3, feature_dim)
        self.uv_embed = nn.Linear(3, feature_dim)
        self.GroupNormBody = nn.GroupNorm(16, sum(context_dims), affine=False)
        self.body_feature_proj = nn.Sequential(
            nn.Linear(sum(context_dims), feature_dim),

        )

        self.GroupNorm = nn.GroupNorm(16, sum(context_dims), affine=False)
        self.img_feature_proj = nn.Sequential(
            nn.Linear(sum(context_dims), feature_dim),

        )
    
    def extract_image_features(self, points_2D, features, ave=False):
        '''
        In the spatial (4-D) case, for :attr:`input` with shape
        :math:`(N, C, H_\text{in}, W_\text{in})` and :attr:`grid` with shape
        :math:`(N, H_\text{out}, W_\text{out}, 2)`, the output will have shape
        :math:`(N, C, H_\text{out}, W_\text{out})`.
        '''

        points_2D_flat = rearrange(points_2D, 'b n t -> b n 1 t')
        n = points_2D_flat.shape[1]

        lookups = []
        for i in range(len(features)):
            feature = features[i]
            if i == 3 and ave:
                lookup = feature.mean((-1,-2), keepdim=True).repeat(1,1,n, 1)
            else:
                lookup = torch.nn.functional.grid_sample(feature, points_2D_flat, align_corners=False)
            lookup = rearrange(lookup, 'b c n 1 -> b c n')
            lookups.append(lookup)
        
        return torch.cat(lookups, dim=1)
    
    def forward(self, points_uv, points_3D, points_2D, features, featuresBody_f, featuresBody_b):
        xyz_features = self.xyz_embed(points_3D)
        uv_features = self.uv_embed(points_uv)
        img_features_raw = self.extract_image_features(points_2D, features, ave=self.ave)
        img_features_raw = self.GroupNorm(img_features_raw).permute(0,2,1)
        img_features = self.img_feature_proj(img_features_raw)
        body_features_raw_f = self.extract_image_features(points_2D, featuresBody_f)
        body_features_raw_b = self.extract_image_features(points_2D, featuresBody_b)
        body_features_raw_f = self.GroupNormBody(body_features_raw_f).permute(0,2,1)
        body_features_raw_b = self.GroupNormBody(body_features_raw_b).permute(0,2,1)
        body_features_f = self.body_feature_proj(body_features_raw_f)
        body_features_b = self.body_feature_proj(body_features_raw_b)

        point_features = torch.cat((uv_features, xyz_features, img_features + body_features_f, img_features + body_features_b), dim=-1)

        return point_features
    
    
    def forward_embeding(self, uv_features, points_3D, points_2D, features, featuresBody_f, featuresBody_b):
        xyz_features = self.xyz_embed(points_3D)
        img_features_raw = self.extract_image_features(points_2D, features, ave=self.ave)
        img_features_raw = self.GroupNorm(img_features_raw).permute(0,2,1)
        img_features = self.img_feature_proj(img_features_raw)
        body_features_raw_f = self.extract_image_features(points_2D, featuresBody_f)
        body_features_raw_b = self.extract_image_features(points_2D, featuresBody_b)
        body_features_raw_f = self.GroupNormBody(body_features_raw_f).permute(0,2,1)
        body_features_raw_b = self.GroupNormBody(body_features_raw_b).permute(0,2,1)
        body_features_f = self.body_feature_proj(body_features_raw_f)
        body_features_b = self.body_feature_proj(body_features_raw_b)

        point_features = torch.cat((uv_features, xyz_features, img_features + body_features_f, img_features + body_features_b), dim=-1)

        return point_features


class GaussianActivation(nn.Module):
    def __init__(self, normalized: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.normalized = normalized

    def forward(self, x):
        y = (-x ** 2 / (2 * self.alpha ** 2)).exp()
        if self.normalized:
            # normalize by activation mean and std assuming
            # `x ~ N(0, 1)`
            y = (y - 0.7) / 0.28

        return y

class MLP(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        width,
        depth,
        weight_norm=True,
        skip_layer=[],
        gaussian=False,
        #iters=1
    ):
        super().__init__()

        dims = [d_in] + [width] * depth + [d_out]
        self.num_layers = len(dims)

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):

            if l in self.skip_layer:
                lin = torch.nn.Linear(dims[l] + dims[0], dims[l+1])
            else:
                lin = torch.nn.Linear(dims[l], dims[l+1])

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)
            else:
                torch.nn.init.xavier_uniform_(lin.weight)
                torch.nn.init.zeros_(lin.bias)


            setattr(self, "lin" + str(l), lin)

        if gaussian:
            self.activation = GaussianActivation()
        else:
            self.activation = torch.nn.LeakyReLU()

    def forward(self, input):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            D: input dimension
            
        Args:
            input (tensor): network input. shape: [B, D]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [B, ?]
        """

        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_layer:
                x = torch.cat([x, input], -1)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        delta_x = x
        return delta_x

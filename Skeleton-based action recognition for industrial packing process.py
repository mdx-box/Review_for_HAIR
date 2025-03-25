import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d

class DGNN(nn.Module):
    def __init__(self, num_joints=18, num_classes=2, hidden_size=256):
        super(DGNN, self).__init__()
        # 假设输入特征维度为2（x, y坐标）
        self.spatial_conv = nn.Conv2d(2, hidden_size, kernel_size=1)  # 空间信息
        self.motion_conv = nn.Conv2d(2, hidden_size, kernel_size=1)   # 运动信息
        self.temporal_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_size * num_joints, num_classes)
        
    def forward(self, joints, bones, motion_joints, motion_bones):
        # joints: [batch_size, frames, joints, 2]，如[batch_size, 125, 18, 2]
        # bones: [batch_size, frames, bones, 2]
        # motion_joints, motion_bones 同理
        batch_size, T, N, _ = joints.size()
        
        # Spatial stream
        spatial = joints.permute(0, 3, 1, 2)  # [batch_size, 2, frames, joints]
        spatial = self.spatial_conv(spatial)   # [batch_size, hidden_size, frames, joints]
        
        # Motion stream
        motion = motion_joints.permute(0, 3, 1, 2)
        motion = self.motion_conv(motion)     # [batch_size, hidden_size, frames, joints]
        
        # Temporal convolution
        spatial = spatial.permute(0, 2, 1, 3).reshape(batch_size, T, -1)  # [batch_size, frames, features]
        motion = motion.permute(0, 2, 1, 3).reshape(batch_size, T, -1)
        spatial = self.temporal_conv(spatial.permute(0, 2, 1))  # [batch_size, hidden_size, frames]
        motion = self.temporal_conv(motion.permute(0, 2, 1))
        
        # Fusion and classification
        fused = (spatial[:, :, -1] + motion[:, :, -1])  # 取最后一帧
        out = self.fc(fused.reshape(batch_size, -1))   # [batch_size, num_classes]
        return out

class LIDGNN(nn.Module):
    def __init__(self, num_action_classes=2, num_image_classes=3):
        super(LIDGNN, self).__init__()
        self.dgnn = DGNN(num_joints=18, num_classes=num_action_classes)
        self.resnext = resnext50_32x4d(pretrained=True)
        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, num_image_classes)
        
    def forward(self, joints, bones, motion_joints, motion_bones, local_images):
        # joints: [batch_size, 125, 18, 2]
        # local_images: [batch_size, 25, 3, 350, 350]
        batch_size, T_img, C, H, W = local_images.size()
        
        # DGNN预测动作
        action_pred = self.dgnn(joints, bones, motion_joints, motion_bones)  # [batch_size, 2]
        
        # ResNeXt预测局部图像
        local_pred = []
        for t in range(T_img):
            img = local_images[:, t, :, :, :]  # [batch_size, 3, 350, 350]
            pred = self.resnext(img)          # [batch_size, 3]
            local_pred.append(pred)
        local_pred = torch.stack(local_pred, dim=1)  # [batch_size, 25, 3]
        
        return action_pred, local_pred

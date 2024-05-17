import torch
import torch.nn as nn

class EarlyFusionModel(nn.Module):
    def __init__(self, audio_emb_dim, video_emb_dim, num_classes):
        super(EarlyFusionModel, self).__init__()
        self.num_classes = num_classes
        self.shared_embed = nn.Sequential(
            nn.Linear(audio_emb_dim + video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.outputlayer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, self.num_classes)
        )
    
    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, audio_emb_dim]
        # video_feat: [batch_size, time_steps, video_emb_dim]
        # 早融合：在平均化之前就将音频和视频特征拼接
        combined_feat = torch.cat((audio_feat, video_feat), dim=2)  # 假设feat_dim是最后一个维度
        combined_emb = combined_feat.mean(dim=1)  # 计算时间步长的平均值
        
        # 通过共享的嵌入层处理
        embed = self.shared_embed(combined_emb)
        output = self.outputlayer(embed)
        return output
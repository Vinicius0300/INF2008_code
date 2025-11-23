import torch
import torch.nn as nn

class SimpleHRNet(nn.Module):
    """
    HRNet simplificado CORRIGIDO para 2 keypoints
    Mantém 3 resoluções em paralelo: 64×64, 32×32, 16×16
    """
    
    def __init__(self, num_keypoints=2):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        
        # ============ STEM: Downsampling inicial ============
        # Input: (B, 3, 256, 256) → Output: (B, 64, 64, 64)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # ============ STAGE 2: 2 Branches (64×64, 32×32) ============
        
        # Branch 1 (alta resolução): 64×64
        self.stage2_branch1 = self._make_branch(64, 32, num_blocks=4)
        
        # Criar branch 2 (média resolução): 32×32
        self.stage2_transition = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage2_branch2 = self._make_branch(64, 64, num_blocks=4)
        
        # Fusion Stage 2: trocar informação entre branches
        # Branch2→Branch1: upsample (32×32 → 64×64)
        self.fuse2_b2_to_b1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        # Branch1→Branch2: downsample (64×64 → 32×32)
        self.fuse2_b1_to_b2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # ============ STAGE 3: 3 Branches (64×64, 32×32, 16×16) ============
        
        # Branch 1 (64×64)
        self.stage3_branch1 = self._make_branch(32, 32, num_blocks=4)
        
        # Branch 2 (32×32)
        self.stage3_branch2 = self._make_branch(64, 64, num_blocks=4)
        
        # Criar branch 3 (baixa resolução): 16×16
        self.stage3_transition = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage3_branch3 = self._make_branch(128, 128, num_blocks=4)
        
        # Fusion Stage 3: trocar informação entre 3 branches
        # Para Branch1 (64×64): recebe de branch2 e branch3
        self.fuse3_b2_to_b1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.fuse3_b3_to_b1 = nn.Sequential(
            nn.Conv2d(128, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        
        # ============ FINAL LAYER: Gerar heatmaps ============
        # Usa apenas branch1 (maior resolução) para gerar heatmaps
        self.final_layer = nn.Conv2d(32, num_keypoints, kernel_size=1)
        
        self._init_weights()
    
    def _make_branch(self, in_channels, out_channels, num_blocks):
        """
        Cria um branch com blocos residuais básicos
        """
        layers = []
        
        # Primeiro bloco: pode mudar número de canais
        layers.append(self._make_residual_block(in_channels, out_channels))
        
        # Blocos subsequentes: mantém canais
        for _ in range(num_blocks - 1):
            layers.append(self._make_residual_block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_residual_block(self, in_channels, out_channels):
        """
        Bloco residual básico com skip connection CORRETA
        """
        # Skip connection
        if in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            shortcut = nn.Identity()
        
        # Bloco principal
        main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Wrapper para aplicar residual
        class ResidualBlock(nn.Module):
            def __init__(self, main, skip):
                super().__init__()
                self.main = main
                self.skip = skip
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                return self.relu(self.main(x) + self.skip(x))
        
        return ResidualBlock(main_path, shortcut)
    
    def _init_weights(self):
        """Inicialização dos pesos"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass com dimensões CORRETAS
        
        Args:
            x: (B, 3, 256, 256)
        
        Returns:
            heatmaps: (B, num_keypoints, 64, 64)
        """
        # ============ STEM ============
        x = self.stem(x)  # (B, 64, 64, 64)
        
        # ============ STAGE 2 ============
        # Branch 1: alta resolução (64×64)
        x1 = self.stage2_branch1(x)  # (B, 32, 64, 64)
        
        # Branch 2: média resolução (32×32)
        x2 = self.stage2_transition(x)  # (B, 64, 32, 32)
        x2 = self.stage2_branch2(x2)    # (B, 64, 32, 32)
        
        # Fusion Stage 2
        # Branch1 recebe info de branch2 (upsampled)
        x1_fused = x1 + self.fuse2_b2_to_b1(x2)  # (B, 32, 64, 64)
        # Branch2 recebe info de branch1 (downsampled)
        x2_fused = x2 + self.fuse2_b1_to_b2(x1)  # (B, 64, 32, 32)
        
        # ============ STAGE 3 ============
        # Branch 1: (64×64)
        x1 = self.stage3_branch1(x1_fused)  # (B, 32, 64, 64)
        
        # Branch 2: (32×32)
        x2 = self.stage3_branch2(x2_fused)  # (B, 64, 32, 32)
        
        # Branch 3: baixa resolução (16×16)
        x3 = self.stage3_transition(x2_fused)  # (B, 128, 16, 16)
        x3 = self.stage3_branch3(x3)           # (B, 128, 16, 16)
        
        # Fusion Stage 3 (apenas para branch1, que gera os heatmaps)
        x1_final = x1 + self.fuse3_b2_to_b1(x2) + self.fuse3_b3_to_b1(x3)  # (B, 32, 64, 64)
        
        # ============ FINAL LAYER ============
        heatmaps = self.final_layer(x1_final)  # (B, num_keypoints, 64, 64)
        
        return heatmaps
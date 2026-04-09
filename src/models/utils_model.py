import torch.nn as nn

def init_weights(m):
    """
    Função de inicialização: Kaiming Normal para Conv e 
    ajuste constante para GroupNorm.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Kaiming é ideal para ReLU
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.GroupNorm):
        # Normalização geralmente começa com peso 1 e bias 0
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
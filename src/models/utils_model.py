import torch
import torch.nn as nn
import timm

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

def import_hrnet_weights(width = 32):
    # Define o modelo que você quer (hrnet_w32 ou hrnet_w48)
    model_name = f'hrnet_w{width}' 

    # Baixa o modelo com pesos da ImageNet
    temp_model = timm.create_model(model_name, pretrained=True)

    # Salva apenas o state_dict para usar no seu FullHRNet
    torch.save(temp_model.state_dict(), f'model_{model_name}_imagenet.pth')

    print(f"Pesos do {model_name} salvos com sucesso!")
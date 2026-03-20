# VFSS

Repositório para extração e manipulação de metadados de vídeos do INCA, incluindo rótulos e atribuições. Além disso, implementa treinamento e avaliação de modelos de visão computacional para segmentação e detecção de pontos em vídeos.

## Estrutura do Repositório

- `data_extraction/`: Scripts para extrair metadados, rótulos e arquivos de atribuição.

## Como Usar

1. Clone o repositório:

   ```bash
   git clone git@github.com:puc-rio-inca/vfss-data-split.git
   ```
2. Navegue até o diretório do projeto:

   ```bash
   cd vfss-data-split
   ```
3. Instale as dependências necessárias:

   ```bash
   pip install -r requirements.txt
   ```
4. Adicione o arquivo `patients_metadata.csv` na pasta `data/metadados/`. Leia a sessão sobre como gerar esse arquivo na seção "Gerando Metadados de Pacientes" abaixo.
5. Prepare os diretórios com os vídeos e rótulos de acordo com a estrutura presente no Google Drive do INCA. A estrutura esperado é:

   - Videos: `data/videos/`
     - É esperado que os vídeos estejam presentes em subdiretórios dentro dessa pasta. Exemplo:
       - `1.avi`
       - `2.avi`
       - `...`
   - Rótulos: `data/rotulos/`
     - É esperado o conteúdo da pasta `anotacoes-tecgraf/` presente no Google Drive do INCA. Exemplo:
       - `anotacoes-tecgraf/VC/1/`
       - `anotacoes-tecgraf/CS/1/`
       - `...`

### VFSSImageDataset:

A estrutura dele funciona da seguinte forma:

```python
VFSSImageDataset(
    video_frame_df: pd.DataFrame,
    output_dim: tuple= (512, 512),
    transform: A.Compose|None=None,
    offline_augmentation: bool=False,
    sigma: int=10)
```

* **video_frame_df**: Data frame no formato que o vfss_to_docker retorna para gente, ou então, ao realizar offline augmentation no formato que este retorna, mas nesse caso *offline_augmentation* deve ser True.
* **output_dim:** Dimensão de output da imagem. Deve condizer com possíveis resizes feitos na parte de transformação.
* **transform**:Transformação que serão aplicadas nas imagens de input e target. Caso haja fatores probabilisticos é importante dizer que o usuário estará fazendo uma online augmentation e isso deve ser usado somente no treinamento do modelo.
* **offline_augmentation**: Cuidado com este parâmetro. Caso seja False, a implementação acontece normalmente, então o dataframe usado será o que veio de vfss_to_docker e faz sentido que a transformação usada tenha fatores estocásticos. Todavia, se for True, o dataframe usado deve ser o que foi retornado da função que criou a base de dados aumentados de forma offline. Além disso, como já existe uma transformação (idealmente) com fatores estocásticos na parte de offline augmentation, então não devemos inserir tanta aleatoriedade na parte da transformação online na classe, a resize para garantir padronização e a passagem para tensor, cotinuam sendo importante.
* **sigma**: Valor para a variância do heatmap gerado através do uma distribuição Gaussiana

A classe considera que os pontos serão entregues, ou seja, os pontos precisam ser válidos, deve haver marcação dos pontos. A partir disso, serão gerados heatmaps para cada ponto e uma ROI que contém todos os pontos. É possível visualizar uma sample a partir do método *plot_sample(idx, display_keypoints, display_heatmaps, display_roi).*

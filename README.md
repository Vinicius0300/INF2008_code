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

6. Ajuste os parâmetros no notebook `notebooks/extrair-metadados/extrair-metadados-inca.ipynb` conforme necessário, como os diretórios dos vídeos e rótulos. Ou se não rode o script diretamente:
   ```bash
   python -m src.data_extraction.video_frame --videos_dir data/videos/ --labels_dir data/rotulos/anotacoes-tecgraf/ --target mask
   ```


## Gerando Metadados de Pacientes

O arquivo `patients_metadata.csv` é gerado através da extração de informações contudos nos _paths_ dos vídeos disponibilizados pelo INCA antes da reindexação. Para facilitar a manipulação dos vídeos eles foram reindexados utilizando DataManager e essa informação ficou armazenada no arquivo `video_dictionary.csv`. Para gerar o arquivo `patients_metadata.csv`, siga os passos abaixo:

1. Adicione o arquivo `video_dictionary.csv` na pasta `data/metadados/`.
2. Gere o arquivo `patients_metadata.csv` rodando:
````
    python -m src.data_extraction.patients
````



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


````



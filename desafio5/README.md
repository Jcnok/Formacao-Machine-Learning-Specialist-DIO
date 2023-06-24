# Projeto de criação de uma base de dados e treinamento da rede YOLOv4.


<img src="img/vision.jpg" referrerpolicy="same-origin" style="display: block; object-fit: cover; border-radius: 0px; width: 100%; height: 30vh; opacity: 1; object-position: center 50%;">

### Aprenda a criar sua própria base de dados, aplicar rótulos e treinar modelos com YOLOv4 usando transfer learning. Descubra o poder da inteligência artificial e revolucione a classificação de imagens. Exemplos práticos e resultados impressionantes. Transforme dados em conhecimento. Experimente!

**Seguindo os exemplos de aula, vamos rotular uma base de dados e aplicar o treinamento com a rede YOLO.**

Há duas maneiras principais para reunir as imagens
* Baixar de um dataset ou repositório como o Open Images Dataset, um dataset da Google que disponibiliza imagens para mais de 600 classes diferentes, ou usar imagens do COCO:https://cocodataset.org/#home.

* A segunda maneira consiste em baixar manualmente as imagens do objeto e catalogá-las utilizando o software LabelMe: http://labelme.csail.mit.edu/Release3.0/ para rotular as imagens. Esse é um processo manual e que pode ser bastante demorado, portanto para esse desafio vamos optar pela primeira opção.

* A maneira mais prática atualmente para baixar as imagens do Open Images Dataset é usar a ferramenta [OIDv4 Toolkit](https://github.com/EscVM/OIDv4_ToolKit).

## Requerimentos:

O nosso conjunto de imagens customizado deve conter:

* Imagens dos objetos que queremos reconhecer, já catalogadas (com os arquivos annotation)
* Arquivos obj.data e obj.names
* Arquivo .cfg customizado
* Arquivo train.txt (test.txt é opcional)

# Coletando e catalogando as imagens do objeto para treinamento

## Etapa 1 - Clonando o repositório da ferramenta


```python
!git clone https://github.com/EscVM/OIDv4_ToolKit.git
```

    Cloning into 'OIDv4_ToolKit'...
    remote: Enumerating objects: 422, done.[K
    remote: Total 422 (delta 0), reused 0 (delta 0), pack-reused 422[K
    Receiving objects: 100% (422/422), 34.08 MiB | 11.93 MiB/s, done.
    Resolving deltas: 100% (146/146), done.
    

## Etapa 2 - Acessando o diretório da ferramenta


```python
ls
```

    [0m[01;34mOIDv4_ToolKit[0m/  [01;34msample_data[0m/
    


```python
cd OIDv4_ToolKit/
```

    /content/OIDv4_ToolKit
    


```python
ls
```

    classes.txt  [0m[01;34mimages[0m/  LICENSE  main.py  [01;34mmodules[0m/  README.md  requirements.txt
    

## Etapa 3 - Instalando todas as bibliotecas necessárias


```python
!pip3 install -r requirements.txt
```

## Etapa 4 - Fazendo o download das imagens

### Conjunto de imagens de Treinamento

- https://storage.googleapis.com/openimages/web/index.html

exemplos de imagens: Penguin , Falcon, Duck, Weapon, Sunglasses, Woman

*  **Vamos utilizar a ferramenta TollKit para realizar o download de 3 objetos com 500 imagens de cada classe para realizarmos o treinamento.**


```python
!python main.py downloader --classes Apple Coffee_cup Horse --type_csv train --limit 500 --multiclasses 1
```

    [92m
    		   ___   _____  ______            _    _    
    		 .'   `.|_   _||_   _ `.         | |  | |   
    		/  .-.  \ | |    | | `. \ _   __ | |__| |_  
    		| |   | | | |    | |  | |[ \ [  ]|____   _| 
    		\  `-'  /_| |_  _| |_.' / \ \/ /     _| |_  
    		 `.___.'|_____||______.'   \__/     |_____|
    	[0m
    [92m
                 _____                    _                 _             
                (____ \                  | |               | |            
                 _   \ \ ___  _ _ _ ____ | | ___   ____  _ | | ____  ____ 
                | |   | / _ \| | | |  _ \| |/ _ \ / _  |/ || |/ _  )/ ___)
                | |__/ / |_| | | | | | | | | |_| ( ( | ( (_| ( (/ /| |    
                |_____/ \___/ \____|_| |_|_|\___/ \_||_|\____|\____)_|    
                                                              

    

* Observe que foi criado uma estrutura de pastas dentro de OIDv4_Tookit/OID/Dataset/Train/
* Em train existe uma pasta com as imagens e uma pasta Label com as coordenadas correspondente a classe de cada imagem.

* Agora vamos utilizar a ferramenta para baixar 100 imagens de cada classe para o realizarmos nossa validação.

### Conjunto de imagens de Validação


```python
!python main.py downloader --classes Apple Coffee_cup Horse --type_csv test --limit 100 --multiclasses 1
```

    [92m
    		   ___   _____  ______            _    _    
    		 .'   `.|_   _||_   _ `.         | |  | |   
    		/  .-.  \ | |    | | `. \ _   __ | |__| |_  
    		| |   | | | |    | |  | |[ \ [  ]|____   _| 
    		\  `-'  /_| |_  _| |_.' / \ \/ /     _| |_  
    		 `.___.'|_____||______.'   \__/     |_____|
    	[0m
    [92m
                 _____                    _                 _             
                (____ \                  | |               | |            
                 _   \ \ ___  _ _ _ ____ | | ___   ____  _ | | ____  ____ 
                | |   | / _ \| | | |  _ \| |/ _ \ / _  |/ || |/ _  )/ ___)
                | |__/ / |_| | | | | | | | | |_| ( ( | ( (_| ( (/ /| |    
                |_____/ \___/ \____|_| |_|_|\___/ \_||_|\____|\____)_|    
                                                              

    

* Observe que foi criado uma estrutura de pastas dentro de OIDv4_Tookit/OID/Dataset/test/
* Em test existe uma pasta com as imagens e uma pasta Label com as coordenadas correspondente a classe de cada imagem.

## Etapa 5 - Convertendo os arquivos de anotação para o formato aceito pelo YOLO

### 1. Colocar as classes no arquivo classes.txt
 * observe que o arquivo classes.txt estão com outras informações, precisamos alterar para as classes correspondente ao nosso dataset.


```python
!cat classes.txt
```

    Apple
    Orange
    Light switch
    


```python
# Alterando o arquivo classes.txt com as classes corretas.
!echo -e 'Apple\nCoffee cup\nHorse' > classes.txt
```


```python
# Conferindo as alterações.
!cat classes.txt
```

    Apple
    Coffee cup
    Horse
    

### 2. Baixar o arquivo converter_annotations.py e adicione-o ao diretório.
 * O Arquivo é um script pronto, necessário para conversão das Labels para o formato aceito pelo YOLO.


```python
# Realizando a conexão como o GoogleDrive.
from google.colab import drive
drive.mount('/content/gdrive')
```

    Mounted at /content/gdrive
    


```python
# Descompactando o arquivo para conversão.
#!unzip /content/gdrive/My\ Drive/YOLO/recursos/TreinamentoYOLO.zip -d /content/
```

* **Obs: Caso alguém queira replicar esse projeto, o aquivo converte_annotations.py está na pasta [scr](src/)**


```python
ls
```

    classes.txt  LICENSE  [0m[01;34mmodules[0m/  README.md
    [01;34mimages[0m/      main.py  [01;34mOID[0m/      requirements.txt
    


```python
# Copiando o arquivo converter_annotations para Pasta OIDv4_Toolkit
!cp /content/TreinamentoYOLO/converter_annotations.py ./
```

### 3. Executar o arquivo de conversão




```python
!python converter_annotations.py
```

    Subdiretorio atual: test
    Convertendo os annotations para a classe:  Apple_Coffee cup_Horse
    100% 300/300 [00:07<00:00, 41.10it/s]
    Subdiretorio atual: train
    Convertendo os annotations para a classe:  Apple_Coffee cup_Horse
    100% 1500/1500 [00:54<00:00, 27.41it/s]
    

## Etapa 6 - Compactar o dataset


```python
cd OID/Dataset/train/
```

    /content/OIDv4_ToolKit/OID/Dataset/train
    


```python
# Listando o nome da pata atual
ls
```

    [0m[01;34m'Apple_Coffee cup_Horse'[0m/
    


```python
import os
# Renomeando a pasta /Apple_Coffee cup_Horse para /obj
caminho_antigo = '/content/OIDv4_ToolKit/OID/Dataset/train/Apple_Coffee cup_Horse'
novo_nome = '/content/OIDv4_ToolKit/OID/Dataset/train/obj'

# Renomear a pasta
os.rename(caminho_antigo, novo_nome)
```


```python
#Conferindo se a pasta foi renomeada.
ls
```

    [0m[01;34mobj[0m/
    


```python
#Compactando a pasta para dentro da pasta OIDv4_Toolkit
!zip -r ../../../obj.zip obj -x obj/Label/*
```

      adding: obj/ (stored 0%)
      adding: obj/880de1e96516d4da.txt (deflated 49%)
      adding: obj/2e5fbb227d7b945c.jpg (deflated 0%)
      adding: obj/00aa0f033ff1df28.jpg (deflated 1%)
      adding: obj/1f250b4532ae9438.jpg (deflated 0%)

    


```python
ls
```

    [0m[01;34mobj[0m/
    


```python
# Voltando para pasta OIDv4_Toolkit.
cd ../../
```


```python
# Conferindo se estou na pasta correta!
ls
```

    classes.txt               [0m[01;34mimages[0m/  main.py   obj.zip  README.md
    converter_annotations.py  LICENSE  [01;34mmodules[0m/  [01;34mOID[0m/     requirements.txt
    


```python
#Copiando o arquivo obj.zip para o google Drive
!cp ./obj.zip /content/gdrive/MyDrive/YOLO/recursos
```

* **Agora vamos fazer o mesmo para o conjunto de validação.**


```python
# Acessando a pasta test
cd OID/Dataset/test/
```

    /content/OIDv4_ToolKit/OID/Dataset/test
    


```python
import os
# Renomeando a pasta de validação
caminho_antigo = '/content/OIDv4_ToolKit/OID/Dataset/test/Apple_Coffee cup_Horse'
novo_nome = '/content/OIDv4_ToolKit/OID/Dataset/test/valid'

# Renomear a pasta
os.rename(caminho_antigo, novo_nome)
```


```python
!zip -r ../../../valid.zip valid -x valid/Label/*
```

      adding: valid/ (stored 0%)
      adding: valid/5c41ee4384cd8ad7.txt (deflated 26%)
      adding: valid/13e366b80a184868.txt (deflated 10%)
      adding: valid/62d100326bad961a.txt (deflated 59%)
      adding: valid/43fd9963c68debf6.txt (deflated 17%)
      adding: valid/7f5a30978d018d2f.jpg (deflated 0%)
      adding: valid/b6024eea086419a9.jpg (deflated 0%)
      adding: valid/7776c836ffd15742.txt (deflated 12%)
      adding: valid/bbe1dfecd7159da2.txt (deflated 23%)
      adding: valid/5a139be6c8da63ed.txt (deflated 43%)
      adding: valid/a01e7b24591ec374.jpg (deflated 0%)

    


```python
# Acessando a pasta OIDv4_Toolkit
cd OIDv4_ToolKit/
```

    /content/OIDv4_ToolKit
    


```python
#Conferindo se o arquivo valid.zip foi compactado.
ls
```

    classes.txt               LICENSE   obj.zip    requirements.txt
    converter_annotations.py  main.py   [0m[01;34mOID[0m/       valid.zip
    [01;34mimages[0m/                   [01;34mmodules[0m/  README.md
    


```python
#Copiando o dataset de validação para o google drive
!cp ./valid.zip /content/gdrive/MyDrive/YOLO/recursos
```

# Editando os arquivos de configuração necessários para o treinamento


```python
# verificando o diretório atual
pwd
```




    '/content'




```python
# Clonando o repositório darknet
!git clone https://github.com/AlexeyAB/darknet
```

    Cloning into 'darknet'...
    remote: Enumerating objects: 15530, done.[K
    remote: Counting objects: 100% (16/16), done.[K
    remote: Compressing objects: 100% (12/12), done.[K
    remote: Total 15530 (delta 5), reused 11 (delta 4), pack-reused 15514[K
    Receiving objects: 100% (15530/15530), 14.22 MiB | 7.53 MiB/s, done.
    Resolving deltas: 100% (10417/10417), done.
    


```python
# Acessando a pasta darknet
cd darknet/
```

    /content/darknet
    


```python
# Realizando a build
!make
```
   

## Etapa 7 - Definindo os arquivos de configuração

### Modificações necessárias no arquivo de configuração .cfg


```python
# Copiando o arquivo de configuração para o google drive.
!cp cfg/yolov4.cfg /content/gdrive/My\ Drive/YOLO/recursos/yolov4_custom.cfg
```

### Nesta etapa após copiar o arquivo yolov4.cfg para o drive, é necessário abrir o arquivo e alterar todas as informações conforme abaixo:

* Alterar o arquivo yolov4_custom.cfg no driver
* subdivisions=64 # consumo de memória
* width=608  # somente se estiver com outro resultado.
* height=608 # somente se estiver com outro resultado.
* max_batches = 6000 #pois recomenda-se 2000 épocas para cada classe com são 3 classes * 2000 = 60000
* steps = 4800, 5400 # recomenda-se 80% dos 6000 = 4800 e 90% dos 6000 = 5400
* filters= 24 # alterar todos os filters para 24
* classes=3   # alterar todas as classes para 3

### Modificações no obj.names e obj.data


```python
# Criando esses arquivos vazios.
!touch obj.names
!touch obj.data
```


```python
# Copiando os aruivos para o google drive.
!cp obj.names /content/gdrive/MyDrive/YOLO/recursos/obj.names
!cp obj.data /content/gdrive/MyDrive/YOLO/recursos/obj.data
```

* Necessário alterar os arquivos **obj.names e obj.data no googledrive**.
* obj.names alterar: # na verdade basta incluir, pois o arquivo esta vazio.
  * Apple
  * Coffee cup
  * Horse
* obj.data alterar:
  * classes = 3 #numero de classes a ser treinada.
  * train = data/train.txt # Caminho do arquivo, mais tarde será copiado para esse local
  * valid = data/test.txt  # Caminho do arquivo, mais tarde será copiado para esse local
  * names = data/obj.names # Caminho do arquivo, mais tarde será copiado para esse local
  * backup = /yolo/recursos # Caminho onde vamos salvar os pesos dos treinamentos a cada 100 épocas.

## Etapa 8 - Gerando o arquivo train.txt e test.txt


```python
pwd
```




    '/content/darknet'




```python
cd ..
```

    /content
    


```python
cd OIDv4_ToolKit/
```

    /content/OIDv4_ToolKit
    


```python
#descompactando o arquivo na pasta data
!unzip obj.zip -d ./data
```


```python
# Descompactando valid na pasta data
!unzip valid.zip -d ./data
```


```python
pwd
```




    '/content/OIDv4_ToolKit'



### Os scripts gerar_train.py e gerar_test.py está na pasta [./scr](src/).


```python
# gera o arquivo train.txt padrão yolo
!python /content/TreinamentoYOLO/gerar_train.py
```


```python
# gera o arquivo test.txt padrão yolo
!python /content/TreinamentoYOLO/gerar_test.py
```


```python
#Necessário estar na pasta data.
cd data
```

    /content/OIDv4_ToolKit/data
    


```python
# copiando o arquivo train.txt gerado para o drive
!cp train.txt /content/gdrive/MyDrive/YOLO/recursos/train.txt
```


```python
# copiando o arquivo test.txt gerado para o drive
!cp test.txt /content/gdrive/MyDrive/YOLO/recursos/test.txt
```

* **Agora temos todos os arquivos necessários para realizar o treinamento com nosso dataset.**

# Realizando o treinamento com o dataset personalizado.

* Importante nesse momento, desconectar notebook do colab, para que possamos iniciar o treinamento sem problemas de memória etc..
* Lembrando que salvamos no google drive:
  * As imagens que queremos reconhecer, catalogadas;
  * Os arquivos Obj.data e Obj.names;
  * O arquivo .cfg com as devidas alterações;
  * Os arquivos train.txt e test.txt;

* Vamos verificar se estamos usando os recursos da GPU do Colab.


```python
import tensorflow as tf
device_name = tf.test.gpu_device_name()
print(device_name)
```

    /device:GPU:0
    

## Realizando a conexão com o google drive:


```python
from google.colab import drive
drive.mount('/content/gdrive')
```

    Mounted at /content/gdrive
    


```python
# consultando a pasta que contém os recursos para o desafio.
!ls /content/gdrive/MyDrive/Yolo4/
```

    modelo_YOLOv4.zip  recursos
    

* Criando um atalho para a pasta Yolo4 com o nome yolo


```python
!ln -s /content/gdrive/MyDrive/Yolo4 /yolo
```


```python
# conferindo se o atalho está funcionando
!ls /yolo/
```

    modelo_YOLOv4.zip  recursos
    

## Download do Darknet


```python
!git clone https://github.com/AlexeyAB/darknet
```

    Cloning into 'darknet'...
    remote: Enumerating objects: 15530, done.[K
    remote: Counting objects: 100% (16/16), done.[K
    remote: Compressing objects: 100% (11/11), done.[K
    remote: Total 15530 (delta 5), reused 13 (delta 5), pack-reused 15514[K
    Receiving objects: 100% (15530/15530), 14.21 MiB | 22.53 MiB/s, done.
    Resolving deltas: 100% (10417/10417), done.
    


```python
# conferindo a pasta
%cd darknet
!ls
```

    /content/darknet
    3rdparty		darknet_video.py       net_cam_v4.sh
    build			data		       README.md
    build.ps1		image_yolov3.sh        results
    cfg			image_yolov4.sh        scripts
    cmake			include		       src
    CMakeLists.txt		json_mjpeg_streams.sh  vcpkg.json
    DarknetConfig.cmake.in	LICENSE		       vcpkg.json.opencv23
    darknet_images.py	Makefile	       video_yolov3.sh
    darknet.py		net_cam_v3.sh	       video_yolov4.sh
    

## Compilando a biblioteca

Para compilar com suporte a GPU/CUDA será modificar o arquivo Makefile para definir algumas variáveis como ativado (de 0 para 1) -> OPENCV, GPU e CUDNN. Ao invés de abrir o arquivo em um editor de texto iremos fazer essa alteração com o comando `sed`.


```python
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
```


```python
# compilando
!make
```

## Preparando o YOLO

### Baixando os pesos do modelo pré-treinado


```python
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

## Testando o Yolo


```python
# testando o modelo com uma imagem exemplo
!./darknet detect cfg/yolov4.cfg yolov4.weights data/person.jpg
```

     CUDA-version: 11080 (12000), cuDNN: 8.9.0, GPU count: 1  
     OpenCV version: 4.2.0
     0 : compute_capability = 750, cudnn_half = 0, GPU: Tesla T4 
    net.optimized_memory = 0 
    mini_batch = 1, batch = 8, time_steps = 1, train = 0 
       layer   filters  size/strd(dil)      input                output
       0 Create CUDA-stream - 0 
     Create cudnn-handle 0 
    conv     32       3 x 3/ 1    608 x 608 x   3 ->  608 x 608 
     161 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
    nms_kind: greedynms (1), beta = 0.600000 
    Total BFLOPS 128.459 
    avg_outputs = 1068395 
     Allocate additional workspace_size = 134.22 MB 
    Loading weights from yolov4.weights...
     seen 64, trained: 32032 K-images (500 Kilo-batches_64) 
    Done! Loaded 162 layers from weights-file 
     Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/person.jpg: Predicted in 2783.190000 milli-seconds.
    dog: 99%
    person: 100%
    horse: 98%
    Unable to init server: Could not connect: Connection refused
    
    (predictions:10651): Gtk-[1;33mWARNING[0m **: [34m01:25:47.866[0m: cannot open display: 
    

### Criando uma função para exibir o resultado.


```python
import cv2
import matplotlib.pyplot as plt

def mostrar(caminho):
  img = cv2.imread(caminho)
  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.show()
```

### Exibindo o resultado.


```python
mostrar("predictions.jpg")
```


    
![png](img/output_108_0.png)
    


* Agora que já conferimos e está tudo funcionando corretamente, podemos iniciar o treinamento do nosso dataset.

## Enviando nosso dataset customizado para Colab

* Como podemos conferir, dentro da pasta darknet a pasta data, e é pra lá que iremos descompactar nosso dataset obj.zip.


```python
#conferindo se nosso atalho yolo está correto
!ls /yolo/recursos
```

    obj.names  obj.zip  test.txt  train.txt  valid.zip  yolov4_custom.cfg
    


```python
# Conferindo se estamos dentro da pasta darknet
!pwd
```

    /content/darknet
    


```python
# Descompactando obj.zip para pasta data
!unzip /yolo/recursos/obj.zip -d ./data/
```

* **Observe que foram descompactados todos os 1500 de imagens e 1500 arquivos txt com as respectivas coordenadas das imagens.**


```python
# Conferindo se a quantidade de arquivos estão corretas.
!ls data/obj/ -1 | wc -l
```

    3001
    

## Enviando nosso dataset de validação para o colab.


```python
!unzip /yolo/recursos/valid.zip -d ./data/
```


```python
# Conferindo se a quantidade de arquivos estão corretas.
!ls data/valid/ -1 | wc -l
```

    601
    

## Enviando os arquivos de configuração para o treinamento.

* Agora vamos enviar os arquivos do drive para as respectivas pastas no colab:
  * yolo4_custom.cfg para a pasta /cfg
  * obj.names para a pasta /data
  * obj.data para a pasta /data
  * train.txt para a pasta /data
  * test.txt para a pasta /data




```python
!ls /yolo/recursos/
```

    obj.data  obj.names  obj.zip  test.txt	train.txt  valid.zip  yolov4_custom.cfg
    


```python
# Copiando os arquivos do drive para as respectivas pastas no colab.
!cp /yolo/recursos/yolov4_custom.cfg ./cfg
!cp /yolo/recursos/obj.names ./data
!cp /yolo/recursos/obj.data ./data
!cp /yolo/recursos/train.txt ./data
!cp /yolo/recursos/test.txt ./data
```

## Baixando os pesos pré-treinados das camadas convolucionais(Transfer Learning).

**O arquivo "yolov4.conv.137" contém os pesos pré-treinados de uma versão do YOLOv4 que passou por um treinamento extenso em um conjunto de dados grande e diversificado, como o COCO (Common Objects in Context). Esses pesos pré-treinados são usados como ponto de partida para o treinamento adicional em um conjunto de dados específico.**

**Ao utilizar o "yolov4.conv.137" como ponto de partida, o treinamento subsequente é acelerado, pois a rede neural já possui conhecimentos iniciais sobre a detecção de objetos gerais. Isso ajuda a melhorar a precisão e a eficiência do modelo, especialmente quando se tem um conjunto de dados menor disponível para treinamento.**


```python
# Baixando os pesos para transfer learning
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

    --2023-06-23 02:04:48--  https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
    Resolving github.com (github.com)... 140.82.121.3
    Connecting to github.com (github.com)|140.82.121.3|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://objects.githubusercontent.com/github-production-release-asset-2e65be/75388965/48bfe500-889d-11ea-819e-c4d182fcf0db?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230623%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230623T020449Z&X-Amz-Expires=300&X-Amz-Signature=392194de6ca29f7b452779254bf4b9b6486889fd7d6e257acbfbd2fb0089d74d&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=75388965&response-content-disposition=attachment%3B%20filename%3Dyolov4.conv.137&response-content-type=application%2Foctet-stream [following]
    --2023-06-23 02:04:49--  https://objects.githubusercontent.com/github-production-release-asset-2e65be/75388965/48bfe500-889d-11ea-819e-c4d182fcf0db?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230623%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230623T020449Z&X-Amz-Expires=300&X-Amz-Signature=392194de6ca29f7b452779254bf4b9b6486889fd7d6e257acbfbd2fb0089d74d&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=75388965&response-content-disposition=attachment%3B%20filename%3Dyolov4.conv.137&response-content-type=application%2Foctet-stream
    Resolving objects.githubusercontent.com (objects.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to objects.githubusercontent.com (objects.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 170038676 (162M) [application/octet-stream]
    Saving to: ‘yolov4.conv.137’
    
    yolov4.conv.137     100%[===================>] 162.16M  48.8MB/s    in 3.4s    
    
    2023-06-23 02:04:52 (47.4 MB/s) - ‘yolov4.conv.137’ saved [170038676/170038676]
    
    


```python
# conferindo se o arquivo yolo4.conv.137 foi baixado
!ls
```

    3rdparty		darknet_video.py       predictions.jpg
    backup			data		       README.md
    build			image_yolov3.sh        results
    build.ps1		image_yolov4.sh        scripts
    cfg			include		       src
    cmake			json_mjpeg_streams.sh  vcpkg.json
    CMakeLists.txt		LICENSE		       vcpkg.json.opencv23
    darknet			Makefile	       video_yolov3.sh
    DarknetConfig.cmake.in	net_cam_v3.sh	       video_yolov4.sh
    darknet_images.py	net_cam_v4.sh	       yolov4.conv.137
    darknet.py		obj		       yolov4.weights
    

## Agora sim enfim, podemos realizar o treinamento do modelo com nosso dataset customizado usando a técnica de Transfer Learning com YOLOv4.


```python
!./darknet detector train data/obj.data cfg/yolov4_custom.cfg yolov4.conv.137 -dont_show -map
```

    [1;30;43mA saída de streaming foi truncada nas últimas 5000 linhas.[0m
     total_bbox = 261642, rewritten_bbox = 0.350479 % 
    v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) Region 139 Avg (IOU: 0.000000), count: 1, class_loss = 0.029831, iou_loss = 0.000000, total_loss = 0.029831 
    v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) 
    
     (next mAP calculation at 1000 iterations) ]2;343/6000: loss=4.6 hours left=18.1
     343: 4.551411, 5.910802 avg loss, 0.000018 rate, 9.015303 seconds, 21952 images, 18.145398 hours left
    Loaded: 0.000042 seconds
    v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) 
    

### Vamos analisar o que cada comando está realizando para um melhor entendimento.

* **"./darknet"**: indica o caminho para o executável do Darknet, que é o framework utilizado para treinar modelos YOLOv4.
* **"detector train"**: especifica que estamos iniciando o treinamento do detector de objetos.
* **"data/obj.data"**: especifica o caminho para o arquivo "obj.data", que contém as configurações do dataset utilizado para treinamento, incluindo o caminho para os arquivos de treinamento e teste, classes de objetos, etc.
* **"cfg/yolov4_custom.cfg"**: indica o caminho para o arquivo de configuração do modelo YOLOv4 personalizado, chamado "yolov4_custom.cfg". Esse arquivo contém informações sobre a arquitetura da rede neural, como número de camadas, filtros, tamanho de imagens de entrada, entre outros.
* **"yolov4.conv.137"**: especifica o caminho para o arquivo de pesos pré-treinados do YOLOv4, que servirá como ponto de partida para o treinamento do modelo personalizado. Esses pesos pré-treinados são fundamentais para inicializar os pesos da rede neural de forma adequada.
* **"-dont_show"**: indica para não exibir as visualizações durante o treinamento. Isso é útil quando se está treinando em um ambiente sem interface gráfica.
* **"-map"**: calcula a média do mAP (mean Average Precision) após cada época de treinamento. O mAP é uma métrica que avalia a precisão do modelo na detecção de objetos.

* **Observe que esse processo pode ser bem demorado, pois para cada classe é recomendado pelo menos 2000 épocas para que o modelo performe bem.**
* **Podemos parar e recomeçar o treinamento a partir do nosso backup dos pesos através do arquivo yolo/recursos/yolov4_custom_last.weghts, pois esses pesos são atualizados a cada 100 épocas.**
* **o arquivo chart.png contém um gráfico com a performace dos erros vs épocas.**


```python
# vizualizando a performace do treinamento pelo gráfico
mostrar('chart.png')
```


    
![png](img/output_133_0.png)
    


* **Observe que noso modelo ainda está bem ruim, e que foram treinados 300 épocas, nessas condições levariamos cerca de 19 horas para treinamos as 6000 épocas.**

## Como Continuar o treinamento a partir do backup.


```python
!./darknet detector train data/obj.data cfg/yolov4_custom.cfg /yolo/recursos/yolov4_custom_last.weights -dont_show -map
```

     CUDA-version: 11080 (12000), cuDNN: 8.9.0, GPU count: 1  
     OpenCV version: 4.2.0
     Prepare additional network for mAP calculation...
     0 : compute_capability = 750, cudnn_half = 0, GPU: Tesla T4 
    net.optimized_memory = 0 
    mini_batch = 1, batch = 64, time_steps = 1, train = 0 
       layer   filters  size/strd(dil)      input                output
       0 Create CUDA-stream - 0 
     Create cudnn-handle 0 
    conv     32       3 x 3/ 1    608 x 608 x   3 ->  608 x 608 
    
     (next mAP calculation at 1000 iterations) ]2;303/6000: loss=5.5 hours left=38.7
     303: 5.451722, 7.834808 avg loss, 0.000011 rate, 20.456419 seconds, 19392 images, 38.692538 hours left
    Loaded: 0.000065 seconds
    v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) Region 139 Avg (IOU: 0.000000), count: 1, class_loss = 0.004652, iou_loss = 0.000000, total_loss = 0.004652 
    
    

* **Observe o modelo começou o treinamento a partir das 300 épocas já treinadas.**

## Verificando Mean Average Precision do modelo


```python
!./darknet detector map data/obj.data cfg/yolov4_custom.cfg /yolo/recursos/yolov4_custom_last.weights
```


```python
'''
class_id = 0, name = Apple, ap = 2.62%   	       (TP = 0, FP = 6)
class_id = 1, name = Coffee cup, ap = 2.54%   	 (TP = 0, FP = 0)
class_id = 2, name = Horse, ap = 4.78%   	       (TP = 0, FP = 0)
'''
```

### Veja que ele mostra algumas informações importantes:
* Uma matriz de confusão com o percentual de acertos.
* Podemos ver que o nosso modelo ainda está muito ruim com uma taxa de menos de 3 % para Apple e Coffe cup e menos de 5 % para horse.
* Dessa forma podemos concluir que será necessário muitas épocas de treinamento para o modelo performar bem.


## Resultado de um treinamento com 4000 épocas.
* **Deixei esse arquivo em [/data](data/) para replicação se desejado**.


```python
!./darknet detector map /yolo/treinamento/obj.data /yolo/treinamento/yolov4_custom.cfg /yolo/treinamento/yolov4_custom_4000_best.weights
```

     CUDA-version: 11080 (12000), cuDNN: 8.9.0, GPU count: 1  
     OpenCV version: 4.2.0
     0 : compute_capability = 750, cudnn_half = 0, GPU: Tesla T4 
    net.optimized_memory = 0 
    mini_batch = 1, batch = 64, time_steps = 1, train = 0 
       layer   filters  size/strd(dil)      input                output
       0 Create CUDA-stream - 0 
     Create cudnn-handle 0 
    conv     32       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 
     160 conv     24       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x  24 0.008 BF
     161 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
    nms_kind: greedynms (1), beta = 0.600000 
    Total BFLOPS 59.578 
    avg_outputs = 490041 
     Allocate additional workspace_size = 134.22 MB 
    Loading weights from /yolo/treinamento/yolov4_custom_4000_best.weights...
     seen 64, trained: 260 K-images (4 Kilo-batches_64) 
    Done! Loaded 162 layers from weights-file 
    
     calculation mAP (mean average precision)...
     Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    300
     detections_count = 3814, unique_truth_count = 509  
    class_id = 0, name = Apple, ap = 38.73%   	 (TP = 155, FP = 388) 
    class_id = 1, name = Coffee cup, ap = 85.68%   	 (TP = 105, FP = 23) 
    class_id = 2, name = Horse, ap = 52.91%   	 (TP = 106, FP = 94) 
    
     for conf_thresh = 0.25, precision = 0.42, recall = 0.72, F1-score = 0.53 
     for conf_thresh = 0.25, TP = 366, FP = 505, FN = 143, average IoU = 31.60 % 
    
     IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
     mean average precision (mAP@0.50) = 0.591086, or 59.11 % 
    Total Detection Time: 10 Seconds
    
    Set -points flag:
     `-points 101` for MS COCO 
     `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data) 
     `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom dataset
    


```python
'''
class_id = 0, name = Apple, ap = 38.73%   	 (TP = 155, FP = 388)
class_id = 1, name = Coffee cup, ap = 85.68%   	 (TP = 105, FP = 23)
class_id = 2, name = Horse, ap = 52.91%   	 (TP = 106, FP = 94)
'''
```

* **Observem que após as 4000 épocas o modelo já começa a performar melhor porém, confirmando que o ideal seria no mínimo as 6000 épocas.**

## Testando o modelo treinado com uma imagem de exemplo (maçã).


```python
!./darknet detector test /yolo/treinamento/obj.data /yolo/treinamento/yolov4_custom.cfg /yolo/treinamento/yolov4_custom_4000_best.weights /yolo/imagens/apple.jpg
```

     CUDA-version: 11080 (12000), cuDNN: 8.9.0, GPU count: 1  
     OpenCV version: 4.2.0
     0 : compute_capability = 750, cudnn_half = 0, GPU: Tesla T4 
    net.optimized_memory = 0 
    mini_batch = 1, batch = 64, time_steps = 1, train = 0 
       layer   filters  size/strd(dil)      input                output
       0 Create CUDA-stream - 0 
     Create cudnn-handle 0 
    conv     32       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  32 0.299 BF
       1 conv     64       3 x 3/ 2    416 x 416 x  32 ->  208 
     160 conv     24       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x  24 0.008 BF
     161 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
    nms_kind: greedynms (1), beta = 0.600000 
    Total BFLOPS 59.578 
    avg_outputs = 490041 
     Allocate additional workspace_size = 134.22 MB 
    Loading weights from /yolo/treinamento/yolov4_custom_4000_best.weights...
     seen 64, trained: 260 K-images (4 Kilo-batches_64) 
    Done! Loaded 162 layers from weights-file 
     Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    /yolo/imagens/apple.jpg: Predicted in 743.770000 milli-seconds.
    Apple: 100%
    Unable to init server: Could not connect: Connection refused
    
    (predictions:119498): Gtk-[1;33mWARNING[0m **: [34m04:07:34.662[0m: cannot open display: 
    


```python
mostrar('predictions.jpg')
```


    
![png](img/output_148_0.png)
    


* **Observem que o modelo treinado com 400 mil épocas conseguiu detectar a maçã com 100% de precisão.**

## Conclusão:

**Em conclusão, o uso da rede YOLOv4 e da técnica de transfer learning para treinamento de modelos de detecção de objetos apresentou resultados fascinantes. Ao prepararmos nosso dataset e realizarmos o treinamento, percebemos a importância de um número adequado de épocas para obter bons resultados. Embora tenhamos observado melhorias significativas a partir de 4000 épocas, recomendamos uma média de 2000 épocas por classe, totalizando 6000 épocas para nosso teste.**

**É importante ressaltar que, embora a performance desse tipo de treinamento demande recursos de hardware mais avançados, como GPUs, podemos minimizar os custos com o conhecimento e as ferramentas corretas. No entanto, vale ressaltar que os benefícios proporcionados pela técnica são extremamente valiosos e podem transformar diversos setores, incluindo a indústria.**

**Com base em minha experiência de 10 anos trabalhando na indústria no Japão, consigo visualizar inúmeras aplicações práticas para essa técnica. Na área de montagem de autopeças, por exemplo, onde já existia um rigoroso processo de verificação realizado de forma humana e mecânica, a aplicação do YOLOv4 poderia auxiliar na detecção de falhas por meio de uma simples imagem. Imagine como seria incrível ter um sistema capaz de identificar e sinalizar o momento exato de um problema, otimizando o processo de qualidade e garantindo maior eficiência na produção.**

**Essa é apenas uma das inúmeras possibilidades oferecidas pela detecção de objetos com YOLOv4. Em áreas como segurança, medicina, logística e muitas outras, essa técnica pode revolucionar processos, aumentar a produtividade e melhorar a precisão dos resultados.**

**Portanto, convido você a explorar o potencial da detecção de objetos com YOLOv4 em seu próprio setor. Com o conhecimento adequado e a utilização das ferramentas corretas, é possível alavancar a eficiência, a precisão e a automação em diversas aplicações. Não deixe essa oportunidade passar! Seja parte dessa transformação e impulsione sua área de atuação para um novo patamar de excelência. O futuro está aqui, esperando por você.**



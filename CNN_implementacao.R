#1. Instalação e Carregamento de Pacotes

#torch: Implementação do PyTorch em R.
#torchvision: Fornece datasets (como CIFAR10) e transformações de imagem.
#luz: Facilita o treinamento de modelos torch (similar ao Keras para TensorFlow).
#ggplot2: Para visualização de resultados.
#magick: Para carregar imagens.

install.packages("torch")
install.packages("torchvision")
install.packages("luz")
install.packages("ggplot2")
install.packages("magick")

library(torch)
library(torchvision)
library(luz)
library(ggplot2)
library(magick)

#2. Carregamento e Preparação dos Dados
#Baixa, normaliza e divide em batches.

#transform: Define um pipeline de pré-processamento:
 #Converte imagens em tensores (transform_to_tensor()).
 #Normaliza os valores dos pixels usando médias e desvios padrão do CIFAR10.
transform <- function(img) {
  img %>%
    transform_to_tensor() %>%          # Converte a imagem para tensor
    transform_normalize(               # Normaliza os valores dos pixels
      mean = c(0.4914, 0.4822, 0.4465),  # Médias padrão do CIFAR10 (R, G, B)
      std = c(0.2470, 0.2435, 0.2616)    # Desvios padrão do CIFAR10
    )
}
#cifar10_dataset(): Carrega o dataset CIFAR10:
 #train = TRUE → Dados de treino (50k imagens).
 #train = FALSE → Dados de teste (10k imagens).
train_ds <- cifar10_dataset(
  root = "./data",       # Diretório onde os dados serão baixados
  train = TRUE,          # Carrega o conjunto de treino (50k imagens)
  download = TRUE,       # Baixa o dataset se não existir
  transform = transform  # Aplica a transformação definida
)

test_ds <- cifar10_dataset(
  root = "./data",
  train = FALSE,         # Carrega o conjunto de teste (10k imagens)
  download = TRUE,
  transform = transform
)
#dataloader(): Divide os dados em batches (lotes) para treinamento eficiente:
 #batch_size = 128 → 128 imagens por lote.
 #shuffle = TRUE → Embaralha os dados de treino para evitar viés.
train_dl <- dataloader(train_ds, batch_size = 128, shuffle = TRUE)  
test_dl <- dataloader(test_ds, batch_size = 128)                   


#3. Definição da Arquitetura CNN
#Quatro camadas convolucionais + pooling + dropout + fully connected + MLP.

net <- nn_module(  # Define uma CNN personalizada
  "CIFAR10_CNN",
  initialize = function() {
    # Camadas convolucionais (nn_conv2d):
     # Extraem features das imagens (32, 64, 128 filtros).
     # kernel_size = 3: Filtros 3x3.
     # padding = 1: Mantém as dimensões espaciais.
    self$conv1 <- nn_conv2d(3, 32, kernel_size = 3, padding = 1)  # 3 canais (RGB) → 32 filtros
    self$conv2 <- nn_conv2d(32, 64, kernel_size = 3, padding = 1) # 32 → 64 filtros
    self$conv3 <- nn_conv2d(64, 128, kernel_size = 3, padding = 1) # 64 → 128 filtros
    self$conv4 <- nn_conv2d(128, 128, kernel_size = 3, padding = 1) # 128 → 128 filtros
    
    # Camadas Pooling:
    self$pool <- nn_max_pool2d(2)       # Reduz dimensão espacial pela metade (tamanho da imagem pela metade)
    self$dropout <- nn_dropout(p = 0.5) # Desativa aleatoriamente 50% dos neurônios para evitar overfitting
    
    #Camadas fully connected (totalmente conectadas), lineares (nn_linear):
    self$fc1 <- NULL                # Definida dinamicamente para ajustar à saída das convoluções (será definida no forward())
    self$fc2 <- nn_linear(512, 10)  # 512 neurônios → 10 classes (saída)
    
    # Camada auxiliar para cálculo de dimensões
    self$dim_calculator <- nn_sequential(
      nn_conv2d(3, 32, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_conv2d(32, 64, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(2),
      nn_conv2d(64, 128, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_conv2d(128, 128, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(2),
      nn_flatten()  # Achata os tensores para a camada linear
    )
  },
   
  forward = function(x) { # Define o fluxo de dados
    # Calcula dimensões na primeira execução
    if (is.null(self$fc1)) {
      test_output <- self$dim_calculator(x)
      input_size <- dim(test_output)[2]
      self$fc1 <- nn_linear(input_size, 512)$to(device = x$device)
    }
    
    x %>% 
      # Bloco 1: Conv → ReLU → Conv → ReLU → Pool
      self$conv1() %>% nnf_relu() %>%
      self$conv2() %>% nnf_relu() %>%
      self$pool() %>%
      
      # Bloco 2: Conv → ReLU → Conv → ReLU → Pool → Dropout 
      self$conv3() %>% nnf_relu() %>%
      self$conv4() %>% nnf_relu() %>%
      self$pool() %>%
      self$dropout() %>%
      
      # Achata os tensores para a camada linear
      torch_flatten(start_dim = 2) %>%
      
      # Classificador (MLP)
      self$fc1() %>% nnf_relu() %>%
      self$dropout() %>%
      self$fc2()  # Saída final (10 classes)
  }
)


#4. Treinamento do Modelo
#Usando Adam e early stopping.

fitted <- net %>%
  setup(                                   # Configura o modelo
    loss = nn_cross_entropy_loss(),        # Função de perda, entropia cruzada (classificação)
    optimizer = optim_adam,                # Otimizador Adam (ajusta os pesos)
    metrics = list(luz_metric_accuracy())  # Métrica será acurácia
  ) %>%
  set_hparams() %>%
  fit(                              # Treina o modelo
    train_dl,                       # Dados de treino
    valid_data = test_dl,           # Dados de validação
    epochs = 20,                    # Número de épocas: passa pelos dados 20 vezes
    callbacks = list(
      luz_callback_early_stopping(patience = 3),         # Interrompe se o modelo não melhorar em 3 épocas
      luz_callback_model_checkpoint(path = "./models/")  # Salva os melhores modelos em ./models/
    )
  )


#5. Avaliação dos Resultados
#Mede acurácia no teste e visualiza resultados.

plot(fitted)  # Gráfico de loss e acurácia
evaluation <- fitted %>% evaluate(data = test_dl)  # Avalia no teste

# Mostra todas as métricas disponíveis
print(evaluation)

# Exibe a acurácia
if (!is.null(evaluation$records$metrics$valid[[1]]$acc)) {
  test_acc <- evaluation$records$metrics$valid[[1]]$acc
  print(paste("Acurácia no teste:", round(test_acc * 100, 2), "%"))
} else {
  print("Não foi possível encontrar a métrica de acurácia")
}


#6. Carregamento e Pré-processamento de Imagens 

# Função para pré-processamento
preprocess_image <- function(image_path) {
  # Carrega a imagem
  img <- image_read(image_path)
  
  # Redimensiona para 32x32 (tamanho do CIFAR10)
  img <- image_resize(img, "32x32!")
  
  # Converte para array numérico (0-255) e depois normaliza (0-1)
  img_array <- as.integer(img[[1]]) / 255
  
  # Reorganiza as dimensões para (C, H, W) - canais primeiro
  img_array <- aperm(img_array, c(3, 1, 2))
  
  # Converte para tensor torch
  img_tensor <- torch_tensor(img_array, dtype = torch_float32())
  
  # Aplica normalização (usando os mesmos parâmetros do treino)
  transform_normalize(
    img_tensor,
    mean = c(0.4914, 0.4822, 0.4465),
    std = c(0.2470, 0.2435, 0.2616)
  )
}


#7. Carregamento do Modelo Treinado

model <- fitted$model


#8. Carregamento e Predição da Imagem

predict_image <- function(image_path) {
  # Classes do CIFAR10
  cifar10_classes <- c("", "avião", "automóvel", "pássaro", "gato",
                       "veado", "cachorro", "sapo", "cavalo", "navio", "caminhão")
  
  # Pré-processa a imagem
  img_tensor <- preprocess_image(image_path)
  
  # Adiciona dimensão de batch (1, 3, 32, 32)
  img_tensor <- img_tensor$unsqueeze(1)
  
  # Faz predição
  model$eval()
  with_no_grad({
    output <- model(img_tensor)
    probs <- nnf_softmax(output, dim = 2)
    pred <- torch_argmax(probs, dim = 2)
  })
  
  # Retorna resultados
  list(
    class = cifar10_classes[as.integer(pred) + 1],
    probability = as.numeric(torch_max(probs)$item())
  )
}


#9. Visualização de Imagens com Predição

# Predição com uma imagem

# Substitua pelo caminho da sua imagem
resultado <- predict_image("G:\\My Drive\\MESTRADO UFRPE (RURAL)\\Disciplinas\\Computação para Análise de Dados (CPAD)\\Projeto 1VA\\imagens\\CIFAR10\\predicao_caminhao_CIFAR10.jpg")
cat(sprintf("Predição: %s (%.2f%% de confiança)\n",
            resultado$class, resultado$probability * 100))

# Visualizar a imagem
image <- image_read("G:\\My Drive\\MESTRADO UFRPE (RURAL)\\Disciplinas\\Computação para Análise de Dados (CPAD)\\Projeto 1VA\\imagens\\CIFAR10\\predicao_caminhao_CIFAR10.jpg")
print(image)


#EXTRA. Processamento e Predição de Múltiplas Imagens

# Define o diretório onde estão as imagens
diretorio <- "G:/My Drive/MESTRADO UFRPE (RURAL)/Disciplinas/Computação para Análise de Dados (CPAD)/Projeto 1VA/imagens/CIFAR10"

# Lista todos os arquivos .jpg ou .png no diretório
arquivos_imagens <- list.files(
  path = diretorio,
  pattern = "\\.(jpg|png|jpeg)$",  # Padrão para extensões de imagem
  full.names = TRUE                # Retorna caminhos completos
)

# Verifica se encontrou arquivos
if (length(arquivos_imagens) == 0) {
  stop("Nenhuma imagem encontrada no diretório especificado.")
}

# Função que aplica predict_image a cada imagem e retorna um dataframe com resultados
predizer_imagens <- function(lista_imagens) {
  resultados <- lapply(lista_imagens, function(caminho) {
    pred <- predict_image(caminho)
    data.frame(
      Imagem = basename(caminho),       # Nome do arquivo (sem caminho completo)
      Classe_Prevista = pred$class,
      Confianca = sprintf("%.2f%%", pred$probability * 100),
      stringsAsFactors = FALSE
    )
  })
  # Combina todos os resultados em um único dataframe
  do.call(rbind, resultados)
}

# Aplica a função
resultados_finais <- predizer_imagens(arquivos_imagens)

# Mostra os resultados
print(resultados_finais)





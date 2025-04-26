# Instalar pacotes se necessário
install.packages("torch")
install.packages("torchvision")
install.packages("luz")
install.packages("ggplot2")
install.packages("imager")

library(torch)
library(torchvision)
library(luz)
library(ggplot2)
library(imager)

# Configurar semente para reprodutibilidade
torch_manual_seed(42)

# 1. Carregamento e preparação o dataset MNIST
transform <- function(x) {
  x %>% 
    transform_to_tensor() %>% 
    transform_normalize(mean = 0.1307, std = 0.3081)
}

train_ds <- mnist_dataset(
  root = "./data", 
  train = TRUE, 
  download = TRUE,
  transform = transform
)

test_ds <- mnist_dataset(
  root = "./data", 
  train = FALSE, 
  download = TRUE,
  transform = transform
)

train_dl <- dataloader(train_ds, batch_size = 128, shuffle = TRUE)
test_dl <- dataloader(test_ds, batch_size = 128)

# 2. Definição da arquitetura da CNN 
net <- nn_module(
  "MNIST_CNN",
  initialize = function() {
    # Primeiro bloco convolucional
    self$conv1 <- nn_conv2d(1, 32, kernel_size = 3, padding = "same")
    self$conv2 <- nn_conv2d(32, 64, kernel_size = 3, padding = "same")
    
    # Segundo bloco convolucional
    self$conv3 <- nn_conv2d(64, 128, kernel_size = 3, padding = "same")
    
    self$dropout1 <- nn_dropout2d(0.25)
    self$dropout2 <- nn_dropout2d(0.5)
    self$maxpool <- nn_max_pool2d(2)
    
    # Camadas fully connected
    self$fc1 <- nn_linear(128 * 7 * 7, 128)  # 7x7 é o tamanho após dois poolings
    self$fc2 <- nn_linear(128, 10)
  },
  
  forward = function(x) {
    # Primeiro bloco
    x <- x %>%
      self$conv1() %>%
      nnf_relu() %>%
      self$conv2() %>%
      nnf_relu() %>%
      self$maxpool() %>%
      self$dropout1()
    
    # Segundo bloco
    x <- x %>%
      self$conv3() %>%
      nnf_relu() %>%
      self$maxpool() %>%
      self$dropout1()
    
    # Achatar para fully connected
    x <- torch_flatten(x, start_dim = 2)
    
    # Camadas fully connected
    x <- x %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$dropout2() %>%
      self$fc2() %>%
      nnf_log_softmax(dim = 1)
    
    return(x)
  }
)

# 3. Treinamento do modelo
model <- net %>%
  setup(
    loss = nn_nll_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_accuracy())
  )

fitted <- model %>% 
  fit(
    train_dl,
    epochs = 2,
    valid_data = test_dl,
    verbose = TRUE
  )

# 4. Visualização resultados
plot(fitted) + 
  theme_minimal() +
  ggtitle("Desempenho do Modelo")

# Avalia o conjunto de teste
evaluation <- fitted %>% evaluate(data = test_dl)

# Mostra todas as métricas disponíveis
print(evaluation)

# Verifica e mostra a acurácia do teste
if (!is.null(evaluation$records$metrics$valid[[1]]$acc)) {
  test_acc <- evaluation$records$metrics$valid[[1]]$acc
  print(paste("Acurácia no teste:", round(test_acc * 100, 2), "%"))
} else {
  print("Não foi possível encontrar a métrica de acurácia")
}


#Fluxo Completo de Processamento:

#Entrada: Imagens 28x28 pixels (1 canal - escala de cinza)
#Conv1: Aplica 32 filtros 3x3 → saída 28x28x32
#Conv2: Aplica 64 filtros 3x3 → saída 28x28x64
#MaxPool: Reduz para 14x14x64
#Conv3: Aplica 128 filtros 3x3 → saída 14x14x128
#MaxPool: Reduz para 7x7x128
#Flatten: Transforma em vetor de 7*7*128 = 6272 elementos
#FC1: Camada densa com 128 neurônios
#FC2: Camada de saída com 10 neurônios (uma para cada dígito)

#Cada camada convolucional é seguida por ReLU para introduzir não-linearidade, 
#e dropout para prevenir overfitting.


#5. Carregamento e Pré-processamento de Imagens 

preprocess_image <- function(image_path) {
  # Carrega a imagem
  img <- imager::load.image(image_path)
  
  # Converte para escala de cinza se for colorida
  if(imager::spectrum(img) == 3) {
    img <- imager::grayscale(img)
  }
  
  # Redimensiona para 28x28 (tamanho do MNIST)
  img <- imager::resize(img, 28, 28)
  
  # Inverte cores (MNIST tem fundo preto)
  img <- 1 - img
  
  # Converte para array e remove dimensões extras
  img_array <- as.array(img)[,,1,1]
  
  # Converte para tensor e formata corretamente
  img_tensor <- torch::torch_tensor(img_array, dtype = torch::torch_float32())$
    unsqueeze(1)$  # Adiciona dimensão do canal
    unsqueeze(1)   # Adiciona dimensão do batch
  
  # Normalização (mesmos parâmetros do MNIST)
  img_tensor <- (img_tensor - 0.1307) / 0.3081
  
  return(img_tensor)
}


#6. Carregamento e Predição da Imagem

# Carregar a imagem
image_path <- "G:\\My Drive\\MESTRADO UFRPE (RURAL)\\Disciplinas\\Computação para Análise de Dados (CPAD)\\Projeto 1VA\\imagens\\predicao_num1_MNIST.png"

# Pré-processamento
img_tensor <- preprocess_image(image_path)

# Verificar dimensões (deve ser [1, 1, 28, 28])
print(dim(img_tensor))

# Fazer a predição
prediction <- fitted %>% predict(img_tensor)

# Obter a classe prevista
predicted_class <- torch::torch_argmax(prediction, dim = 2)$item()

# Obter probabilidades (convertendo de log-softmax)
probabilities <- torch::torch_exp(prediction)$squeeze() %>% as.numeric()

# Mostrar resultados
cat("Dígito previsto:", predicted_class, "\n")
cat("Probabilidades para cada classe (0-9):\n")
print(round(probabilities, 4))


#7. Visualização da Imagem com Predição

# Mostrar a imagem com a predição
par(mfrow = c(1, 1), mar = c(0, 0, 2, 0))
plot(imager::load.image(image_path), axes = FALSE)
title(main = paste("Dígito previsto:", predicted_class), cex.main = 2)



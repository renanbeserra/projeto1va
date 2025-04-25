#1. Instalação e Carregamento de Pacotes

#torch: Implementação do PyTorch em R.
#torchvision: Fornece datasets (como CIFAR10) e transformações de imagem.
#luz: Facilita o treinamento de modelos torch (similar ao Keras para TensorFlow).
#ggplot2: Para visualização de resultados.

install.packages("torch")
install.packages("torchvision")
install.packages("luz")
install.packages("ggplot2")

library(torch)
library(torchvision)
library(luz)
library(ggplot2)


#2. Carregamento e Preparação dos Dados
#Baixa, normaliza e divide em batches.

#transform: Define um pipeline de pré-processamento:
 #Converte imagens em tensores (transform_to_tensor()).
 #Normaliza os valores dos pixels usando médias e desvios padrão do CIFAR10.

#cifar10_dataset(): Carrega o dataset CIFAR10:
 #train = TRUE → Dados de treino (50k imagens).
 #train = FALSE → Dados de teste (10k imagens).

#dataloader(): Divide os dados em batches (lotes) para treinamento eficiente:
 #batch_size = 128 → 128 imagens por lote.
 #shuffle = TRUE → Embaralha os dados de treino para evitar viés.

transform <- function(img) {
  img %>%
    transform_to_tensor() %>%          # Converte a imagem para tensor
    transform_normalize(               # Normaliza os valores dos pixels
      mean = c(0.4914, 0.4822, 0.4465),  # Médias padrão do CIFAR10 (R, G, B)
      std = c(0.2470, 0.2435, 0.2616)    # Desvios padrão do CIFAR10
    )
}

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

train_dl <- dataloader(train_ds, batch_size = 128, shuffle = TRUE)  # Cria batches de treino
test_dl <- dataloader(test_ds, batch_size = 128)                    # Cria batches de teste


#3. Definição da Arquitetura CNN
#4 camadas convolucionais + pooling + dropout + MLP.

#nn_module(): Define uma CNN personalizada.

#Camadas convolucionais (nn_conv2d):
 #Extraem features das imagens (32, 64, 128 filtros).
 #kernel_size = 3: Filtros 3x3.
 #padding = 1: Mantém as dimensões espaciais.

#Pooling (nn_max_pool2d): Reduz a dimensionalidade (tamanho da imagem pela metade).

#Dropout (nn_dropout): Desativa aleatoriamente 50% dos neurônios para evitar overfitting.

#Camadas lineares (nn_linear):
 #fc1: Definida dinamicamente para ajustar à saída das convoluções.
 #fc2: Produz logits para 10 classes.

#forward(): Define o fluxo de dados:
 #Passa pelas convoluções + ReLU.
 #Aplica pooling e dropout.
 #Achata os tensores para a camada linear.
 #Classificação final.

net <- nn_module(
  "CIFAR10_CNN",
  initialize = function() {
    # Camadas convolucionais
    self$conv1 <- nn_conv2d(3, 32, kernel_size = 3, padding = 1)  # 3 canais (RGB) → 32 filtros
    self$conv2 <- nn_conv2d(32, 64, kernel_size = 3, padding = 1) # 32 → 64 filtros
    self$conv3 <- nn_conv2d(64, 128, kernel_size = 3, padding = 1) # 64 → 128 filtros
    self$conv4 <- nn_conv2d(128, 128, kernel_size = 3, padding = 1) # 128 → 128 filtros
    
    # Camadas de pooling e dropout
    self$pool <- nn_max_pool2d(2)       # Reduz dimensão espacial pela metade
    self$dropout <- nn_dropout(p = 0.5)  # Regularização (evita overfitting)
    
    # Camadas fully connected (ajustadas dinamicamente)
    self$fc1 <- NULL                    # Será definida no forward()
    self$fc2 <- nn_linear(512, 10)      # 512 neurônios → 10 classes (saída)
    
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
  
  forward = function(x) {
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
      
      # Achata para a camada linear
      torch_flatten(start_dim = 2) %>%
      
      # Classificador (MLP)
      self$fc1() %>% nnf_relu() %>%
      self$dropout() %>%
      self$fc2()  # Saída final (10 classes)
  }
)


#4. Treinamento do Modelo
#Usando Adam e early stopping.

#setup(): Configura o modelo:
 #Loss: Entropia cruzada (para classificação).
 #Otimizador: Adam (ajusta os pesos).
 #Métricas: Acurácia.

#fit(): Treina o modelo:
 #epochs = 5: Passa pelos dados 5 vezes.
 #Callbacks:
  #early_stopping: Interrompe se o modelo não melhorar em 3 épocas.
  #model_checkpoint: Salva os melhores modelos em ./models/.

fitted <- net %>%
  setup(
    loss = nn_cross_entropy_loss(),  # Função de perda (classificação)
    optimizer = optim_adam,          # Otimizador Adam
    metrics = list(luz_metric_accuracy())  # Acompanha a acurácia
  ) %>%
  set_hparams() %>%
  fit(
    train_dl,                       # Dados de treino
    valid_data = test_dl,           # Dados de validação
    epochs = 5,                     # Número de épocas
    callbacks = list(
      luz_callback_early_stopping(patience = 3),  # Para se não melhorar em 3 épocas
      luz_callback_model_checkpoint(path = "./models/")  # Salva os melhores modelos
    )
  )


#5. Avaliação dos Resultados
#Mede acurácia no teste e visualiza resultados.

plot(fitted)  # Gráfico de loss e acurácia
evaluation <- fitted %>% evaluate(data = test_dl)  # Avalia no teste

# Exibe a acurácia
if (!is.null(evaluation$records$metrics$valid[[1]]$acc)) {
  test_acc <- evaluation$records$metrics$valid[[1]]$acc
  print(paste("Acurácia no teste:", round(test_acc * 100, 2), "%"))
} else {
  print("Não foi possível encontrar a métrica de acurácia")
}

# Visualiza a arquitetura
model <- net()
print(model)


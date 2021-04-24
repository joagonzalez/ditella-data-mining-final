rm(list = ls())

library(data.table)
library(caret)
library(dplyr)
library(pROC)

setwd('C:/Users/a310005/Desktop/DiTella/Data Mining/data')
getwd()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 1er bloque

load_csv_data <- function(csv_file, sample_ratio = 1, drop_cols = NULL,
                          sel_cols = NULL) {
    
    dt <- fread(csv_file, header = TRUE, sep = ",", stringsAsFactors = TRUE,
                na.strings = "", drop = drop_cols, select = sel_cols,
                showProgress = TRUE)
    
    if (sample_ratio < 1) {
        sample_size <- as.integer(sample_ratio * nrow(dt))
        dt <- dt[sample(.N, sample_size)]
    }
    
    return(dt)
}

# Defino que variables voy a cargar
TO_KEEP <- c("platform", "age", "install_date", "id",
             "TutorialStart", "Label_max_played_dsi",
             "StartSession_sum_dsi3")

# Cargo uno de los datasets de entrenamiento del tp
train_set <- load_csv_data("train_3.csv",
                           sample_ratio = 0.3
                           
                           ,
                           sel_cols = TO_KEEP)  # Para dejar todas las variables "sel_cols = NULL"

train_set[, train_sample := TRUE]  # Idéntico a train_set$train_sample <- TRUE

# Cargo el dataset de evaluación del TP
eval_set <- load_csv_data("evaluation.csv",
                          sel_cols = setdiff(TO_KEEP, "Label_max_played_dsi"))  # Para dejar todas las variables "sel_cols = NULL"

eval_set[, train_sample := FALSE]

# Uno los datasets
data_set <- rbind(train_set, eval_set, fill = TRUE)
rm(train_set, eval_set)
gc()

# Modifico las variables que considere necesario
data_set[, Label := factor(ifelse(Label_max_played_dsi == 3,
                                  "churn", "no_churn"))]  # Idéntico a "data_set$Label <- factor(ifelse(data_set$Label_max_played_dsi == 3, "churn", "no_churn"))"
data_set[, Label_max_played_dsi := NULL]  # Elimina la variable Label_max_played_dsi. Idéntico a "data_set$Label_max_played_dsi <- NULL"

table(train_sample = data_set$train_sample, churn = data_set$Label)

# Aquí sería un buen lugar para crear nuevas variables

# Separo en conjunto de training y evaluación de nuevo
train_set <- data_set[(train_sample),]  # Idéntico a "train_set <- data_set[data_set$train_sample,]"
train_set[, train_sample := NULL]
eval_set <- data_set[(!train_sample),]
eval_set[, train_sample := NULL]
rm(data_set)
gc()

# Vemos cómo usar caret

# Primero se debe definir el esquema que se 
# utilizará para validar el modelo
# k-fold cross validation de 3
fitControl <- trainControl(method = "cv",
                           number = 3,
                           verboseIter = TRUE)

# Luego se entrena y valida el modelo de acuerdo al esquema
# definido
rpart_fit_grid <- train(x = train_set %>% select(-Label),
                        y = train_set$Label,
                        method = "rpart",
                        trControl = fitControl,
                        na.action = "na.omit")

print(rpart_fit_grid)

# Probemos holdout set con AUC como métrica de evaluación
val_index <- list("holdout_1" = c(1:10000))

train_index <- list("holdout_1" = setdiff(c(1:nrow(train_set)),
                                          val_index[["holdout_1"]]))

fitControl <- trainControl(method = "cv",  ## OJO: El método da igual
                           index = train_index,
                           indexOut = val_index,
                           verboseIter = TRUE,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

# Entrenamos un árbol
rpart_fit_grid <- train(x = train_set %>% select(-Label),
                        y = train_set$Label,
                        method = "rpart",
                        trControl = fitControl,
                        na.action = "na.omit",
                        metric = "ROC")

print(rpart_fit_grid)

# Entrenamos knn con la misma metodología de evaluación
# (age tiene NA y knn no los acepta)

knn_fit_grid <- train(Label ~ ., # Con esta sintaxis hace one-hot-econding
                      data = train_set %>% select(-age),
                      method = "knn",
                      trControl = fitControl,
                      na.action = "na.omit",
                      metric = "ROC")

print(knn_fit_grid)

knn_fit_grid_scaled <- train(Label ~ .,
                             data = train_set %>% select(-age),
                             method = "knn",
                             trControl = fitControl,
                             na.action = "na.omit",
                             preProcess = c("center", "scale"),
                             metric = "ROC")

print(knn_fit_grid_scaled)

knn_fit_grid_tuned <- train(Label ~ .,
                            data = train_set %>% select(-age),
                            method = "knn",
                            trControl = fitControl,
                            na.action = "na.omit",
                            preProcess = c("center", "scale"),
                            tuneGrid = expand.grid(k = c(50, 55, 60)),
                            metric = "ROC")

print(knn_fit_grid_tuned)

# Predecimos sobre los datos de evaluación
eval_preds <- data.frame(id = eval_set$id,
                         Label = predict(knn_fit_grid_tuned,
                                         eval_set, type = "prob")[,"churn"])

mean(eval_preds$Label)

# Armo el archivo para subir a Kaggle
write.table(eval_preds, "modelo_super_basico.csv",
            sep = ",", row.names = FALSE, quote = FALSE)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 2do bloque

data_set <- read.table("bankruptcy_data.txt", sep = "\t", header = TRUE, stringsAsFactors = TRUE)
prop.table(table(data_set$class))

fit_on_bkr <- list("holdout" = sample(c(1:nrow(data_set)),
                                      round(0.2 * nrow(data_set))))

val_on_bkr <- list("holdout" = setdiff(c(1:nrow(data_set)),
                                       fit_on_bkr[["holdout"]]))

fitControl <- trainControl(method = "cv",
                           index = fit_on_bkr, indexOut = val_on_bkr,
                           verboseIter = TRUE,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

data_set$class <- factor(ifelse(data_set$class,
                                "bankruptcy", "no_bankruptcy"))

# Modelo con las variables original
knn_fit <- train(x = data_set %>% select(-class),
                 y = data_set$class,
                 method = "knn",
                 trControl = fitControl,
                 metric = "ROC")

knn_fit$results[which.max(knn_fit$results$ROC),]

# Modelo con las variables en z-scores
knn_fit_scaled <- train(x = data_set %>% select(-class),
                        y = data_set$class,
                        method = "knn",
                        trControl = fitControl,
                        preProcess = c("center","scale"),
                        metric = "ROC")

knn_fit_scaled$results[which.max(knn_fit_scaled$results$ROC),]

# Modelo con las variables en logaritmos
data_set_log <- data_set
for (v in setdiff(names(data_set_log), "class")) {
    data_set_log[[v]] <- log(data_set_log[[v]] + 1 - min(data_set_log[[v]]))
}

knn_fit_log <- train(x = data_set_log %>% select(-class),
                     y = data_set_log$class,
                     method = "knn",
                     trControl = fitControl,
                     metric = "ROC")

knn_fit_log$results[which.max(knn_fit_log$results$ROC),]

knn_fit_log_scaled <- train(x = data_set_log %>% select(-class),
                            y = data_set_log$class,
                            method = "knn",
                            trControl = fitControl,
                            preProcess = c("center","scale"),
                            metric = "ROC")

knn_fit_log_scaled$results[which.max(knn_fit_log_scaled$results$ROC),]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 3er bloque

nb_fit <- train(x = data_set %>% select(-class),
                y = data_set$class,
                method = "nb",
                trControl = fitControl,
                tuneGrid = data.frame(fL = 1, usekernel = FALSE, adjust = 0),
                metric = "ROC")

nb_fit$results[which.max(nb_fit$results$ROC),]

discretize <- function(input_data, bins = 10, type = "equal") {
    # Función que dado un vector input lo discretiza
    if (type == "equal") {
        cut_points <- seq(from = min(input_data), to = max(input_data),
                          length.out = bins+1)
    } else if (type == "freq") {
        cut_points <- unique(quantile(input_data,
                                      prob = seq(from = 0, to = 1,
                                                 length.out = bins+1)))
    } else {
        return(NULL)
    }
    cut_points[1] <- -Inf
    cut_points[length(cut_points)] <- Inf
    return(cut(input_data, breaks = cut_points))
}

data_set_disc_eq <- data_set
for (v in setdiff(names(data_set_log), "class")) {
    data_set_disc_eq[[v]] <- discretize(data_set_disc_eq[[v]],
                                        bins = 15, "equal")
}

nb_fit_disc_equal <- train(x = data_set_disc_eq %>% select(-class),
                           y = data_set_disc_eq$class,
                           method = "nb",
                           trControl = fitControl,
                           tuneGrid = expand.grid(fL = 1, usekernel = FALSE, adjust = 0),
                           metric = "ROC")

nb_fit_disc_equal$results[which.max(nb_fit_disc_equal$results$ROC),]

data_set_disc_fq <- data_set
for (v in setdiff(names(data_set_log), "class")) {
    data_set_disc_fq[[v]] <- discretize(data_set_disc_fq[[v]],
                                        bins = 15, "freq")
}

nb_fit_disc_freq <- train(x = data_set_disc_fq %>% select(-class),
                          y = data_set_disc_fq$class,
                          method = "nb",
                          trControl = fitControl,
                          tuneGrid = expand.grid(fL = 1, usekernel = FALSE,
                                                 adjust = 0),
                          metric = "ROC")

nb_fit_disc_freq$results[which.max(nb_fit_disc_freq$results$ROC),]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 4to bloque

# Simulamos un dataset totalmente aleatorio
y <- factor(ifelse(sample(rep_len(c(0, 1), 2500)), "y", "n"))
X <- matrix(rnorm(2500 * 10000), ncol = 10000)

# Seleccionamos atributos en base a t-tests
var_imp <- apply(X, 2, function(x) {t.test(x[y == "y"], x[y != "y"])$p.value})
top_vars <- order(var_imp)[c(1:150)]

# Ordenamos para que sea un data.frame entrenable y separamos en training y validation
X <- data.frame(X[,top_vars])
names(X) <- paste("Col", c(1:ncol(X)), sep = "")
fit_on <- list("holdout" = sample(c(1:nrow(X)), round(0.5 * nrow(X))))
val_on <- list("holdout" = setdiff(c(1:nrow(X)), fit_on[["holdout"]]))

# Definimos la estructura del esquema de validación y entrenamos
fitControl <- trainControl(method = "cv",
                           index = fit_on, indexOut = val_on,
                           verboseIter = TRUE)

xgb_sim_fit <- train(x = data.frame(X), y = y,
                     method = "xgbTree", trControl = fitControl)
xgb_sim_fit$results[which.max(xgb_sim_fit$results$Accuracy),]  # Obtuvimos información del azar!

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# 5to bloque

data_bank <- read.table("bank-full.csv", sep = ";", header = TRUE, stringsAsFactors = TRUE)
prop.table(table(data_bank$y))

dmy <- dummyVars( ~ . - y, data = data_bank)
data_bank <- cbind(data.frame(predict(dmy, newdata = data_bank)),
                   data_bank[, "y", FALSE])

# Separamos en training y evaluation
train_index <- sample(c(1:nrow(data_bank)), 36200)
training_bank <- data_bank[train_index,]
evaluation_bank <- data_bank[-train_index,]

fitControl <- trainControl(method = "LGOCV",
                           number = 1,
                           p = 0.8,
                           verboseIter = TRUE,
                           classProbs = TRUE)

xgbFit <- train(x = training_bank %>% select(-y),
                y = training_bank$y,
                na.action = "na.omit",
                method = "xgbTree",
                trControl = fitControl)
xgbFit$results[which.max(xgbFit$results$Accuracy),]

preds_p <- predict(xgbFit, newdata = evaluation_bank, type = "prob")[,2]

# Distribución de las probabilidades predichas
ggplot(data.frame(p = preds_p, clase = evaluation_bank$y),
       aes(x = p, col = clase, fill = clase)) +
    geom_density(alpha = 0.5)

prec_recall <- function(conf_m, verbose = TRUE) {
    
    if (verbose) {
        print(conf_m)
    }
    
    prec <- conf_m["yes", "yes"] / sum((conf_m[, "yes"]))
    rec <- conf_m["yes", "yes"] / sum((conf_m["yes",]))
    f1 <- 2* prec * rec / (prec + rec)
    fpr <- conf_m["no", "yes"] / sum((conf_m["no",]))
    return(data.frame(acc= sum(diag(conf_m)) / sum(conf_m),
                      prec = prec, rec = rec,
                      f1 = f1, tpr = rec, fpr = fpr))
}

# Valor predefinido
prec_recall(table(actual = evaluation_bank$y,
                  pred = factor(ifelse(preds_p >= 0.5, "yes", "no"))))

# Valor que debería aumentar recall
prec_recall(table(actual = evaluation_bank$y,
                  pred = factor(ifelse(preds_p >= 0.3, "yes", "no"))))

# Valor que debería aumentar precision
prec_recall(table(actual = evaluation_bank$y,
                  pred = factor(ifelse(preds_p >= 0.7, "yes", "no"))))

# Grafico la curva ROC
data_roc <- data.frame()
for (p in seq(min(preds_p), max(preds_p), by = 0.01)) {
    data_roc <- rbind(data_roc,
                      prec_recall(table(actual = evaluation_bank$y,
                                        pred = factor(ifelse(preds_p > p,
                                                             "yes", "no"))),
                                  verbose = FALSE)[,c("fpr", "tpr")])
}
plot(data_roc, type = "l")

# Cálculo del AUC
auc(roc(ifelse(evaluation_bank$y == "yes", 1, 0), preds_p))

# Matriz de costos
cost_matrix <- matrix(c(0, 120, 1000, 120), ncol = 2, byrow = TRUE)

# Corte 0.5
m050 <- table(actual = evaluation_bank$y,
              pred = factor(ifelse(preds_p >= 0.5, "yes", "no")))
sum(m050 * cost_matrix) # Costo con un corte de 0.5

# Corte 0.12
m012 <- table(actual = evaluation_bank$y,
              pred = factor(ifelse(preds_p >= 0.12, "yes", "no")))
sum(m012 * cost_matrix) # Costo con un corte de 0.12

# Grafico la evolución del costo
data_cost <- data.frame()
for (p in seq(min(preds_p) + 0.001, max(preds_p) - 0.001, by = 0.01)) {
    mp <- table(actual = evaluation_bank$y,
                pred = factor(ifelse(preds_p >= p, "yes", "no")))
    data_cost <- rbind(data_cost, data.frame(p = p, cost = sum(mp * cost_matrix)))
}

plot(data_cost, type = "l")
data_cost[which.min(data_cost$cost),]

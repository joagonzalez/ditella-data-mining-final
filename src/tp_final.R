#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# TP Final DataMining MiM Analytics - Di Tella 2021
# Alumnos:
# Joaquin Gonzalez - joagonzalez@gmail.com
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Librerias
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

rm(list = ls())

library(data.table)
library(caret)
library(dplyr)
library(pROC)
library(Matrix)
library(xgboost)

setwd('C:/Users/a310005/Desktop/DiTella/Data Mining/data')
getwd()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Definicion de funciones 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

one_hot_sparse <- function(data_set) {
    
    require(Matrix)
    
    created <- FALSE
    
    if (sum(sapply(data_set, is.numeric)) > 0) {  # Si hay, Pasamos los numéricos a una matriz esparsa (sería raro que no estuviese, porque "Label"  es numérica y tiene que estar sí o sí)
        out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.numeric), with = FALSE]), "dgCMatrix")  # Si en lugar de pasar un objeto de data table, pasan un data.frame común no debe decir ", with = FALSE" en ninguna instrucción
        created <- TRUE
    }
    
    if (sum(sapply(data_set, is.logical)) > 0) {  # Si hay, pasamos los lógicos a esparsa y lo unimos con la matriz anterior
        if (created) {
            out_put_data <- cbind2(out_put_data,
                                   as(as.matrix(data_set[,sapply(data_set, is.logical),
                                                         with = FALSE]), "dgCMatrix"))
        } else {
            out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.logical), with = FALSE]), "dgCMatrix")
            created <- TRUE
        }
    }
    
    # Identificamos las columnas que son factor (OJO: el data.frame no debería tener character)
    fact_variables <- names(which(sapply(data_set, is.factor)))
    
    # Para cada columna factor hago one hot encoding
    i <- 0
    
    for (f_var in fact_variables) {
        
        f_col_names <- levels(data_set[[f_var]])
        f_col_names <- gsub(" ", ".", paste(f_var, f_col_names, sep = "_"))
        j_values <- as.numeric(data_set[[f_var]])  # Se pone como valor de j, el valor del nivel del factor
        
        if (sum(is.na(j_values)) > 0) {  # En categóricas, trato a NA como una categoría más
            j_values[is.na(j_values)] <- length(f_col_names) + 1
            f_col_names <- c(f_col_names, paste(f_var, "NA", sep = "_"))
        }
        
        if (i == 0) {
            fact_data <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                      x = rep(1, nrow(data_set)),
                                      dims = c(nrow(data_set), length(f_col_names)))
            fact_data@Dimnames[[2]] <- f_col_names
        } else {
            fact_data_tmp <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                          x = rep(1, nrow(data_set)),
                                          dims = c(nrow(data_set), length(f_col_names)))
            fact_data_tmp@Dimnames[[2]] <- f_col_names
            fact_data <- cbind(fact_data, fact_data_tmp)
        }
        
        i <- i + 1
    }
    
    if (length(fact_variables) > 0) {
        if (created) {
            out_put_data <- cbind(out_put_data, fact_data)
        } else {
            out_put_data <- fact_data
            created <- TRUE
        }
    }
    return(out_put_data)
}


random_grid <- function(size,
                        min_nrounds, max_nrounds,
                        min_max_depth, max_max_depth,
                        min_eta, max_eta,
                        min_gamma, max_gamma,
                        min_colsample_bytree, max_colsample_bytree,
                        min_min_child_weight, max_min_child_weight,
                        min_subsample, max_subsample) {
    
    rgrid <- data.frame(nrounds = if (min_nrounds == max_nrounds) {
        rep(min_nrounds, size)
    } else {
        sample(c(min_nrounds:max_nrounds),
               size = size, replace = TRUE)
    },
    max_depth = if (min_max_depth == max_max_depth) {
        rep(min_max_depth, size)
    } else {
        sample(c(min_max_depth:max_max_depth),
               size = size, replace = TRUE)
    },
    eta = if (min_eta == max_eta) {
        rep(min_eta, size)
    } else {
        round(runif(size, min_eta, max_eta), 7)
    },
    gamma = if (min_gamma == max_gamma) {
        rep(min_gamma, size)
    } else {
        round(runif(size, min_gamma, max_gamma), 7)
    },
    colsample_bytree = if (min_colsample_bytree == max_colsample_bytree) {
        rep(min_colsample_bytree, size)
    } else {
        round(runif(size, min_colsample_bytree, max_colsample_bytree), 7)
    },
    min_child_weight = if (min_min_child_weight == max_min_child_weight) {
        rep(min_min_child_weight, size)
    } else {
        round(runif(size, min_min_child_weight, max_min_child_weight), 7)
    },
    subsample = if (min_subsample == max_subsample) {
        rep(min_subsample, size)
    } else {
        round(runif(size, min_subsample, max_subsample), 7)
    })
    
    return(rgrid)
}


train_xgboost <- function(data_train, data_val, rgrid) {
    
    watchlist <- list(train = data_train, valid = data_val)
    
    predicted_models <- list()
    
    for (i in seq_len(nrow(rgrid))) {
        print(i)
        print(rgrid[i,])
        
        trained_model <- xgb.train(data = data_train,
                                   params=as.list(rgrid[i, c("max_depth",
                                                             "eta",
                                                             "gamma",
                                                             "colsample_bytree",
                                                             "subsample",
                                                             "min_child_weight")]),
                                   nrounds = rgrid[i, "nrounds"],
                                   watchlist = watchlist,
                                   objective = "binary:logistic",
                                   eval.metric = "auc",
                                   print_every_n = 10)
        
        perf_tr <- tail(trained_model$evaluation_log, 1)$train_auc
        perf_vd <- tail(trained_model$evaluation_log, 1)$valid_auc
        print(c(perf_tr, perf_vd))
        
        predicted_models[[i]] <- list(results = data.frame(rgrid[i,],
                                                           perf_tr = perf_tr,
                                                           perf_vd = perf_vd),
                                      model = trained_model)
        
        rm(trained_model)
        
        gc()
    }
    
    return(predicted_models)
}


result_table <- function(pred_models) {
    
    res_table <- data.frame()
    i <- 1
    
    for (m in pred_models) {
        res_table <- rbind(res_table, data.frame(i = i, m$results))
        i <- i + 1
    }
    
    res_table <- res_table[order(-res_table$perf_vd),]
    
    return(res_table)
}
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Separeacion de dataset 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Variables a utilizar para predicciones
TO_KEEP_TEST <- c("platform", "age", "install_date", "id",
             "TutorialStart", "Label_max_played_dsi",
             "StartSession_sum_dsi3")

TO_KEEP <- c("platform", "age", "install_date", "id",
             "TutorialStart", "Label_max_played_dsi",
             "StartSession_sum_dsi0", "StartSession_sum_dsi1",
             "StartSession_sum_dsi2", "StartSession_sum_dsi3",
             "categorical_1", "categorical_2", "categorical_3",
             "categorical_4", "categorical_5", "categorical_6",
             "categorical_7", "device_model")

# Carga de datasets
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Feature Engineering
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Model training and predictions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# Cross Validation de los modelos entrenados con k-fold cross validation de 3
fitControl <- trainControl(method = "cv",
                           number = 3,
                           verboseIter = TRUE)

# Entranamiento y validacion de modelo Decision Trees
rpart_fit_grid <- train(x = train_set %>% select(-Label),
                        y = train_set$Label,
                        method = "rpart",
                        trControl = fitControl,
                        na.action = "na.omit")

print(rpart_fit_grid)

# Cross Validation de los modelos entrenados con holdout-set 
val_index <- list("holdout_1" = c(1:10000))

train_index <- list("holdout_1" = setdiff(c(1:nrow(train_set)),
                                          val_index[["holdout_1"]]))

fitControl <- trainControl(method = "cv",  
                           index = train_index,
                           indexOut = val_index,
                           verboseIter = TRUE,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

# Entrenamos modelo
rpart_fit_grid <- train(x = train_set %>% select(-Label),
                        y = train_set$Label,
                        method = "rpart",
                        trControl = fitControl,
                        na.action = "na.omit",
                        metric = "ROC")

print(rpart_fit_grid)

# Entrenamos knn con la misma metodologia de evaluacion
# (age tiene NA y knn no los acepta)

knn_fit_grid <- train(Label ~ ., # Con esta sintaxis hace one-hot-econding
                      data = train_set %>% select(-age),
                      method = "knn",
                      trControl = fitControl,
                      na.action = "na.omit",
                      metric = "ROC")

print(knn_fit_grid)



knn_fit_grid_tuned <- train(Label ~ .,
                            data = train_set %>% select(-age),
                            method = "knn",
                            trControl = fitControl,
                            na.action = "na.omit",
                            preProcess = c("center", "scale"),
                            tuneGrid = expand.grid(k = c(50, 55, 60)),
                            metric = "ROC")

print(knn_fit_grid_tuned)

# Predecimos sobre los datos de evaluacion
eval_preds <- data.frame(id = eval_set$id,
                         Label = predict(knn_fit_grid_tuned,
                                         eval_set, type = "prob")[,"churn"])

mean(eval_preds$Label)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Resultados
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

write.table(eval_preds, "modelo_super_basico.csv",
            sep = ",", row.names = FALSE, quote = FALSE)


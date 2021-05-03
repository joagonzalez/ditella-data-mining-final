rm(list=ls())
setwd('/home/jgonzalez/Downloads/data')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 1er bloque

library(Matrix)
library(data.table)
library(xgboost)
library(dplyr)


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


one_hot_sparse <- function(data_set) {
    
    require(Matrix)
    
    created <- FALSE
    
    if (sum(sapply(data_set, is.numeric)) > 0) {  # Si hay, Pasamos los numÃ©ricos a una matriz esparsa (serÃ­a raro que no estuviese, porque "Label"  es numÃ©rica y tiene que estar sÃ­ o sÃ­)
        out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.numeric), with = FALSE]), "dgCMatrix")  # Si en lugar de pasar un objeto de data table, pasan un data.frame comÃºn no debe decir ", with = FALSE" en ninguna instrucciÃ³n
        created <- TRUE
    }
    
    if (sum(sapply(data_set, is.logical)) > 0) {  # Si hay, pasamos los lÃ³gicos a esparsa y lo unimos con la matriz anterior
        if (created) {
            out_put_data <- cbind2(out_put_data,
                                   as(as.matrix(data_set[,sapply(data_set, is.logical),
                                                         with = FALSE]), "dgCMatrix"))
        } else {
            out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.logical), with = FALSE]), "dgCMatrix")
            created <- TRUE
        }
    }
    
    # Identificamos las columnas que son factor (OJO: el data.frame no deberia tener character)
    fact_variables <- names(which(sapply(data_set, is.factor)))
    
    # Para cada columna factor hago one hot encoding
    i <- 0
    
    for (f_var in fact_variables) {
        
        f_col_names <- levels(data_set[[f_var]])
        f_col_names <- gsub(" ", ".", paste(f_var, f_col_names, sep = "_"))
        j_values <- as.numeric(data_set[[f_var]])  # Se pone como valor de j, el valor del nivel del factor
        
        if (sum(is.na(j_values)) > 0) {  # En categÃ³ricas, trato a NA como una categorÃ­a mÃ¡s
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


## Defino que variables voy a cargar. Las variables fueron elegidas en base a varianza y NA rate y experiencias de corridas previas
TO_KEEP <- c("platform", "age", "install_date", "id",
             "TutorialStart", "TutorialFinish", "Label_max_played_dsi",
             "StartSession_sum_dsi0", "StartSession_sum_dsi1",
             "StartSession_sum_dsi2", "StartSession_sum_dsi3",
             "categorical_1", "categorical_2", "categorical_3",
             "categorical_4", "categorical_5", "categorical_6",
             "categorical_7", "device_model", "country", 
             "BuyCard_sum_dsi0", "BuyCard_sum_dsi1", "BuyCard_sum_dsi2", "BuyCard_sum_dsi3",
             "LoseBattle_sum_dsi1", "LoseBattle_sum_dsi2", "LoseBattle_sum_dsi3",
             "WinBattle_sum_dsi1", "WinBattle_sum_dsi2", "WinBattle_sum_dsi3",
             "WinTournamentBattle_sum_dsi1",
             "StartTournamentBattle_sum_dsi1",
             "StartBattle_sum_dsi0", "StartBattle_sum_dsi2", "StartBattle_sum_dsi3",
             "EnterShop_sum_dsi1", "EnterShop_sum_dsi2", "EnterShop_sum_dsi3", 
             "EnterDeck_sum_dsi0", "EnterDeck_sum_dsi1",
             "OpenChest_sum_dsi3", "OpenChest_sum_dsi2", "OpenChest_sum_dsi1",
             "UpgradeCard_sum_dsi0", 
             "PiggyBankModifiedPoints_sum_dsi3", "PiggyBankModifiedPoints_sum_dsi2", "PiggyBankModifiedPoints_sum_dsi1","PiggyBankModifiedPoints_sum_dsi0", 
             "hard_positive", "hard_negative",
             "soft_positive", "soft_negative")

## Cargo uno de los datasets de entrenamiento del tp
# train_set_1 <- load_csv_data("train_1.csv", sample_ratio = 0.2, sel_cols = TO_KEEP)
train_set_2 <- load_csv_data("train_2.csv", sample_ratio = 0.4, sel_cols = TO_KEEP)
train_set_3 <- load_csv_data("train_3.csv", sample_ratio = 0.8, sel_cols = TO_KEEP)
train_set_4 <- load_csv_data("train_4.csv", sample_ratio = 0.75, sel_cols = TO_KEEP)
train_set_5 <- load_csv_data("train_5.csv", sample_ratio = 0.8, sel_cols = TO_KEEP)
#train_set <- rbind(train_set_1, train_set_2, train_set_3, train_set_4, train_set_5, fill = TRUE)
train_set <- rbind(train_set_2, train_set_3, train_set_4, train_set_5, fill = TRUE)

train_set[, train_sample := TRUE]

## Cargo el dataset de evaluaciÃ³n del TP
eval_set <- load_csv_data("evaluation.csv", sel_cols = setdiff(TO_KEEP, "Label_max_played_dsi"))

eval_set[, train_sample := FALSE]

## Uno los datasets
data_set <- rbind(train_set, eval_set, fill = TRUE)
rm(train_set, eval_set, train_set_2, train_set_3, train_set_4, train_set_5)
gc()

## Hago algo de ingenieria de atributos

# quitamos filas en las cuales no se puede confiar Lable_max_played_dsi == 3 y install_date [383-395]
num_rows <- nrow(data_set)
data_set <- data_set[!(data_set$install_date == 383 & data_set$Label_max_played_dsi == 3), ]
data_set <- data_set[!(data_set$install_date == 384 & data_set$Label_max_played_dsi == 3), ]
data_set <- data_set[!(data_set$install_date == 385 & data_set$Label_max_played_dsi == 3), ]
data_set <- data_set[!(data_set$install_date == 386 & data_set$Label_max_played_dsi == 3), ]
data_set <- data_set[!(data_set$install_date == 387 & data_set$Label_max_played_dsi == 3), ]
data_set <- data_set[!(data_set$install_date == 388 & data_set$Label_max_played_dsi == 3), ]
data_set <- data_set[!(data_set$install_date == 389 & data_set$Label_max_played_dsi == 3), ]
data_set <- data_set[!(data_set$install_date == 390 & data_set$Label_max_played_dsi == 3), ]
data_set <- data_set[!(data_set$install_date == 391 & data_set$Label_max_played_dsi == 3), ]
data_set <- data_set[!(data_set$install_date == 392 & data_set$Label_max_played_dsi == 3), ]
data_set <- data_set[!(data_set$install_date == 393 & data_set$Label_max_played_dsi == 3), ]
data_set <- data_set[!(data_set$install_date == 394 & data_set$Label_max_played_dsi == 3), ]
data_set <- data_set[!(data_set$install_date == 395 & data_set$Label_max_played_dsi == 3), ]
cat(sprintf("registros censurados: %d",  num_rows-nrow(data_set)))

data_set[, Label := as.numeric(Label_max_played_dsi == 3)]
data_set[, Label_max_played_dsi := NULL]

data_set[, max_StartSession_sum := pmax(StartSession_sum_dsi0,
                                        StartSession_sum_dsi1,
                                        StartSession_sum_dsi2,
                                        StartSession_sum_dsi3,
                                        na.rm = TRUE)]

data_set[, min_StartSession_sum := pmin(StartSession_sum_dsi0,
                                        StartSession_sum_dsi1,
                                        StartSession_sum_dsi2,
                                        StartSession_sum_dsi3,
                                        na.rm = TRUE)]

data_set[, min_BuyCard_sum := pmin(BuyCard_sum_dsi1,
                                   BuyCard_sum_dsi2,
                                   BuyCard_sum_dsi3,
                                   na.rm = TRUE)]

data_set[, max_BuyCard_sum := pmax(BuyCard_sum_dsi1,
                                   BuyCard_sum_dsi2,
                                   BuyCard_sum_dsi3,
                                   na.rm = TRUE)]

data_set[, min_LoseBattle_sum := pmin(LoseBattle_sum_dsi1,
                                   LoseBattle_sum_dsi2,
                                   LoseBattle_sum_dsi3,
                                   na.rm = TRUE)]

data_set[, max_LoseBattle_sum := pmax(LoseBattle_sum_dsi1,
                                      LoseBattle_sum_dsi2,
                                      LoseBattle_sum_dsi3,
                                      na.rm = TRUE)]

data_set[, max_EnterShop_sum := pmax(EnterShop_sum_dsi3,
                                     EnterShop_sum_dsi2,
                                     EnterShop_sum_dsi1,
                                      na.rm = TRUE)]

data_set[, min_EnterShop_sum := pmin(EnterShop_sum_dsi3,
                                     EnterShop_sum_dsi2,
                                     EnterShop_sum_dsi1,
                                     na.rm = TRUE)]


data_set[, max_WinBattle_sum := pmax(WinBattle_sum_dsi3,
                                     WinBattle_sum_dsi2,
                                     WinBattle_sum_dsi1,
                                     na.rm = TRUE)]

data_set[, min_WinBattle_sum := pmin(WinBattle_sum_dsi3,
                                     WinBattle_sum_dsi2,
                                     WinBattle_sum_dsi1,
                                     na.rm = TRUE)]


data_set[, max_PiggyBankModifiedPoints_sum := pmax(PiggyBankModifiedPoints_sum_dsi3,
                                     PiggyBankModifiedPoints_sum_dsi2,
                                     PiggyBankModifiedPoints_sum_dsi1,
                                     na.rm = TRUE)]

data_set[, min_PiggyBankModifiedPoints_sum := pmin(PiggyBankModifiedPoints_sum_dsi3,
                                     PiggyBankModifiedPoints_sum_dsi2,
                                     PiggyBankModifiedPoints_sum_dsi1,
                                     na.rm = TRUE)]

data_set[, max_OpenChest_sum := pmax(OpenChest_sum_dsi3,
                                     OpenChest_sum_dsi2,
                                     OpenChest_sum_dsi1,
                                     na.rm = TRUE)]

data_set[, min_OpenChest_sum := pmin(OpenChest_sum_dsi3,
                                     OpenChest_sum_dsi2,
                                     OpenChest_sum_dsi1,
                                     na.rm = TRUE)]

# Analizamos varianza y NAs de variables
variance <- data_set %>% summarise_if(is.numeric, var) # varianza
means <- data_set %>% summarise_if(is.numeric, mean) # promedio
nas <- colSums(is.na(data_set)) # NAs
variance
means

## Hago one hot encoding
data_set <- one_hot_sparse(data_set)
gc()

## Separo en conjunto de training y evaluaciÃ³n de nuevo
train_set <- data_set[as.logical(data_set[,"train_sample"]),]
train_set <- train_set[, setdiff(colnames(train_set), "train_sample")]
eval_set <- data_set[!as.logical(data_set[,"train_sample"]),]
eval_set <- eval_set[, setdiff(colnames(eval_set), "train_sample")]
rm(data_set)
gc()

# Entreno xgboost
val_index <- c(1:10000)

train_index <- setdiff(c(1:nrow(train_set)), val_index)

dtrain <- xgb.DMatrix(data = train_set[train_index,
                                       colnames(train_set) != "Label"],
                      label = train_set[train_index,
                                        colnames(train_set) == "Label"])

dvalid <- xgb.DMatrix(data = train_set[val_index, colnames(train_set) != "Label"],
                      label = train_set[val_index, colnames(train_set) == "Label"])

rgrid <- random_grid(size = 10,
                     min_nrounds = 250, max_nrounds = 600,
                     min_max_depth = 1, max_max_depth = 6,
                     min_eta = 0.0025, max_eta = 0.1,
                     min_gamma = 0, max_gamma = 1,
                     min_colsample_bytree = 0.6, max_colsample_bytree = 1,
                     min_min_child_weight = 1, max_min_child_weight = 10,
                     min_subsample = 0.75, max_subsample = 1)

predicted_models <- train_xgboost(dtrain, dvalid, rgrid)
gc()

res_table <- result_table(predicted_models)
print(res_table)

## Analyze variable importance
var_imp <- as.data.frame(xgb.importance(model = predicted_models[[res_table[1, "i"]]]$model))
print(head(var_imp, 20))

# Predicts in evaluation
eval_preds <- data.frame(id = eval_set[, "id"],
                         Label = predict(predicted_models[[res_table[1, "i"]]]$model,
                                         newdata = eval_set[, setdiff(colnames(eval_set), c("Label"))]))

#print(eval_preds)
# Armo el archivo para subir a Kaggle
options(scipen = 999)  # Para evitar que se guarden valores en formato cientÃ­fico
write.table(eval_preds, "xgbst_modelo_con_random_search_feature_engineering-vfinal-clean.csv",
            sep = ",", row.names = FALSE, quote = FALSE)
options(scipen=0, digits=7)  # Para volver al comportamiento tradicional

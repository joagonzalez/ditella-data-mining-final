#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Trabajo practico Mineria de Datos - Universidad Torcuato Di Tella - MiM + Analytics
# Alumnos: Joaquin Gonzalez, Martin Di Lello, Franco Piccardo
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

rm(list=ls())
setwd('/home/jgonzalez/Downloads/data')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Librerias
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

library(Matrix)
library(data.table)
library(xgboost)
library(dplyr)
library(ggplot2)
library(RColorBrewer)

#set parallel backend
#install.packages("parallelMap")
library(parallel)
library(parallelMap) 
detectCores()
# ayuda para las tareas previas a boosting, ya que al depender de arboles previos no se puede paralelizar
parallelStartSocket(cpus = 6)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Funciones Custom
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Carga de datos
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Las variables fueron elegidas en base a varianza, NA rate, correlacion y experiencias de ejecuciones previas

TO_KEEP <- c("platform", "install_date", "id", "age",
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

## Cargo datasets
SAMPLE_RATIO <- 0.75
sequence <- c(1:5)
for (val in sequence){
    assign(paste0("train_set_", val), load_csv_data(paste("train_", val, ".csv", sep=""), sample_ratio = SAMPLE_RATIO, sel_cols = TO_KEEP))
}

# Excluimos train_set_1 pues observamos caida en performance de xgboost, puede que datos viejos no sean representativos
# en decisiones de churn actuales
train_set <- rbind(train_set_2, train_set_3, train_set_4, train_set_5, fill = TRUE)
train_set[, train_sample := TRUE]

## Cargo dataset de evaluacion
eval_set <- load_csv_data("evaluation.csv", sel_cols = setdiff(TO_KEEP, "Label_max_played_dsi"))
eval_set[, train_sample := FALSE]

## Uno los datasets
data_set <- rbind(train_set, eval_set, fill = TRUE)
rm(train_set, eval_set, train_set_1, train_set_2, train_set_3, train_set_4, train_set_5)
gc()

## Limpieza de dataset

# quitamos filas en las cuales no se puede confiar Lable_max_played_dsi == 3 y install_date [383-395]
# para evitar data leakage
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

# generamos variable churn
data_set[, Label := as.numeric(Label_max_played_dsi == 3)]
data_set[, Label_max_played_dsi := NULL]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Analisis exploratorio de datos
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Analisis de correlacion entre variables predictoras y target
for (val in sequence){
    assign(paste0("train_set_plot", val), load_csv_data(paste("train_", val, ".csv", sep=""), sample_ratio = SAMPLE_RATIO, sel_cols = TO_KEEP))
}

df <- rbindlist(list(train_set_plot1, train_set_plot2, train_set_plot3, train_set_plot4, train_set_plot5), fill=TRUE)
rm(train_set_plot1, train_set_plot2, train_set_plot3, train_set_plot4, train_set_plot5)
gc()

df[, Label := factor(ifelse(Label_max_played_dsi == 3, "Churn", "No churn"), levels=c("No churn", "Churn"))]
df[, BuyCard_sum_dsi3 := factor(ifelse(BuyCard_sum_dsi3 > 0, "Buy", "No Buy"), levels=c("No Buy", "Buy"))]

data_for_plot <- df %>% 
    filter(install_date < 378) %>%
    group_by(WinBattle_sum_dsi3, BuyCard_sum_dsi3) %>%
    summarise(churn_rate = mean(Label=="Churn"))

g1 <- ggplot(data_for_plot, aes(x=WinBattle_sum_dsi3, y=churn_rate, color = BuyCard_sum_dsi3)) +
    geom_line(alpha=0.5) +
    geom_smooth(se=FALSE) +
    ggtitle("Churn rate vs WinBattles") + 
    ylab("Churn rate") +
    xlab("WinBattle_sum_dsi3") +
    xlim(0, 10) +
    ylim(0, 0.35) +
    theme_bw()
ggsave("/home/jgonzalez/Downloads/data/WinBattle_sum_dsi3.jpg", g1, height = 6, width = 12)


data_for_plot <- df %>% 
    filter(install_date < 378) %>%
    group_by(PiggyBankModifiedPoints_sum_dsi3, BuyCard_sum_dsi3) %>%
    summarise(churn_rate = mean(Label=="Churn"))
g2 <- ggplot(data_for_plot, aes(x=PiggyBankModifiedPoints_sum_dsi3, y=churn_rate, color = BuyCard_sum_dsi3)) +
    geom_line(alpha=0.5) +
    geom_smooth(se=FALSE) +
    ggtitle("Churn rate vs PiggyBankModifiedPoints_sum_dsi3") + 
    ylab("Churn rate") +
    xlab("PiggyBankModifiedPoints_sum_dsi3") +
    xlim(0, 20) +
    ylim(0, 0.35) +
    theme_bw()
ggsave("/home/jgonzalez/Downloads/data/PiggyBankModifiedPoints_sum_dsi3.jpg", g2, height = 6, width = 12)
rm(df, data_for_plot)
gc()

# result <- cor(data_set[,-"Label"], data_set$Label)
# plot(result, main = "Correlacion variables predictoras y target", ylab = "correlacion", xlab = "predictores")

# Anàlisis de clases desbalanceadas 
prop.table(table(data_set$Label))
prop.table(table(data_set$platform))
prop.table(table(data_set$install_date))
prop.table(table(data_set$TutorialFinish))
prop.table(table(data_set$PiggyBankModifiedPoints_sum_dsi3))
prop.table(table(data_set$StartTournamentBattle_sum_dsi1))

color <- brewer.pal(length(count), "Set2") 
pie(table(data_set$Label), main= "Balance Label" ,labels = c("Churn", "No Churn"), col = color, cex = 1)
pie(table(data_set$platform), main= "Balance Platform" ,labels = c("Android", "iOS"), col = color, cex = 1)
plot(table(data_set$install_date), main = "Balance install_date", ylab = "Frecuencia", xlab = "install_date", col = rep(1:3, each = 10))

pie(table(data_set$TutorialFinish), main= "Balance TutorialFinish" , col = color, cex = 1, labels = c("No Finish","Finish"))
plot(table(data_set$PiggyBankModifiedPoints_sum_dsi3), main = "Balance PiggyBankModifiedPoints_sum_dsi3", ylab = "Frecuencia", xlab = "PiggyBankModifiedPoints_sum_dsi3", col = rep(1:3, each = 5), xlim = c(1,180))
plot(table(data_set$WinBattle_sum_dsi3), main = "Balance WinBattle_sum_dsi3", ylab = "Frecuencia", xlab = "WinBattle_sum_dsi3", col = rep(1:3, each = 5), xlim = c(1,180))
plot(table(data_set$StartSession_sum_dsi3), main = "Balance StartSession_sum_dsi3", ylab = "Frecuencia", xlab = "StartSession_sum_dsi3", col = rep(1:3, each = 5), xlim = c(1,180))
plot(table(data_set$StartTournamentBattle_sum_dsi1), main = "Balance StartTournamentBattle_sum_dsi1", ylab = "Frecuencia", xlab = "StartTournamentBattle_sum_dsi1", col = rep(1:3, each = 5), xlim = c(1,85))
plot(table(data_set$WinTournamentBattle_sum_dsi1), main = "Balance WinTournamentBattle_sum_dsi1", ylab = "Frecuencia", xlab = "WinTournamentBattle_sum_dsi1", col = rep(1:3, each = 5), xlim = c(1,80))

# Analizamos varianza y NAs de variables
variance <- data_set %>% summarise_if(is.numeric, var) # varianza
means <- data_set %>% summarise_if(is.numeric, mean) # promedio

nas <- colSums(is.na(data_set)) # NAs
variance
means
nas
na.omit(nas[!nas < 100])
pie_labels <- paste0(c("age","Label","categorical_7","country","OpenChest_sum_dsi2"), " = ", round(100 * na.omit(nas[!nas < 100])/sum(na.omit(nas[!nas < 100])), 2), "%")
pie(na.omit(nas[!nas < 100]), main = "Missing values", col = color, cex = 1, labels = pie_labels)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Feature Engineering
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# variable creditor para analizar si hay correlacion entre usuarios deudores y churn
data_set[, creditor := ((soft_positive + hard_positive) - (soft_negative + hard_negative)) > 0]
tail(data_set[, c("creditor", "soft_positive", "hard_positive", "soft_negative", "hard_negative")])


# variables max y min sobre las sum_dsi para analizar si hay correlacion entre extremos de estas variables y churn
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


## Hago one hot encoding
data_set <- one_hot_sparse(data_set)
gc()


## Separo en conjunto de training y evaluacion de nuevo
train_set <- data_set[as.logical(data_set[,"train_sample"]),]
train_set <- train_set[, setdiff(colnames(train_set), "train_sample")]
eval_set <- data_set[!as.logical(data_set[,"train_sample"]),]
eval_set <- eval_set[, setdiff(colnames(eval_set), "train_sample")]
rm(data_set)
gc()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Model Training
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# separo 80k valores para validacion del training set. Por esto es que las predicciones dan mejor aca
# que en kaggle. La data de validacion no fueron datos con lo que entrene a los modelos entonces puede haber
# informacion que no estaba en training set sin importar que haya cross validation

# validacion de modelos

val_index <- c(1:800000) # 20% aprox de los datos

train_index <- setdiff(c(1:nrow(train_set)), val_index)

dtrain <- xgb.DMatrix(data = train_set[train_index,
                                       colnames(train_set) != "Label"],
                      label = train_set[train_index,
                                        colnames(train_set) == "Label"])

dvalid <- xgb.DMatrix(data = train_set[val_index, colnames(train_set) != "Label"],
                      label = train_set[val_index, colnames(train_set) == "Label"])

# usamos random search para optimizacion de hiperparametros, buscamos los mejores iterando aleatoriamente 
# con distintos valores. Para explorar suficientemente el espacio de hiperparametros entrenamos 50 modelos
rgrid <- random_grid(size = 50,
                     #min_nrounds = 300, max_nrounds = 600,
                     min_nrounds = 80, max_nrounds = 200,
                     min_max_depth = 3, max_max_depth = 25,
                     min_eta = 0.0020, max_eta = 0.1,
                     min_gamma = 0, max_gamma = 1,
                     min_colsample_bytree = 0.6, max_colsample_bytree = 0.8,
                     min_min_child_weight = 1, max_min_child_weight = 10,
                     min_subsample = 0.5, max_subsample = 0.8)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Prediction Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

predicted_models <- train_xgboost(dtrain, dvalid, rgrid)
gc()
# imprimimos la performance de todos los modelos entrenados con sus respectivos hiperparamentros ordernados
res_table <- result_table(predicted_models)
print(res_table)

## Analizamos importancia de variables predictoras del modelo con mejor performance en validation set
var_imp <- as.data.frame(xgb.importance(model = predicted_models[[res_table[1, "i"]]]$model))
print(head(var_imp, 20))
# Ploteamos variable importance
mat <- xgb.importance (feature_names = colnames(dtrain),model = predicted_models[[res_table[1, "i"]]]$model)
xgb.plot.importance (importance_matrix = mat[1:20]) 

# Realizamos predicciones en evaluation set con el modelo que tuvo mejor performance en validation set
eval_preds <- data.frame(id = eval_set[, "id"],
                         Label = predict(predicted_models[[res_table[1, "i"]]]$model,
                                         newdata = eval_set[, setdiff(colnames(eval_set), c("Label"))]))

#print(eval_preds)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Results
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Exportamos predicciones
options(scipen = 999)  # Para evitar que se guarden valores en formato cientifico
write.table(eval_preds, "xgbst_modelo_con_random_search_feature_engineering.csv",
            sep = ",", row.names = FALSE, quote = FALSE)
options(scipen=0, digits=7)  # Para volver al comportamiento tradicional

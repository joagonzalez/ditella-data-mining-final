rm(list=ls())
setwd('C:/Users/a310005/Desktop/DiTella/Data Mining/data')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 1er bloque

library(Matrix)
library(data.table)
library(xgboost)


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


## Defino que variables voy a cargar
TO_KEEP <- c("platform", "age", "install_date", "id",
             "TutorialStart", "Label_max_played_dsi",
             "StartSession_sum_dsi0", "StartSession_sum_dsi1",
             "StartSession_sum_dsi2", "StartSession_sum_dsi3",
             "categorical_1", "categorical_2", "categorical_3",
             "categorical_4", "categorical_5", "categorical_6",
             "categorical_7", "device_model")

## Cargo uno de los datasets de entrenamiento del tp
# train_set_1 <- load_csv_data("train_1.csv", sample_ratio = 0.05, sel_cols = TO_KEEP)
# train_set_2 <- load_csv_data("train_2.csv", sample_ratio = 0.05, sel_cols = TO_KEEP)
train_set_3 <- load_csv_data("train_3.csv", sample_ratio = 0.3, sel_cols = TO_KEEP)
train_set_4 <- load_csv_data("train_4.csv", sample_ratio = 0.3, sel_cols = TO_KEEP)
train_set_5 <- load_csv_data("train_5.csv", sample_ratio = 0.3, sel_cols = TO_KEEP)
# train_set <- rbind(train_set_1, train_set_2, train_set_3, train_set_4, train_set_5, fill = TRUE)
train_set <- rbind(train_set_3, train_set_4, train_set_5, fill = TRUE)

train_set[, train_sample := TRUE]

## Cargo el dataset de evaluación del TP
eval_set <- load_csv_data("evaluation.csv", sel_cols = setdiff(TO_KEEP, "Label_max_played_dsi"))

eval_set[, train_sample := FALSE]

## Uno los datasets
data_set <- rbind(train_set, eval_set, fill = TRUE)
rm(train_set, eval_set)
gc()

## Hago algo de ingeniería de atributos
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

## Hago one hot encoding
data_set <- one_hot_sparse(data_set)
gc()

## Separo en conjunto de training y evaluación de nuevo
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
                     min_nrounds = 20, max_nrounds = 100,
                     min_max_depth = 1, max_max_depth = 10,
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

# Armo el archivo para subir a Kaggle
options(scipen = 999)  # Para evitar que se guarden valores en formato científico
write.table(eval_preds, "xgbst_modelo_con_random_search.csv",
            sep = ",", row.names = FALSE, quote = FALSE)
options(scipen=0, digits=7)  # Para volver al comportamiento tradicional

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 2do bloque de código

library(MASS)  # Sólo para tener el dataset Boston disponible

#Dataset con el que vamos a trabajar
head(Boston)
str(Boston)
summary(Boston)

# Veamos los datos
plot(Boston)

# Prueba de clustering jerárquico
hc.complete <- hclust(dist(Boston), method = "complete")
hc.average <- hclust(dist(Boston), method = "average")
hc.single <- hclust(dist(Boston), method = "single")

# Visualizamos las particiones
par(mfrow = c(1, 3))
plot(hc.complete, main = "Complete Linkage", xlab = "" , sub = "" , cex = .9, labels = FALSE)
plot(hc.average, main = "Average Linkage", xlab = "" , sub = "" , cex = .9, labels = FALSE)
plot(hc.single, main = "Single Linkage", xlab = "" , sub = "" , cex = .9, labels = FALSE)
par(mfrow = c(1, 1))

# Vemos que asignaciones quedarían con 4 cortes
asignaciones <- cutree(hc.complete, 4)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 3er bloque de código
library(ggplot2)
library(dplyr)
library(reshape2)

# Cargo y exploto mínimanente los datos
meters_data <- readRDS("preprocessed_data.RDS")
head(meters_data)
str(meters_data)
length(unique(meters_data$meter_id))

# Gráfico exploratorio (25 medidores)
sample_meters <- sample(unique(meters_data$meter_id), 25)
ggplot(meters_data %>% filter(meter_id %in% sample_meters),
       aes(x = reading_time, y = energy, group=meter_id)) +
    geom_line(alpha=0.05) + theme_bw()

# Me quedo sólo con los días de semana
meters_data <- meters_data %>% filter(!weekend)

# Normalizo la serie por medidor
meters_data <- meters_data %>% group_by(meter_id) %>%
                   mutate(energy_std = (energy - mean(energy))/sd(energy)) %>%
                   ungroup()

# Visualizo las series estandarizadas
ggplot(meters_data %>% filter(meter_id %in% sample_meters),
       aes(x = reading_time, y = energy_std, group=meter_id)) +
    geom_line(alpha=0.05) + theme_bw()

# Visualizo cómo se distribuyen en un día típico
ggplot(meters_data %>% filter(meter_id %in% sample_meters[c(1:4)]),
       aes(x = hour_minute, y = energy_std)) +
    geom_boxplot(outlier.shape=NA) +
    scale_y_continuous(limits = c(-1, 3.5)) +
    scale_x_discrete(breaks = unique(meters_data$hour_minute)[seq(0, 48, by = 6)]) + 
    facet_wrap(. ~ meter_id, scales = "free_y") +
    theme_bw()

# Calculo por medidor e intervalo de 30 minutos el consumo mediano
load_curves <- meters_data %>% group_by(meter_id, hour_minute) %>%
    summarize(energy_std_median = median(energy_std))

# Visualizo los patrones de consumo diario
ggplot(load_curves, aes(x = hour_minute, y = energy_std_median, group=meter_id)) +
    scale_x_discrete(breaks = unique(meters_data$hour_minute)[seq(0, 48, by = 6)]) + 
    geom_line(alpha=0.025) + theme_bw()

# Ejecuto k-medias sobre las curvas de carga
data_for_cluster <- dcast(data = load_curves, meter_id ~ hour_minute,
                          value.var = "energy_std_median")  # Vean https://seananderson.ca/2013/10/19/reshape/
head(data_for_cluster)

clusters <- kmeans(data_for_cluster %>% select(-meter_id), centers=6)

data_for_cluster$cluster_k <- factor(clusters$cluster)

# Visualizo los clusters
load_curves <- merge(load_curves,
                     data_for_cluster %>% select(meter_id, cluster_k),
                     by = "meter_id")

ggplot(load_curves,
       aes(x = hour_minute, y = energy_std_median,
           group = meter_id, color = cluster_k)) +
       coord_cartesian(ylim = c(-1, 3), expand=0) +
       geom_line(alpha = 0.05) + 
       stat_summary(fun = mean, geom = "line",
                    lwd = 1, col = "black", aes(group=1)) + # see: https://stackoverflow.com/questions/40879800/add-mean-line-to-ggplot?rq=1
       scale_x_discrete(breaks = unique(meters_data$hour_minute)[seq(0, 48, by = 12)]) + 
       facet_wrap( ~ cluster_k) + xlab("Time of the day") +
       ylab("Energy (Z-soce)") + theme_bw() + theme(legend.position = "none")

# Intentemos con algoritmos jerárquicos
hc.complete <- hclust(dist(data_for_cluster %>% select(-meter_id, -cluster_k)),
                           method = "complete")

plot(hc.complete, main = "Complete Linkage", xlab = "" , sub = "" , cex = .9, labels = FALSE)

data_for_cluster$cluster_h <- factor(cutree(hc.complete, 6))

load_curves <- merge(load_curves,
                     data_for_cluster %>% select(meter_id, cluster_h),
                     by = "meter_id")

ggplot(load_curves,
       aes(x = hour_minute, y = energy_std_median,
           group = meter_id, color = cluster_h)) +
    coord_cartesian(ylim = c(-1, 3), expand=0) +
    geom_line(alpha = 0.05) + 
    stat_summary(fun = mean, geom = "line",
                 lwd = 1, col = "black", aes(group=1)) + # see: https://stackoverflow.com/questions/40879800/add-mean-line-to-ggplot?rq=1
    scale_x_discrete(breaks = unique(meters_data$hour_minute)[seq(0, 48, by = 12)]) + 
    facet_wrap( ~ cluster_h) + xlab("Time of the day") +
    ylab("Energy (Z-soce)") + theme_bw() + theme(legend.position = "none")

# Por qué ocurre lo siguiente?
X <- matrix(runif(1000000 * 20), ncol = 20)
dim(X)

kmeans(X, centers = 6)  # Esto se ejecuta y eventualmente termina
hclust(dist(X), method = "complete")  # Esto ni siquiera se ejecuta

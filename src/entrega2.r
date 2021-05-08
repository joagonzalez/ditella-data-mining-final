library(data.table)
library(dplyr)
library(ggplot2)

setwd('/home/jgonzalez/Downloads/data')

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


df1 <- load_csv_data("train_1.csv", sample_ratio = 0.8)  # Seguro esta no es una manera elegante de unir los df
df2 <- load_csv_data("train_2.csv", sample_ratio = 0.8)
df3 <- load_csv_data("train_3.csv", sample_ratio = 0.8)
df4 <- load_csv_data("train_4.csv", sample_ratio = 0.8)
df5 <- load_csv_data("train_5.csv", sample_ratio = 0.8)
df <- rbindlist(list(df1, df2, df3, df4, df5), fill=TRUE)
rm(df1, df2)
gc()

df[, Label := factor(ifelse(Label_max_played_dsi == 3, "Churn", "No churn"), levels=c("No churn", "Churn"))]
df[, BuyCard_sum_dsi3 := factor(ifelse(BuyCard_sum_dsi3 > 0, "Buy", "No Buy"), levels=c("No Buy", "Buy"))]

# Figure 1
g1 <- ggplot(df, aes(x = Label, y = log10(StartSession_sum_dsi3+1))) +
          geom_boxplot() +
          facet_wrap(~platform) +
          xlab(NULL) +
          ylab("StartSession_sum_dsi3 (log10(x+1))") +
          ggtitle("Distribution of StartSession_sum_dsi3 for churners and no churners (by platform)") + 
          theme_bw()

ggsave("C/home/jgonzalez/Downloads/data/gallardeta_figure1.jpg", g1, height = 6, width = 12)

# Figure 2
data_for_plot <- df %>% 
                     filter(install_date < 378) %>%
                     group_by(install_date, TutorialFinish) %>%
                     summarise(churn_rate = mean(Label=="Churn"))

g2 <- ggplot(data_for_plot, aes(x=install_date, y=churn_rate, color = TutorialFinish)) +
          geom_line(alpha=0.5) +
          geom_smooth(se=FALSE) +
          ggtitle("Churn rate through time (by TutorialFinish)") + 
          ylab("Churn rate") +
          xlab("Install date") +
          theme_bw()

ggsave("/home/jgonzalez/Downloads/data/gallardeta_figure2.jpg", g2, height = 6, width = 12)

hist(df$age, breaks = c(20, 50))


g3 <- ggplot(data=df, aes(df$age)) + geom_histogram(breaks=seq(20, 50, by=2), col="red", fill="green")
ggsave("/home/jgonzalez/Downloads/data/gallardeta_figure3.jpg", g3, height = 6, width = 12)

g4 <- ggplot(data=df, aes(df$WinBattle_sum_dsi3)) + geom_histogram(breaks=seq(20, 50, by=2), col="blue", fill="blue") + 
      xlab("WinBattle 3 days since install ") +
      ylab("Count") +
      ggtitle("Distribution of WinBattle_sum_dsi3 for churners and no churners")
  
ggsave("/home/jgonzalez/Downloads/data/gallardeta_figure4.jpg", g4, height = 6, width = 12)

data_for_plot <- df %>% 
  filter(install_date < 378) %>%
  group_by(WinBattle_sum_dsi3, BuyCard_sum_dsi3) %>%
  summarise(churn_rate = mean(Label=="Churn"))

g5 <- ggplot(data_for_plot, aes(x=WinBattle_sum_dsi3, y=churn_rate, color = BuyCard_sum_dsi3)) +
  geom_line(alpha=0.5) +
  geom_smooth(se=FALSE) +
  ggtitle("Churn rate vs WinBattles") + 
  ylab("Churn rate") +
  xlab("WinBattle_sum_dsi3") +
  xlim(0, 10) +
  ylim(0, 0.35) +
  theme_bw()

ggsave("/home/jgonzalez/Downloads/data/gallardeta_figure5.jpg", g5, height = 6, width = 12)


data_for_plot <- df %>% 
  filter(install_date < 378) %>%
  group_by(PiggyBankModifiedPoints_sum_dsi3, BuyCard_sum_dsi3) %>%
  summarise(churn_rate = mean(Label=="Churn"))

g6 <- ggplot(data_for_plot, aes(x=PiggyBankModifiedPoints_sum_dsi3, y=churn_rate, color = BuyCard_sum_dsi3)) +
  geom_line(alpha=0.5) +
  geom_smooth(se=FALSE) +
  ggtitle("Churn rate vs PiggyBankModifiedPoints_sum_dsi3") + 
  ylab("Churn rate") +
  xlab("PiggyBankModifiedPoints_sum_dsi3") +
  xlim(0, 20) +
  ylim(0, 0.35) +
  theme_bw()

ggsave("/home/jgonzalez/Downloads/data/gallardeta_figure6.jpg", g6, height = 6, width = 12)

TO_KEEP <- c("age", "install_date", "id",
             "TutorialStart", "TutorialFinish", "Label_max_played_dsi",
             "StartSession_sum_dsi0", "StartSession_sum_dsi1",
             "StartSession_sum_dsi2", "StartSession_sum_dsi3",
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
data_set <- load_csv_data("train_2.csv", sample_ratio = 0.9, sel_cols = TO_KEEP)
data_set[, Label := as.numeric(Label_max_played_dsi == 3)]

result <- cor(data_set[,-"Label"], data_set$Label)
plot(result, main = "Correlacion variables predictoras y target", ylab = "correlacion", xlab = "predictores")

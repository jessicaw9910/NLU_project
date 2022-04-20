library(dplyr)
library(ggplot2)
library(janitor)

path <- 'C:/Users/jessb/OneDrive/PhD/NYU/NLU/project/assets'

df_gdsc <- read.csv(paste(path, 'GDSC2_fitted_dose_response_25Feb20.csv', sep = '/'))
# View(df_gdsc)
idx_unclass <- which(df_gdsc$TCGA_DESC == 'UNCLASSIFIED' |
                       df_gdsc$TCGA_DESC == '')
# table(df_gdsc$TCGA_DESC[-idx_unclass])
# length(unique(paste(df_gdsc$DRUG_ID, df_gdsc$TCGA_DESC)))
# length(unique(paste(df_gdsc$DRUG_ID[-idx_unclass], df_gdsc$TCGA_DESC[-idx_unclass])))

df_gdsc <- df_gdsc[-idx_unclass, ]
df_gdsc$IC50 <- exp(df_gdsc$LN_IC50)
# table(df_gdsc$DRUG_NAME, df_gdsc$TCGA_DESC) %>% View()

df_mad <- df_gdsc %>%
  group_by(DRUG_NAME, TCGA_DESC) %>%
  summarise(MEDIAN_IC50 = median(IC50), 
            MEAN_IC50 = mean(IC50), 
            MAD = mad(AUC), 
            N = n())

idx_five <- which(df_mad$N < 5)
df_mad <- df_mad[-idx_five, ]
df_mad$BROAD_EFFECT <- (df_mad$MAD > 0.13 & df_mad$MEAN_IC50 < 1)
df_mad$SENSITIVE <- (df_mad$MEDIAN_IC50 < 1)

# sum(df_mad$MEDIAN_IC50 < 1)
# sum(df_mad$MEAN_IC50 < 1)

# df_gdsc[(df_gdsc$DRUG_NAME == 'ABT737' & df_gdsc$TCGA_DESC == 'LCML'), ] %>% View()

# ggplot(df_mad, aes(MEDIAN_IC50)) + 
#   geom_histogram()

# df_mad$NO_EFFECT <- mapply(function(x, y) ifelse(sum(df_gdsc$IC50[(df_gdsc$DRUG_NAME == x & df_gdsc$TCGA_DESC == y)] < 1 |
#                                                        df_gdsc$AUC[(df_gdsc$DRUG_NAME == x & df_gdsc$TCGA_DESC == y)] > 0.2) < 5, 
#                                                  1, 0),
#                            df_mad$DRUG_NAME, df_mad$TCGA_DESC)


df_tcga <- read.csv(paste(path, 'tcga_dictionary.csv', sep = '/'))
df_tcga <- clean_names(df_tcga)

df_mad <- merge(df_mad, df_tcga, 
                by.x = 'TCGA_DESC', 
                by.y = 'study_abbreviation', 
                all.x = TRUE)

# temp[is.na(df_mad$study_name), 'TCGA_DESC'] %>% table()

tcga_desc <- c('ALL', 'COREAD', 'MB', 'MM', 'NB', 'SCLC')
study_name <- c('Acute lymphoblastic leukemia',
                'Colon and rectal adenocarcinoma',
                'Medulloblastoma',
                'Multiple myeloma',
                'Neuroblastoma',
                'Small cell lung cancer')

for (i in seq_along(tcga_desc)){
  df_mad$study_name[df_mad$TCGA_DESC == tcga_desc[i]] <- study_name[i]
}

# df_mad$study_name[df_mad$TCGA_DESC == 'ALL'] <- 'Acute lymphoblastic leukemia'
# df_mad$study_name[df_mad$TCGA_DESC == 'COREAD'] <- 'Colon and rectal adenocarcinoma'
# df_mad$study_name[df_mad$TCGA_DESC == 'MB'] <- 'Medulloblastoma'
# df_mad$study_name[df_mad$TCGA_DESC == 'MM'] <- 'Multiple myeloma'
# df_mad$study_name[df_mad$TCGA_DESC == 'NB'] <- 'Neuroblastoma'
# df_mad$study_name[df_mad$TCGA_DESC == 'SCLC'] <- 'Small cell lung cancer'

# dim(df_mad)

# View(df_mad)

write.csv(df_mad, paste(path, 'df_mad.csv', sep = '/'), row.names = FALSE)

##################
### NOT IN USE ###
##################

# df <- read.csv(paste(path, 'ComboDrugGrowth_Nov2017.csv', sep = '/'))
# df_names <- read.csv(paste(path, 'ComboCompoundNames_small.txt', sep = '/'), 
#                      sep = '\t', header = FALSE)
# colnames(df_names) <- c('NSC', 'DRUGS')
# 
# df_mean <- df %>% 
#   group_by(NSC1, NSC2, PANEL) %>%
#   summarise(MEAN = mean(SCORE), MEDIAN = median(SCORE))
# 
# df_mean <- df_mean[-is.na(df_mean$mean), ]
# 
# df_mean$DRUG1 <- sapply(df_mean$NSC1,
#                         function(x) df_names$DRUGS[match(x, df_names$NSC)])
# df_mean$DRUG2 <- sapply(df_mean$NSC2,
#                         function(x) df_names$DRUGS[match(x, df_names$NSC)])
# 
# View(df_mean)
# 
# ggplot(df_mean, aes(x = mean)) +
#   geom_histogram(binwidth = 1)
#  facet_wrap(~PANEL)

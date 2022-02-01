library(data.table)
library(tidyverse)
library(purrr)
library(ggplot2)
library(ggpubr)
library(MOFA2)
library(dplyr)

dt <- fread("ftp://ftp.ebi.ac.uk/pub/databases/mofa/microbiome/data.txt.gz")
metadata <- fread("ftp://ftp.ebi.ac.uk/pub/databases/mofa/microbiome/metadata.txt.gz")

str(dt)
dim(dt)
str(metadata)
dim(metadata)

dt_wide = copy(dt)[, fea := paste(feature, view, sep = "_")][,c("sample","value","fea")] %>%
  reshape(idvar = "sample", timevar = "fea", direction = "wide")
names(dt_wide)[2:241] = sapply(names(dt_wide)[2:241], function(x) substring(x,7))
dat = copy(metadata)[, Category := case_when(
  Category == "Sepsis" ~ 0,
  Category == "Non septic ICU" ~ 1,
  Category == "Healthy, no antibiotics" ~ 2,
  Category == "Healthy, antibiotics" ~ 3)][, c("sample", "Category")]

dat = merge(dt_wide, dat, by="sample")

row.has.na <- apply(dat, 1, function(x){any(is.na(x))})
dat$sample[row.has.na] # "TKI_F22" "TKI_F56"
dat = dat[!row.has.na,]

dat = dat %>% arrange(Category)

write.csv(dat, "./data/cleaned_data.csv",row.names = FALSE)

dat_test = read.csv("./data/cleaned_data.csv")
View(dat_test)

#########################################
# data description:                     #
# dim(dat): 57, 242                     #
# col 2-181: Bacteria                   #
# col 182-199: Fungi                    #
# col 200-241: Viruses                  #
# row 1-23: 0(Sepsis)                   #
# row 24-32: 1(Non septic ICU)          #
# row 33-52: 2(Healthy, no antibiotics) #
# row 53-57: 3(Healthy, antibiotics)    #
#########################################
g0 = sample(0:22, 23 )
g1 = sample(23:31, 9)
g2 = sample(32:51, 20)
g3 = sample(52:56, 5)
i1 = c(g0[1:5],g1[1:2],g2[1:4],g3[1])
i2 = c(g0[6:10],g1[3:4],g2[5:8],g3[2])
i3 = c(g0[11:14],g1[5:6],g2[9:12],g3[3])
i4 = c(g0[15:18],g1[7:8],g2[13:16],g3[4])
i5 = c(g0[19:23],g1[9],g2[17:20],g3[5])
i1 # 1 15 13 12  8 25 28 50 41 35 34 54
i2 # 16  3 18 19 21 30 24 51 47 46 33 55
i3 # 17  0 11  6 29 23 44 39 43 40 53
i4 # 10  4  9  2 26 27 38 36 32 42 52
i5 # 5 22 20 14  7 31 45 37 48 49 56

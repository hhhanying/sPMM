library(data.table)
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

dt_wide = copy(dt)[, fea := paste(feature, view, sep = ",")][,c("sample","value","fea")] %>%
  reshape(idvar = "sample", timevar = "fea", direction = "wide")
dat = copy(metadata)[, Category := case_when(
  Category == "Sepsis" ~ 0,
  Category == "Non septic ICU" ~ 1,
  Category == "Healthy, no antibiotics" ~ 2,
  Category == "Healthy, antibiotics" ~ 3)][, c("sample", "Category")]

dat = merge(dt_wide, dat, by="sample")

row.has.na <- apply(dat, 1, function(x){any(is.na(x))})
dat$sample[row.has.na] # "TKI_F22" "TKI_F56"
dat = dat[!row.has.na,]

write.csv(dat, "cleaned_data.csv")





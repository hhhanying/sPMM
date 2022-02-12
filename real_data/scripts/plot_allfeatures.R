library(ggplot2)
library(tidyverse)
k0 = 9
su_file = paste("./simulated/sim_x_", as.character(k0),".csv",sep = "")
dat_su1 = read.table(su_file, sep = ",",header = FALSE)
dat_un1 = read.table(un_file, sep = ",",header = FALSE)
true_data = read.csv("/Users/Patron/Documents/GitHub/sPMM/real_data/data/cleaned_data.csv")
colnames(dat_un1) = colnames(dat_su1) = c(colnames(true_data)[2:241],"sample")
i = 1
fea = colnames(dat_su1)[i]
fea
dat_su1 %>% ggplot(aes(x=sample, y=Acidaminococcus_Bacteria, group = sample)) +
  geom_boxplot() +
  geom_point(data = true_data,
             aes(x = 1:57, y = Acinetobacter.phage_Viruses), size = 0.5, colour="red")
dat_un1 %>% ggplot(aes(x=sample, y=Acidaminococcus_Bacteria, group = sample)) +
  geom_boxplot() +
  geom_point(data = true_data,
             aes(x = 1:57, y = Acinetobacter.phage_Viruses), size = 0.5, colour="red")
impor_gene = c(217, 202, 205, 231, 200, 198, 199, 224, 209, 207, 211, 204, 233,
               235, 218, 234, 230, 223, 201, 206)-1
impor_gene
colnames(true_data)[218]
col_no = 50
f = function(x){
  col_no = x
true_data %>% ggplot(aes_string(x="Category", y = colnames(true_data)[col_no], group = "Category")) +
  geom_boxplot()}
f(181)
table(true_data$Category)


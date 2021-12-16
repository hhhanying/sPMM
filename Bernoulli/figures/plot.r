setwd("C:\\Users\\Patron\\Desktop\\sPMM\\Bernoulli")
library(tidyr)
library(dplyr)
library(ggplot2)

res = read.csv("data\\res_fix_ntopic.csv")

res = res %>% 
  mutate(supervised = (supervised_1+supervised_2+supervised_3+supervised_4+supervised_5)/5) %>% 
  mutate(unsupervised = (unsupervised_1+unsupervised_2+unsupervised_3+unsupervised_4+unsupervised_5)/5) %>% 
  mutate(supervised_var = apply(res[13:17], 1, var)) %>%
  mutate(unsupervised_var = apply(res[18:22], 1, var)) %>%
  mutate(prior = apply(res[1:2], 1, function(x) paste("(", as.character(x[1]), ", ", as.character(x[2]),")", sep = "")))
    
View(res)
dim(res)

res %>%
  pivot_longer(supervised:unsupervised, names_to = "method", values_to = "accuracy") %>%
  ggplot(aes(x=nlabel,y=accuracy,color=log(a)))+geom_point(size=1)+facet_grid(method ~ d,space ="free_x")
res %>%
  pivot_longer(supervised:unsupervised, names_to = "method", values_to = "accuracy") %>%
  ggplot(aes(x=nlabel,y=accuracy,color=method))+
  geom_point()+labs(x="number of unshared topics")+
  facet_grid(prior ~ d,space ="free_x")

res_d100 = read.csv("data\\res_d100.csv")

res_d100 = res_d100 %>% 
  mutate(supervised = (supervised_1+supervised_2+supervised_3+supervised_4+supervised_5)/5) %>% 
  mutate(unsupervised = (unsupervised_1+unsupervised_2+unsupervised_3+unsupervised_4+unsupervised_5)/5) %>% 
  mutate(supervised_var = apply(res_d100[13:17], 1, var)) %>%
  mutate(unsupervised_var = apply(res_d100[18:22], 1, var)) 
dim(res_d100)
tem = res %>%
  filter(d == 100)
res =rbind(res_d100, tem)
dim(res)

res %>%
  pivot_longer(supervised:unsupervised, names_to = "method", values_to = "accuracy") %>%
  ggplot(aes(x=nlabel,y=accuracy,color=method))+
  geom_point(size=1)+
  facet_grid(k1 ~ k0)

# random
res = read.csv("data\\res_random.csv")
res = res %>% 
  mutate(supervised = (supervised_1+supervised_2+supervised_3+supervised_4+supervised_5)/5) %>% 
  mutate(unsupervised = (unsupervised_1+unsupervised_2+unsupervised_3+unsupervised_4+unsupervised_5)/5) %>% 
  mutate(supervised_var = apply(res[13:17], 1, var)) %>%
  mutate(unsupervised_var = apply(res[18:22], 1, var))  %>%
  pivot_longer(supervised:unsupervised, names_to = "method", values_to = "accuracy")

dim(res)

ggplot(res,aes(x=k0,y=accuracy,color=method))+geom_point()+labs(x="number of unshared topics")
ggplot(res,aes(x=k1,y=accuracy,color=method))+geom_point()+labs(x="number of shared topics")
ggplot(res,aes(x=nlabel*k0-k1,y=accuracy,color=method))+geom_point()+labs(x="k0*nlabel-k1")
ggplot(res,aes(x=d,y=accuracy,color=method))+geom_point()+labs(x="dimension of data")
ggplot(res,aes(x=nlabel,y=accuracy,color=method))+geom_point()+labs(x="number of labels")

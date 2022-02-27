
G %>%  pivot_longer(everything(), names_to="topic", values_to="membership") %>%
  ggboxplot(x="topic", y = "membership") +
  ggtitle("G")

# row 1-23: 0(Sepsis)                   #
# row 24-32: 1(Non septic ICU)          #
# row 33-52: 2(Healthy, no antibiotics) #
# row 53-57: 3(Healthy, antibiotics) 

i = 0
start = i*k0+1
U_supervised[start: (start + k0)]%>% mutate(sample = dat$sample) %>%
  merge(metadata,by="sample",sort=FALSE)%>% 
  pivot_longer(starts_with("V") , names_to="topic", values_to="membership") %>%
  ggboxplot(x="topic", y = "membership", color="Category") +
  ggtitle("U supervised")

i = 1
start = i*k0+1
U_supervised[start: (start + k0-1)]%>% mutate(sample = dat$sample) %>%
  merge(metadata,by="sample",sort=FALSE)%>% 
  pivot_longer(starts_with("V") , names_to="topic", values_to="membership") %>%
  ggboxplot(x="topic", y = "membership", color="Category") +
  ggtitle("U supervised")

i = 2
start = i*k0+1
U_supervised[start: (start + k0-1)]%>% mutate(sample = dat$sample) %>%
  merge(metadata,by="sample",sort=FALSE)%>% 
  pivot_longer(starts_with("V") , names_to="topic", values_to="membership") %>%
  ggboxplot(x="topic", y = "membership", color="Category") +
  ggtitle("U supervised")

i = 3
start = i*k0+1
U_supervised[start: (start + k0-1)]%>% mutate(sample = dat$sample) %>%
  merge(metadata,by="sample",sort=FALSE)%>% 
  pivot_longer(starts_with("V") , names_to="topic", values_to="membership") %>%
  ggboxplot(x="topic", y = "membership", color="Category") +
  ggtitle("U supervised")

i = 4
start = i*k0+1
U_supervised[start: (start + k0-1)]%>% mutate(sample = dat$sample) %>%
  merge(metadata,by="sample",sort=FALSE)%>% 
  pivot_longer(starts_with("V") , names_to="topic", values_to="membership") %>%
  ggboxplot(x="topic", y = "membership", color="Category") +
  ggtitle("U supervised")

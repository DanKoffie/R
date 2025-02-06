#Load the libraries
library(dplyr)
library(readr)
library(ggplot2)
library(corrplot)
library(tidymodels)

#set working directory
setwd("C:/Users/DAN KOFFIE/Desktop/NANA")

#DATA LOADING AND PRE-PROCESSING

#loading first dataset 
covid_data <- read.csv("covid19_Confirmed_dataset.csv")
View(covid_data)

#View the dimensions of the covid dataset
dim(covid_data)

#Check for missing values
sum(is.na(covid_data))

#Summarize the dataset
summary(covid_data)

#Aggregate the COVID-19 data by country
covid_data_aggregated <- covid_data %>%
  select(-Lat, -Long, -Province.State) %>%
  group_by(Country.Region) %>%
  summarise(across(everything(), sum))
View(covid_data_aggregated)

#Calculate the maximum infection rate for each country and add it to the covid data
max_infection_rates <- covid_data_aggregated %>%
  rowwise() %>%
  mutate(max_infection_rate = max(diff(as.numeric(c_across(starts_with("X")))), na.rm = TRUE))
View(max_infection_rates)

#Create a new dataframe with only the needed columns
new_covid_data <- max_infection_rates %>%
  select(Country.Region, max_infection_rate)
View(new_covid_data)
write.csv(new_covid_data, "df.csv", row.names = FALSE)

#loading second dataset 
hapiness_data <- read.csv("worldwide_happiness_report.csv")
View(hapiness_data)

#View the dimensions of the happiness dataset
dim(hapiness_data)

#Check for missing values
sum(is.na(hapiness_data))

#Summarize the dataset
summary(hapiness_data)

#create a subset (feature selection)
new_hapiness_data <- hapiness_data %>%
  select(Country.or.region, Social.support, Healthy.life.expectancy,Freedom.to.make.life.choices,GDP.per.capita)
View(new_hapiness_data)

#View two final datasets
View(new_covid_data)
View(new_hapiness_data)

#Rename the columns to have a common key for merging
names(new_covid_data)[names(new_covid_data) == "Country.Region"] <- "Country"
names(new_hapiness_data)[names(new_hapiness_data) == "Country.or.region"] <- "Country"


#Merge the datasets on the 'Country' column
merged_df <- merge(new_covid_data, new_hapiness_data, by = "Country", all = FALSE)
View(merged_df)


#EDA on merged Dataset 

#structure of the data
str(merged_df)

#summary of the data
summary(merged_df)

#Distribution of Variables
hist(merged_df$max_infection_rate, main="Max Infection Rate", xlab="Max Infection Rate")
hist(merged_df$Social.support, main="Social Support", xlab="Social Support")
hist(merged_df$Healthy.life.expectancy, main="Healthy Life Expectancy", xlab="Healthy Life Expectancy")
hist(merged_df$Freedom.to.make.life.choices, main="Freedom to Make Life Choices", xlab="Freedom to Make Life Choices")
hist(merged_df$GDP.per.capita, main="GDP per Capita", xlab="GDP per Capita")


#Correlation matrix
cor_matrix <- cor(merged_df[, sapply(merged_df, is.numeric)])
corrplot(cor_matrix, method="circle")

#Scatter plots
plot(merged_df$GDP.per.capita, merged_df$max_infection_rate, main="GDP vs Max Infection Rate", xlab="GDP per Capita", ylab="Max Infection Rate")
plot(merged_df$Social.support, merged_df$max_infection_rate, main="Social Support vs Max Infection Rate", xlab="Social Support", ylab="Max Infection Rate")

#Box plots
boxplot(merged_df$max_infection_rate, main="Boxplot of Max Infection Rate", ylab="Max Infection Rate")
boxplot(merged_df$Social.support, main="Boxplot of Social Support", ylab="Social Support")

#Density plots
plot(density(merged_df$max_infection_rate), main="Density Plot of Max Infection Rate", xlab="Max Infection Rate")
plot(density(merged_df$Social.support), main="Density Plot of Social Support", xlab="Social Support")

#Pair plots
pairs(merged_df[ , sapply(merged_df, is.numeric)], main="Pair Plot")


#PREDICTIVE MODELING

#View Data 
View(merged_df)

#Prepare the data
merged_df <- merged_df %>% 
  select(max_infection_rate, Social.support, Healthy.life.expectancy, Freedom.to.make.life.choices, GDP.per.capita)

##Data Partitioning
set.seed(100)
data_split <- initial_split(merged_df, prop = 0.75)

#Training data
train_data <- data_split %>%
  training()
glimpse(train_data)


#Testing data
test_data <- data_split %>%
  testing()
glimpse(test_data)

nrow(train_data)
nrow(test_data)

#Define the model recipe
data_recipe <- recipe(max_infection_rate ~ ., data = train_data)

#Prep the recipe and bake it to create the pre-processed training data
prepped_recipe <- prep(data_recipe, training = train_data)
train_data_prepped <- bake(prepped_recipe, new_data = train_data)
test_data_prepped <- bake(prepped_recipe, new_data = test_data)

#Model specification
#Random Forest
rf_model <- rand_forest(mtry = 3, trees = 500, min_n = 5) %>% 
  set_mode("regression") %>% 
  set_engine("randomForest")

#Decision Tree
dt_model <- decision_tree(cost_complexity = 0.01, tree_depth = 10, min_n = 5) %>% 
  set_mode("regression") %>% 
  set_engine("rpart")

#Linear Regression
lr_model <- linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm")

#Model fitting

#Random Forest
rf_fit <- rf_model %>% 
  fit(max_infection_rate ~ ., data = train_data_prepped)

#Decision Tree
dt_fit <- dt_model %>% 
  fit(max_infection_rate ~ ., data = train_data_prepped)

#Linear Regression
lr_fit <- lr_model %>% 
  fit(max_infection_rate ~ ., data = train_data_prepped)

#Make predictions
rf_predictions <- predict(rf_fit, new_data = test_data_prepped) %>% bind_cols(test_data_prepped)
dt_predictions <- predict(dt_fit, new_data = test_data_prepped) %>% bind_cols(test_data_prepped)
lr_predictions <- predict(lr_fit, new_data = test_data_prepped) %>% bind_cols(test_data_prepped)

rf_predictions
dt_predictions
lr_predictions

#Model Evaluation
#Random Forest
rf_metrics <- rf_predictions %>% 
  metrics(truth = max_infection_rate, estimate = .pred) %>% 
  mutate(model = "Random Forest")

#Decision Tree
dt_metrics <- dt_predictions %>% 
  metrics(truth = max_infection_rate, estimate = .pred) %>% 
  mutate(model = "Decision Tree")

#Linear Regression
lr_metrics <- lr_predictions %>% 
  metrics(truth = max_infection_rate, estimate = .pred) %>% 
  mutate(model = "Linear Regression")

#Combine the metrics
all_metrics <- bind_rows(rf_metrics, dt_metrics, lr_metrics) %>% 
  select(model, .metric, .estimate) %>% 
  spread(.metric, .estimate)

View(all_metrics)

#Decision on which model to use
best_model <- all_metrics %>% 
  filter(mae == min(mae)) %>% 
  select(model)

print(paste("The best model based on MAE is:", best_model$model))






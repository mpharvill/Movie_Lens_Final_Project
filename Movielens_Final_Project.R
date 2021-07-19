## Matt Harvill
## Movielens Final Project
## Harvardx: PH125.9x - Capstone
## https://github.com/mpharvill

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

#### Note: this process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

### MovieLens 10M dataset:
### https://grouplens.org/datasets/movielens/10m/
### http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

### if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
### if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

### Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

### Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

### Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

## Data Analysis ###

### first 7 rows with header
head(edx) 

### basic summary statistics
summary(edx)

### number of unique users and movies
edx %>% summarise(n_users = n_distinct(userId),
                  n_movies = n_distinct(movieId))

### ratings distribution
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "black") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating distribution")

### five most given ratings in order from most to least
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))

### average rating across all movies
edx %>% summarise(mean(rating))

### distribution of ratings by movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black", fill = "blue") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Distribution of movie ratings")

### distribution of ratings by user
edx %>% 
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black", fill = "red") +
  scale_x_log10() +
  xlab("Number of users") +
  ylab("Number of ratings") +
  ggtitle("Distribution of user ratings")

### Residual Mean Square Error formula for testing accuracy of predictions
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# Building the Recommendation System ####

## Average movie rating model ###

### average of all ratings across all users
mu <- mean(edx$rating)
mu

### predict all unknown ratings with mu
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse

### create a table to store results of prediction approaches
rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

### table showing naive RMSE effect results
rmse_results %>% knitr::kable()

### model accounting for movie effect (b_i)
movie_avgs <- edx %>% 
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))

### plot the number of movies with computed b_i
movie_avgs %>% qplot(b_i, geom = "histogram", bins = 10, data = ., color = I("black"),
                     ylab = "Number of movies", main = "Number of movies with computed b_i")

### test and save RMSE results
predicted_ratings <- mu + validation %>%
  left_join(movie_avgs, by='movieId') %>%
  .$b_i
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",
                                     RMSE = model_1_rmse))

### table showing movie effect model results
rmse_results %>% knitr::kable()

### plot showing user effect for users with more than 100 ratings
edx %>% 
  group_by(userId) %>%
  summarise(b_u = mean(rating)) %>%
  filter(n()>100) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  ggtitle("User Effect Distribution")

### model accounting for user effect (b_u) + movie effect
user_avgs <- edx %>%
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

### test and save new RMSE results
predicted_ratings <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie + User Effects Model",
                                     RMSE = model_2_rmse))

### table showing movie + user effect model results
rmse_results %>% knitr::kable()

## Regularization of movie + user effect model ##

### lambda is a tuning parameter, chosen by cross-validation
lambdas <- seq(0, 10, 0.25)

#### below code may take several minutes to run
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})

### plot lambdas and RMSEs to select optimal lambda
qplot(lambdas, rmses)

### find optimal lambda
lambda <- lambdas[which.min(rmses)]
lambda

### regularized model accounting for movie + user effect
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Regularized Movie + User Effect Model",
                                     RMSE = min(rmses)))

### table showing regularized movie + user effect model results
rmse_results %>% knitr::kable()

## Matrix factorization of genres to refine prediction ##

### split movies with multiple genres in train set and validation set
#### below code may take several minutes to run
genre_split_edx <- edx %>% separate_rows(genres, sep = "\\|")
genre_split_validation <- validation %>% separate_rows(genres, sep = "\\|")

### view genre split
head(genre_split_edx)

### add genre effect to prediction model
#### below code may take several minutes to run
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- genre_split_edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- genre_split_edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- genre_split_edx %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarise(b_g = sum(rating - b_i - b_u - mu)/(n()+1))
  
  predicted_ratings <- genre_split_validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    .$pred
  
  return(RMSE(predicted_ratings, genre_split_validation$rating))
})

### plot lambdas and RMSEs to select optimal lambda
qplot(lambdas, rmses)

### find optimal lambda
lambda <- lambdas[which.min(rmses)]
lambda

### regularized model accounting for movie + user + genre effect
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Regularized Movie + User + Genre Effect Model",
                                     RMSE = min(rmses)))

# Results ##

### final RMSE results
rmse_results %>% knitr::kable()

head(genre_split_edx)

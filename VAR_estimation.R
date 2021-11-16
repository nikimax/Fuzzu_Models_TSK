library(urca)
library(vars)
library(xts)
library(forecast)
library(gridExtra)

# load data
data <- read.csv('Some_test_data.csv', header = TRUE, sep = ",", nrows = 200)

# train test split
data_test <- xts(data[171:200,-1], order.by = as.Date(data[171:200,1], "%Y-%m-%d"))
data_train <- xts(data[0:170,-1], order.by = as.Date(data[0:170,1], "%Y-%m-%d"))

# Unin root tests
# Table for tests results
P_values <- matrix(, nrow = 3, ncol = 12)
colnames(P_values) <- colnames(data_train)
rownames(P_values) <- c('KPSS', 'DF-GLS', 'ADF')

# KPSS unit root test
for (column in colnames(data_train)){
  kpss <- ur.kpss(data_train[,column], type = "tau", lags = "short")
  if (kpss@teststat <= kpss@cval[1]){P_values[1, column] <- 0}
  else if (kpss@teststat <= kpss@cval[4]){P_values[1, column] <- 1}
  else {P_values[1, column] <- 2}}

# DF-GLS unit root test
for (column in colnames(data_train)){
  ers <- ur.ers(data_train[,column], model = 'constant', lag.max = 6)
  if (ers@teststat <= ers@cval[1]){P_values[2, column] <- 0}
  else if (ers@teststat <= ers@cval[3]){P_values[2, column] <- 1}
  else {P_values[2, column] <- 2}}

# ADF unit root test
for (column in colnames(data_train)){
  adf <- ur.df(data_train[,column], lags = 6, selectlags = "AIC", type = "drift")
  if (adf@teststat[1] <= adf@cval[1, 1]){P_values[3, column] <- 0}
  else if (adf@teststat[1] <= adf@cval[1, 3]){P_values[3, column] <- 1}
  else {P_values[3, column] <- 2}}

# number of lugs estimation for VAR
VARselect(index_ts, lag.max = 7, type = "const")

# VAR estimation
var <- VAR(index_ts, p = 5, type = "const")
summary(var)

# Forecast
var_fcst <- forecast(var, h = 30)

# tests for residuals autocorrelation
serial.test(var1, lags.pt = 10, type = "PT.asymptotic")
serial.test(var1, lags.bg = 10, type = "BG")

# normality tests for residuals
normality.test(var1, multivariate.only = FALSE)
normality.test(var2, multivariate.only = FALSE)

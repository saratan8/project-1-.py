library(copula)
library(psych)
library(VineCopula)

# Read returns of MDY and IJH
setwd("C:/Users/12451/OneDrive/桌面/BU/2023 Spring/IAQF/Data")
pair_data <- read.table("SPY_VTI_r.csv",sep = ',')

# Data cleaning
# Set index tags as col names
colnames(pair_data) <- pair_data[1,]
pair_data <- pair_data[-1,]
# get ride of unecessary columns
pair_data <- pair_data[-c(1,2)]

ijh <- pobs(pair_data[1])
mdy <- pobs(pair_data[2])
set.seed(500)

# Find the best copulas to fit in
selectCopula <- BiCopSelect(ijh,mdy)
summary(selectCopula)
# For IJH-MDY Pair, we got t-Copula
t.cop <- tCopula(dim = 2)
qtl_data <- as.matrix(cbind(ijh,mdy))
t.fit <- fitCopula(t.cop, qtl_data,method = 'ml', optim.method = 'BFGS')
rho <- coef(t.fit)[[1]]
df <- coef(t.fit)[[2]]
# Draw distribution of fitted copula
par(mar=c(1,1,1,1))
persp(tCopula(dim = 2, rho,df=df),dCopula, nticks=4, xlab="x")

# Signal:Mispricing Index
cop <- BiCopEst(ijh,mdy,family = 2)

mi <- BiCopHfunc(ijh,mdy, cop)
mi_df <- data.frame(mi)
write.csv(mi_df, "SPY_VTI_cop.csv")

# Test set signal
IJH_MDY_test <- read.table("IJH_MDY_test.csv", sep=",")
IJH_MDY_test <- IJH_MDY_test[-c(1,2)]
colnames(IJH_MDY_test) <- IJH_MDY_test[1,]
IJH_MDY_test <- IJH_MDY_test[-1,]

# Generate test-set conditional cdf 
temp_df <- rbind(pair_data, IJH_MDY_test)
test_ijh <- pobs(temp_df[1])
test_mdy <- pobs(temp_df[2])
cop <- BiCopEst(test_ijh, test_mdy,family = 2)
mi <- BiCopHfunc(test_ijh,test_mdy, cop)
mi_df <- data.frame(mi)
write.csv(mi_df, "IJH_MDY_test_cop.csv")


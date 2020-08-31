library(dplyr)
library(ggplot2)
library(tidyverse)
library(corrplot)
library(randomForest)
library(e1071)
library(Metrics)
library(pracma)
library(mice)

train <- read.csv ("train.csv", header=TRUE, sep=",", stringsAsFactors=FALSE)
test <- read.csv("test.csv", header=TRUE, sep=",", stringsAsFactors=FALSE )

SalePrice <- train$SalePrice
Id <- test$Id
all<- rbind(within(train, rm("SalePrice")), test)
all_plot<- rbind(within(train, rm("SalePrice")), test)
summary(train$SalePrice)
##房價分佈直方圖
ggplot(data=train, aes(x=SalePrice/100)) + geom_histogram (binwidth = 20 )
colSums(is.na(all))
##賣出的月份及年份
ggplot(train, aes(x=as.factor(MoSold), y=SalePrice)) + geom_bar(stat='summary', fun.y = "median") + geom_label(stat = "count", aes(label = ..count.., y = ..count..)) + labs(x="Month Sold")
ggplot(train, aes(x=as.factor(YrSold), y=SalePrice)) + geom_bar(stat='summary', fun.y = "median") + geom_label(stat = "count", aes(label = ..count.., y = ..count..)) + labs(x="Year Sold")
##製造年份
ggplot(data=train,aes(x=YearBuilt))+geom_bar(stat="count")
##Data Cleaning##
all_plot$LotFrontage[is.na(all_plot$LotFrontage)] <- 0
all_plot$MasVnrArea[is.na(all_plot$MasVnrArea)] <- 0
all_plot$BsmtFinSF1[is.na(all_plot$BsmtFinSF1)] <- 0
all_plot$BsmtFinSF2[is.na(all_plot$BsmtFinSF2)] <- 0
all_plot$BsmtUnfSF[is.na(all_plot$BsmtUnfSF)] <- 0
all_plot$TotalBsmtSF[is.na(all_plot$TotalBsmtSF)] <- 0
all_plot$BsmtFullBath[is.na(all_plot$BsmtFullBath)] <- 0
all_plot$BsmtHalfBath[is.na(all_plot$BsmtHalfBath)] <- 0
all_plot$GarageYrBlt[is.na(all_plot$GarageYrBlt)] <- 0
all_plot$GarageCars[is.na(all_plot$GarageCars)] <- 0
all_plot$GarageArea[is.na(all_plot$GarageArea)] <- 0
all_plot$KitchenQual[is.na(all_plot$KitchenQual)] <-"None"
#Typos in the dataset##
all_plot$GarageYrBlt[all_plot$GarageYrBlt==2207] <- 2007
all_plot[is.na(all_plot)] <- "None"


all_plot$ExterQual<- recode(all_plot$ExterQual,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
all_plot$ExterCond<- recode(all_plot$ExterCond,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
all_plot$BsmtQual<- recode(all_plot$BsmtQual,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
all_plot$BsmtCond<- recode(all_plot$BsmtCond,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
all_plot$BsmtExposure<- recode(all_plot$BsmtExposure,"None"=0,"No"=1,"Mn"=2,"Av"=3,"Gd"=4)
all_plot$BsmtFinType1<- recode(all_plot$BsmtFinType1,"None"=0,"Unf"=1,"LwQ"=2,"Rec"=3,"BLQ"=4,"ALQ"=5,"GLQ"=6)
all_plot$BsmtFinType2<- recode(all_plot$BsmtFinType2,"None"=0,"Unf"=1,"LwQ"=2,"Rec"=3,"BLQ"=4,"ALQ"=5,"GLQ"=6)
all_plot$HeatingQC<- recode(all_plot$HeatingQC,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
all_plot$KitchenQual<- recode(all_plot$KitchenQual,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
all_plot$Functional<- recode(all_plot$Functional,"None"=0,"Sev"=1,"Maj2"=2,"Maj1"=3,"Mod"=4,"Min2"=5,"Min1"=6,"Typ"=7)
all_plot$FireplaceQu<- recode(all_plot$FireplaceQu,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
all_plot$GarageFinish<- recode(all_plot$GarageFinish,"None"=0,"Unf"=1,"RFn"=2,"Fin"=3)
all_plot$GarageQual<- recode(all_plot$GarageQual,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
all_plot$GarageCond<- recode(all_plot$GarageCond,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
all_plot$PoolQC<- recode(all_plot$PoolQC,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
all_plot$Fence<- recode(all_plot$Fence,"None"=0,"MnWw"=1,"GdWo"=2,"MnPrv"=3,"GdPrv"=4)
all_plot$LotShape<-recode(all_plot$LotShape,"Reg"=4,"IR1"=3,"IR2"=2,"IR3"=1)
##Adding features##
all_plot$TotalSF = all_plot$TotalBsmtSF + all_plot$X1stFlrSF + all_plot$X2ndFlrSF
all_plot$AllSF = all_plot$TotalBsmtSF + all_plot$GrLivArea

train <- all_plot[1:1460,]
train$SalePrice = SalePrice
test <- all_plot[1461:2919,]

##How many numerical variables?##
num_vars <- train %>% select_if(is.numeric) %>% names()
non_num_var <- setdiff(names(train),num_vars)
df_num <- train[ ,num_vars]
df_non_num <- train[,non_num_var]

##Exploratory Visualization##

cor_num <- cor(df_num,use="pairwise.complete.obs")
cor_sorted <- as.matrix(sort(cor_num[,'SalePrice'], decreasing = TRUE))
cor_high <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.2)))
cor_num <- cor_num[cor_high, cor_high]
cex.before <- par("cex")
par(cex = 0.45)
corrplot.mixed(cor_num, diag = "l", tl.col="black", tl.pos = "lt")
par(cex = cex.before)

##data processing 把character轉為factor
chr <- all[,sapply(all,is.character)]
int <- all[,sapply(all,is.integer)]
chr[is.na(chr)] <- "None"
fac <- chr %>% lapply(as.factor) %>% as.data.frame()
all <- bind_cols(fac,int)
#利用mice函數補缺值
micemod <- all %>% mice(method='rf')
all <- complete(micemod)
#重新分成train＆test
train <- all[1:length(SalePrice),]
test<-all[(length(SalePrice)+1):nrow(all),]
#利用svm modeling
svm_model<-svm(SalePrice~., data=train, cost = 3)

svm_pred <- predict(svm_model,newdata = test)
solution <- data.frame(Id=Id,SalePrice=svm_pred)

write.csv(solution,"house_price_prediction.csv",row.names = F)

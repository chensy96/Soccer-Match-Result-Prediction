####################################
## Install packages - if necessary #
####################################

# If any of these following packages have not been installed
# uncomment and run the necessary package installations

install.packages("party")
install.packages("rpart")
install.packages("entropy")
install.packages("arules")
install.packages("devtools")
install_github("ggbiplot", "vqv")
install.packages("corrplot")
install.packages("aplpack")
install.packages("modes")
install.packages("googleVis")

install.packages("FSelector")


###################
## Load libraries #
###################

library(party)
library(rpart)
library(entropy)
library(arules)
library(devtools)
library(ggbiplot)
library(corrplot)
library(aplpack)
library(modes)
library(googleVis)

library(FSelector)

##############################################
## Load the data set from the CSV file ##
##############################################

# Load the data into a data frame

GL1<-read.csv("GL1.csv")

# Display the first few lines of the data
writeLines("\n data samples:")
print(head(GL1))

GL1$HSTR <- GL1$HST / GL1$HS

GL1$ASTR <- GL1$AST / GL1$AS

GL1$STRdf <- GL1$HSTR-GL1$ASTR

GL1$Fdf <- GL1$HF-GL1$AF

GL1$Ydf <- GL1$HY-GL1$AY

GL1$Rdf <- GL1$HR-GL1$AR

GL1$Cdf <- GL1$HC-GL1$AC


###########################################
## Normalizing and standardizing features #
###########################################


minHT <- min(GL1$HT)
maxHT <- max(GL1$HT)
GL1$norm.HT <- (GL1$HT - minHT) / (maxHT - minHT)

str(GL1)

#######################################
## Remove a feature from a data frame #
#######################################

GL1$HST <- NULL
GL1$AST <- NULL
GL1$FTHG <- NULL
GL1$FTAG <- NULL
GL1$AS <- NULL
GL1$HS <- NULL
GL1$HF <- NULL
GL1$AF <- NULL
GL1$HC <- NULL
GL1$AC <- NULL
GL1$HY <- NULL
GL1$AY <- NULL
GL1$HSTR <- NULL
GL1$ASTR <- NULL
GL1$HT <- NULL
GL1$HR <- NULL
GL1$AR <- NULL
GL1$AwayTeam <- NULL
GL1$HomeTeam <- NULL
GL1$HST. <- NULL
GL1$AST. <- NULL

# prove the feature is gone
print(str(GL1))

#GL1 <- GL1[c(2,1,3,4,5,6,7)]

######################
## Calculate Correlation #
######################

cor.test(as.numeric(GL1$FTR), GL1$norm.HT)
cor.test(as.numeric(GL1$FTR), GL1$STRdf)
cor.test(as.numeric(GL1$FTR), GL1$Fdf)
cor.test(as.numeric(GL1$FTR), GL1$Ydf)
cor.test(as.numeric(GL1$FTR), GL1$Rdf)
cor.test(as.numeric(GL1$FTR), GL1$Cdf)

plot(GL1$FTR, GL1$norm.HT, 
     main="CORRR", 
     xlab="FTR", ylab="ATT")

#stat <- lm(FTR ~ norm.HT + STRdf + Fdf + Ydf + Rdf + Cdf, data = GL1)
#summary(lm(stat))

###############################################
## Define a function to calculate Pearson's r #
## for inclusion in a scatter plot matrix     #
###############################################
panel.cor <- function(x, y, digits = 2, cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  # correlation coefficient
  r <- cor(x, y)
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste("r= ", txt, sep = "")
  text(0.5, 0.6, txt)
  
  # p-value calculation
  p <- cor.test(x, y)$p.value
  txt2 <- format(c(p, 0.123456789), digits = digits)[1]
  txt2 <- paste("p= ", txt2, sep = "")
  if(p<0.01) txt2 <- paste("p= ", "<0.01", sep = "")
  text(0.5, 0.4, txt2)
}

############################
## Correlation plots of dfs#
############################
  
corr = cor(GL1[,2:7])
  
# Plot the correlations using circles
png("GL1Correlation.png", width=900, height=500)
corrplot(corr, method="circle")
dev.off() 

##############################
## Perform feature selection #
##############################

# Use random forest to choose the most important attributes (e.g. decision tree)
# Inclding sort_group which has a 1.0 correspondence with sort_group_name
GL1.rf.scores <- random.forest.importance(FTR ~ ., GL1)

# Display the feature scores
print(GL1.rf.scores)

##Add a numeric target, using the factor values
#GL1$FTR.num <- as.numeric(GL1$FTR)
#GL1$FTR.num <- NULL

# Build decision tree
tree = rpart(FTR ~ ., GL1[,1:7])

# Get predictions
p = predict(tree, GL1[,1:7], type="c")

# Get count of correct predictions
cp = GL1$FTR == p

# get ratio of correct predictions
sum(cp) / nrow(GL1)

## Use FSelector library for feature selection ##

## Define an evaluation function  for the data using a tree   
evaluator.GL1.tree <- function(subset) {
  # Use k-fold cross validation
  k <- 5
  splits <- runif(nrow(GL1))
  results = sapply(1:k, function(i) {
    test.idx <- (splits >= (i - 1) / k) & (splits < i / k)
    train.idx <- !test.idx
    test <- GL1[test.idx, , drop=FALSE]
    train <- GL1[train.idx, , drop=FALSE]
    tree <- rpart(as.simple.formula(subset, "FTR"), train)
    error.rate = sum(test$FTR != predict(tree, test, type="c")) / nrow(test)
    return(1 - error.rate)
  })
  print(subset)
  print(mean(results))
  return(mean(results))
}

## Use forward greedy search on iris data - tree eval 
subset <- forward.search(names(GL1)[2:7], evaluator.GL1.tree)

# Obtain the selected subset of features
ft <- as.simple.formula(subset, "FTR")

# Display the selected subset of features
print(ft)
   
######################
## Calculate entropy #
######################

# Get the number of unique values for some features
# This is the cardinality for each feature
length(unique(GL1$FTR))
length(unique(GL1$HF))
length(unique(GL1$AF))
length(unique(GL1$HS))
length(unique(GL1$AS))
length(unique(GL1$HST))
length(unique(GL1$AST))
length(unique(GL1$HST.))
length(unique(GL1$AST.))
length(unique(GL1$HC))
length(unique(GL1$AC))
length(unique(GL1$HR))
length(unique(GL1$AR))


# Get the frequency of values 
FTR.table <- table(GL1$FTR)
HF.table <- table(GL1$HF)
AF.table <- table(GL1$AF)
HS.table <- table(GL1$HS)
AS.table <- table(GL1$AS)
HST.table <- table(GL1$HST)
AST.table <- table(GL1$AST)
HST..table <- table(GL1$HST.)
AST..table <- table(GL1$AST.)
HC.table <- table(GL1$HC)
AC.table <- table(GL1$AC)
HR.table <- table(GL1$HR)
AR.table <- table(GL1$AR)

# Turn the table into a data frame
cdData.cat.df <- data.frame(cdData.cat.table)

FTR.df <- data.frame(FTR.table)
HF.df <- data.frame(HF.table)
AF.df <- data.frame(AF.table)
HS.df <- data.frame(HS.table)
AS.df <- data.frame(AS.table)
HST.df <- data.frame(HST.table)
AST.df <- data.frame(AST.table)
HST..df <- data.frame(HST..table)
AST..df <- data.frame(AST..table)
HC.df <- data.frame(HC.table)
AC.df <- data.frame(AC.table)
HR.df <- data.frame(HR.table)
AR.df <- data.frame(AR.table)

# Fix the "var1" feature name to be "category" and "rec_tech"

names(FTR.df)[1] <- "full time result"
names(HF.df)[1] <- "home fouls"
names(AF.df)[1] <- "AWAY FOULS"
names(HS.df)[1] <- "HOME SHOTS"
names(AS.df)[1] <- "AWAY SHOTS"
names(HST..df)[1] <- "HOME SHOTS ACCURACY"
names(AST..df)[1] <- "AWAY SHOTS ACCURACY"
names(HC.df)[1] <- "HOME YELLOW CARDS"
names(AC.df)[1] <- "AWAY YELLOW CARDS"
names(HR.df)[1] <- "HOME RED CARDS"
names(AR.df)[1] <- "AWAY RED CARDS"

# Plot the frequencies

op <- par(mfrow=c(1,1))
png("freq.FTR", width=900, height=500)
plot(FTR.df)
dev.off()

png("freq.HF", width=900, height=500)
plot(HF.df)
dev.off()

png("freq.AF", width=900, height=500)
plot(AF.df)
dev.off()

png("freq.HS", width=900, height=500)
plot(HS.df)
dev.off()

png("freq.AS", width=900, height=500)
plot(AS.df)
dev.off()

png("freq.HST.", width=900, height=500)
plot(HST..df)
dev.off()

png("freq.AST.", width=900, height=500)
plot(AST..df)
dev.off()

png("freq.HC", width=900, height=500)
plot(HC.df)
dev.off()

png("freq.AC", width=900, height=500)
plot(AC.df)
dev.off()

png("freq.HR", width=900, height=500)
plot(HR.df)
dev.off()

png("freq.AR", width=900, height=500)
plot(AR.df)
dev.off()

par(op)


# Calculate entropy of some attributes

j <- as.factor(GL1$HomeTeam)
entropy(table(j), unit="log2")

k <- as.factor(GL1$AwayTeam)
entropy(table(k), unit="log2")
 
entropy(table(GL1$FTR), unit="log2")
entropy(table(GL1$HS), unit="log2")
entropy(table(GL1$AS), unit="log2")
entropy(table(GL1$HST.), unit="log2")
entropy(table(GL1$AST.), unit="log2")
entropy(table(GL1$HF), unit="log2")
entropy(table(GL1$AF), unit="log2")
entropy(table(GL1$HC), unit="log2")
entropy(table(GL1$AC), unit="log2")
entropy(table(GL1$AR), unit="log2")
entropy(table(GL1$HR), unit="log2")





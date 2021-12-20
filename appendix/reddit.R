# Libraries
library(RedditExtractoR)
library(radarchart)
library(tm)
library(syuzhet)

# https://cran.r-project.org/web/packages/RedditExtractoR/RedditExtractoR.pdf
# Getting Reddit Data
links <- find_thread_urls(keywords="GME",subreddit="wallstreetbets",
                          period="year")

# User network plot
content <- get_thread_content(links$url[4])
comments = content[["comments"]]

# Sentiment analysis 
com <- iconv(comments$comment, to = 'utf-8')
gme <- get_nrc_sentiment(com)

# Radar chart
x1 <- data.frame(AMC=100*colSums(amc)/sum(amc)) 
x2 <- data.frame(GameStop=100*colSums(gme)/sum(gme)) 
z <- cbind(x1, x2)
labs <- rownames(z)
chartJSRadar(z, 
             labs = labs,
             labelSize = 40,
             main = 'AMC Vs GME: Sentiment Analysis of Reddit Comments')

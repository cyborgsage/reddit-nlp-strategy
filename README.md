![wsbkju](./images/wsbkju.png)



# Wallstreetbets Subreddit Sentiment Analysis

**Author**: [Doug Mill](mailto:douglas_mill@live.com)

## Overview 

Interest in the stock market has grown in the past few years, with COVID seemingly fueling some of the risk on behavior within markets. Last year, this increased interest crossed over into many other facets of our lives. With the convergence of reddit, social media, memes, the stock market, and mainstream media, the famed GameStop and AMC short squeezes captivated financial audiences in late January of this year. Through data science and research, we can take these newfound information sources and use them to benefit the broader retail trading audience.

## Business Understanding

I am representing an independent asset management and research firm that was contacted by an international pension and sovereign wealth fund. Previously, they had only been subscribed to traditional market news. When the GME and AMC squeezes broke the news, the fund took notice. They have asked me to publish research about whether the wallstreetbets subreddit is worth mining as an information source and inquired about diversifying into meme stocks. My goal here is to fulfill their needs efficiently and practically. I took a two-pronged approach to NLP; training classification models with VADER sentiment as my target, and using NLP to find the top mentioned tickers on r/wsb. My firm has decided to make a financial product based on the most frequently mentioned stocks. This product is known as RSAH and is an equally weighted ETF based on wallstreetbets mention data and will be rebalanced monthly.

## Data Understanding

The data that was used for this project was exclusively scraped from the wallstreetbets subreddit of the social media platform Reddit. A subreddit is a subgroup of the platform dedicated to their own shared interest. In the case of wallstreetbets, that happens to be trading. When I first joined the group in 2019, directional single leg options were almost exclusively promoted. Shares were frowned upon. With the humungous movements in stocks such as GME and AMC, things changed in the subreddit forever. The popularity of the board skyrocketed, reaching mainstream appeal, and people were seeing huge gains in shares. What made these companies unique was that AMC was a company on the verge of bankruptcy. Gamestop was an outdated brick and mortar video game retailer. How did these companies explode? The answer lies partially in shorts, gamma squeezing, and the wallstreetbets subreddit. In order to investigate, I scraped 100k comments containing the keyword "AMC" and 100k comments containing the keyword "GME" from wallstreetbets. The comments were posted between 1/1/21 and 11/20/21, when the project commenced. I used the PushShift API which does a great job with collected big amounts of data relative to the Reddit API.

## Data Preparation

For data preparation, I took my files straight from the API and then cleaned them out. I kept 6 relevant columns out of 40. These columns included "body" (actual comment), "score" (peer feedback), "total awards received" (peer feedback), "created utc" (date-time), "author", and "permalink" (kept these to verify legitimacy of comments). After cleaning out the irrelevant data, I cleaned the comments themselves. I then applied the default VADER for my time series data. This gave me positive, neutral, negative, and compound scores for each comment. I shipped these dataframes off to csvs. For the classification datasets known as "amc_modeling_2" and "gme_modeling", I tuned VADER by updating the lexicon with some new words and values for added context. My full EDA can be found for both AMC and GME in the appendix folder.

![Sentiment Distribution AMC](./images/tunedAMCVader.png) ![Sentiment Distribution GME](./images/tunedGMEVader.png)

The data preprocessing included converting comments to lowercase, removing "@" mentions between users, removing links, removing stopwords, and vectorizing.

## Modeling

I modeled AMC and GME comment sentiments with classification such as naive bayes, decision tree, random forest, and xgboost. I also implemented pipelines and cross validation for each model. I tried to predict the sentiment for each comment with the VADER labeled sentiment as my target column. The data I inserted was the preprocessed comments.

To take a closer look at the sentiment data, I have taken the unupdated VADER results and plotted them in a time series. I normalized the sentiments and stock price also. I then overlayed the chart with the stock price. I want to show the AMC graph, because it illustrates how sentiment can potentially be used as an indicator and supplement to your analysis.

![AMC sent-price](./images/amc_sentprice_ma.png)

## Regression Results

Given that the XGBoost model had the highest accuracy scores on both AMC and GME, it can be concluded that it was the best model. It was able to score 75% on AMC comments and 72% on GME comments. It fared better than Naive Bayes in both models which was second due to overfitting on the random forest and decision tree.

![AMC XGB](./images/xgb_AMC.png)

![GME XGB](./images/xgb_GME.png)

## Conclusion

The model did well classifying the sentiment of comments based on the assigned target which was the sentiment registered in EDA through VADER. The alternative to using VADER labeling as my target would be manual labeling of comments in a couple of 100k entry datasets. Therefore the best "truth" in my opinion was using VADER. I did try to tune the lexicon. Given the evolving lingo as well as the sarcasm and context involved with each comment, there is no fool proof method including manual labeling. While I was reading the comments, I found myself between sentiments at times, often even considering all 3. The involvement of VADER in the bigger picture proves to be useful and reliable.
Through iterative modeling, I was able to test out Naive Bayes, decision tree, random forest, and XGBoost. Decision tree and random forest had issues with overfitting which ruled them out for me. Naive Bayes served well as a baseline model but was slightly usurped by XGBoost at 75% accuracy score on AMC and 72% accuracy score on GME.
I would say that the best use of this model would be to predict comment sentiments in the same way VADER does, but only categorically. It is very possible that this model is actually doing better than VADER, but the only way to see that would be to manually label the comments and test the accuracy of the model against VADER. Regardless, I would feel just as comfortable applying this model to predict sentiments as I would with VADER or TextBlob, which I tried out in EDA but decided was inferior. Based on the results of my random sample of 100 comments from 50/25/25 distribution, VADER had 43% accuracy against manual labels so 75% * 43% would give a worst case of 32% accuracy if labels were correct/wrong at the same rate for this model as VADER against manual labels. The upside is definitely there as random guessing would net you around the same as the model floor while the ceiling would be ~57%.
Given the context of the business problem, a financial product based on NLP designed with the stakeholder as well as the broader audience is in my opinion the best way at this time to profit from the information on wallstreetbets. This financial product is an ETF named RSAH (reddit sentiment analysis holdings). It holds the top 10 most mentioned tickers on WSB for the past 30d period. The ETF will be rebalanced monthly with proprietary weighting.

## Future Research

Future work for this project includes manually labeling the comments and then training models based upon these labels. This would be tough work as well as tedious as many comments are quite subjective and interpretable in multiple ways.
RSAH rebalancing algorithms and automation would be another area of improvement for the future. Licensing and listing would then come after the product is published.
Lastly, I would like to explore further in my time series work with NLP. I thought I was able to make good progress there. One thing I worked on was researching sentiment trends over time. I started working with stacked LSTMs to predict price as well. My intuition would be that the viable way to proceed would be to refine the sentiment analyzer before fusing it with several layers of quantitative financial data in some type of neural network.
Obviously over the course of several years, this may progress to development and implementation of low latency algorithmic trading strategies integrating machine readable indicators such as NLP.

## For More Information

See the full analysis in the Jupyter Notebook or review this Presentation.

For additional info contact Doug Mill.

## Repository Structure

```
├── appendix
├── data
├── images
├── .gitignore
├── README.md
├── Reddit_Sentiment_Analysis.ipynb
├── environment.yml
└── presentation.pdf

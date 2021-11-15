import VSO
import ContentAnalysis
import SentimentAnalysis
#import PredictImage
import TwitterScore
import RelatednessAnalysis
import CLIP
import SentimentAnalysis
from collections import namedtuple
import datetime


test = namedtuple("test",['data','content','score'])
date = datetime.date.today().strftime("%Y-%m-%d")
print(type(date))

#sentiment analysis of user's tweets
tweets =input("please input your tweet:")
tweets_content = ContentAnalysis.Content_Analysis(tweets)
tweets_score = TwitterScore.get_TwitterScore(tweets)


# sentiment analysis of user's image
image = "/home/yinghongda/DataSet/predict/adorable cat.jpg"
image_content = CLIP.get_imageContent(image)
print(image_content)
image_Score = VSO.get_SentimentScore(image_content)


#proprecess
# TweetsContent = tweets_content.split(' ')[1]
ImageContent = image_content.split('_')[1]

##the relevance analysis of user's tweets and image
relatedness = RelatednessAnalysis.get_Relatedness(tweets_content,ImageContent)
print("Tweet Content:%s;\nTweet Score:%s;\nImage Content:%s;\nImage Score:%s;\nthe relevance of image and tweet:%s"%(tweets_content,tweets_score,image_content,image_Score,relatedness))

user_score = SentimentAnalysis.sentimentScoreOfImageAndTweet(tweets_score,image_Score,relatedness)

userprofile = test(date,tweets_content,user_score)
print(userprofile)

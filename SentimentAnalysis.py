


def sentimentScoreOfImageAndTweet(tweet_score,image_score,relatedness_score):
    threshold = 0.2
    user_score = None
    if ((tweet_score > 0 and image_score < 0) or (tweet_score < 0 and image_score > 0)) and relatedness_score > 0.5:
        if (tweet_score + image_score) < 0:
            if abs(tweet_score + image_score) <= threshold:
                user_score = max(tweet_score,image_score)
            else:
                user_score = min(tweet_score,image_score)
        else:
            user_score = max(tweet_score,image_score)
    else:
        user_score = tweet_score + image_score
    return user_score







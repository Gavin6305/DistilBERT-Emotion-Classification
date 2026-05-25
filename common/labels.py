""" Label definitions """

# Actual labels (index corresponds to label number)
actual_labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
                 "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement",
                 "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
                 "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

# Taken from the GoEmotions GitHub repository (but I manually put "neutral" in ambiguous; where else would it go?)
sentiment_dict = {"positive": ["amusement", "excitement", "joy", "love", "desire", "optimism",
                               "caring", "pride", "admiration", "gratitude", "relief", "approval"],

                  "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness",
                               "grief", "disgust", "anger", "annoyance", "disapproval"],

                  "ambiguous": ["realization", "surprise", "curiosity", "confusion", "neutral"]}

ekman_mapping = {"anger": ["anger", "annoyance", "disapproval"],
                 "disgust": ["disgust"],
                 "fear": ["fear", "nervousness"],
                 "joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love",
                         "optimism", "relief", "pride", "admiration", "desire", "caring"],
                 "sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
                 "surprise": ["surprise", "realization", "confusion", "curiosity", "neutral"]}


# Produce inverse mappings for an easier time
def get_inverse(mapping):
    result = {}
    for i, key in enumerate(mapping):
        for emotion in mapping[key]:
            result[emotion] = i
    return result


# Used for preprocessing and plots
inv_sentiment_dict = get_inverse(sentiment_dict)
inv_ekman_dict = get_inverse(ekman_mapping)
import sys
sys.path.insert(1, '/ABSA/')
import ABSA
from ABSA import ae
from ABSA import asc
from ABSA.BertForSequenceLabeling import BertForSequenceLabeling
from TM import tm
import nltk
import string
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')
import argparse

class category_features(object):
    """A single set of features of category."""
    def __init__(self, name, sentiment, aspects):
        self.aspects = aspects # list of aspects_features object
        self.name = name # list of aspects' category
        self.sentiment = sentiment #list of category's sentiment

class aspect_features(object):
    """
    A single set of features of aspect. Alert: repetitive aspects in one sentence might cause error
    """
    def __init__(self, name, sentiment):
        self.name = name # list of aspect names
        self.sentiment = sentiment # list of sentiment, same order with the self.name.


class sentiment_analyzer:
    """
    sentiment analyzer object. Load model requires some time and we don't want to load model every time we want to
    analyze a text. So You can pre-load this object and for every text, call generate_sentiment to analyze.
    """
    def __init__(self):
        self.ae_model = ae.BERT() # load ae model
        self.asc_model = asc.BERT() # load asc model
        self.tm_model = tm.tm(model_dir = "glove-wiki-gigaword-50") # load tm model. This is where you can change word vectors you want TM to use
        self.category =  ["food", "service", "price", "environment"] # pre-defined category
        self.stop = set(stopwords.words('english')) # stop word dictionary

    def generate_sentiment(self, text):
        """
        generate aspect, aspect sentiments, category, category sentiment. Main gate to the program
        :param text: text to be proceed
        :return: three list corresponding to aspect, aspect sentiments, category with one dictionary of category sentiment
        """
        aspects = self.ae_model.predict(text)
        aspects_sentiment = self.asc_model.predict(text,aspects)
        aspects = self.remove_sw_punct(aspects)
        aspects_category = self.tm_model.predict(aspects, self.category)
        category_sentiment = self.generate_category_sentiment(aspects_category,aspects_sentiment)
        return aspects,aspects_sentiment,aspects_category,category_sentiment

    def remove_sw_punct(self, aspects):
        """
        some aspects might contain stop words. Remove those stop words to increase the accuracy.
        :param aspects: list - a list of aspects
        :return: list, a list of aspects
        """
        result = []
        for i,aspect in enumerate(aspects):
            temp_aspect = nltk.word_tokenize(aspect)
            # temp_aspect = [t for index, t in enumerate(temp_aspect) if t[1] not in self.None_tag]
            temp_aspect = [t for t in temp_aspect if t not in self.stop and t not in string.punctuation]
            if len(temp_aspect) != 0:
                result.append(" ".join(temp_aspect))
        return result

    def generate_category_sentiment(self, aspects_category, aspects_sentiment):
        """
        generate category sentiment. One category is positive if it contains more positive aspect than negative.
        vice versa
        :param aspects_category: list - a list of aspects category
        :param aspects_sentiment: list - a list of aspect sentiments
        :return: dict - category sentiment
        """
        category_sentiment = {}
        for val in self.category:
            category_sentiment[val] = float("-inf")

        for idx, val in enumerate(aspects_category):
            if category_sentiment[val] == float("-inf"):
                category_sentiment[val] = 0
            if aspects_sentiment[idx] == "positive":
                category_sentiment[val] += 1
            elif aspects_sentiment[idx] == "negative":
                category_sentiment[val] += -1
            else:
                category_sentiment[val] += -1

        for key in category_sentiment:
            if category_sentiment[key] > 0:
                category_sentiment[key] = "positive"
            elif category_sentiment[key] == 0:
                category_sentiment[key] = "neutral"
            elif category_sentiment[key] == float("-inf"):
                category_sentiment[key] = "unknown"
            else:
                category_sentiment[key] = "negative"
        return category_sentiment

    # def output(self, aspects, aspects_sentiment,category_sentiment):

def output(aspects,aspects_sentiment,aspects_category,category_sentiment, label_list):
    """
    generate return value: a list of category_feature object
    :param aspects: list - list of aspects
    :param aspects_sentiment: - aspect
    :param aspects_category:
    :param category_sentiment:
    :param label_list:
    :return:
    """
    category_objects = []
    for label in label_list:
        category_name = label
        sentiment = category_sentiment[category_name]
        aspects_object = []
        for idx, aspect in enumerate(aspects):
            if aspects_category[idx] == label:
                aspect_name = aspect
                aspect_sentiment = aspects_sentiment[idx]
                aspects_object.append(aspect_features(name = aspect_name, sentiment = aspect_sentiment))
        category_objects.append(category_features(name=category_name, sentiment = sentiment, aspects = aspects_object))
    return category_objects

def play(text):

    # These are just for testing
    a = "Continuously have a poor experience here. The employees ALWAYS seem put out that I'm here to order a drink. They make me feel uncomfortable if I ask for a sample. It's too bad because I love the juices and the açaí bowl. I work across the street. I am always super close and would love to go over there for my lunch breaks. Each time I give them another chance, I am thoroughly disappointed once again. Customer service is awful."
    b = "This place is the bomb! They used to be solely a cafe but they applied for and received their liquor license. It's a quaint intimate space. Feels very New York. They KNOW how to mix drinks perfectly. I was very impressed. The staff is friendly. The atmosphere is laid back but fun. This will def be our new watering hole"
    c = "It was ok. I got the lobster roll with Mac salad and seaweed salad. The seaweed salad is ok the portions are small. The Mac salad had corn in it and I just wasn't a fan.  Roll was ok. I also got an Avocado boba which tasted really bad. It was watered down and had no actual avocado flavor. The check in free drink is only for the prior owner so don't ask for it."
    d = "This is my first time try Greek food and it was great! I ordered lamb gryo with salad and French fries for $8.34 in total. There is so much lamb and I love lamb! The bread they put underneath is soft and tasty as well. Gryo is yummy and in good price, and you can get Frenchfries for 1.99 and soda for 0.99. I enjoy my Greek lunch with GryoExpress."
    e = "My wife and I enjoyed a quick tomato salad and glass of wine. The servers were attentive, restaurant was clean and laid out well. The patio out front is a spectacular Las Vegas dinner experience."
    f = "Kelly's is special for a few reasons. Not only is it a quality bar for cocktails, but food. The mac and cheese in homemade, with a top layer crust crisped with a blow torch.  I agree with Simone H. assessment - it really is a best kept secret. I like the old diner style, and the drinks and beer are both top shelf - you can't beat $4 for a cocktail. It's dark inside yes but the red booths add a certain warmth. I might have seen a DJ spinning there."
    i = "Clean healthy eating. Losing a star for 3 reasons..1. Was LOST trying to order off the menu. When your leaving the gym and your starving the last thing you want to do is struggle through all the little details of what's being put into your dish. 2. I'm on a low carb diet and the name is deceiving. I had to filter through buns, potatoes, wraps. Ended up with a salad. Ugh. 3. Overpriced BUT good quality. Doesnt have to be so spendy."
    j = "Oscar was a very polite and informative server would definitely eat here again because of his service."
    k = "The view is amazing and the food is really good and reasonable.  I enjoyed the pounded steak with potatoes.  In a place where there are a million restaurants, this was interesting."
    h = "Every bite of the tamale cakes was delicious. The ribeye steaks and Greek salad were also extremely good. The pineapple upside down cheesecake was just sweet enough. The topper, however, was our waitress, Aurelia. She was the best."
    l = "The food is amazing. Bagels are really great!!!! Friendly staff and clean. Only problem is I didn't know they don't accept cards of any kind. CASH only so we went to the back a minute walk away and payed 4$ to withdraw cash to pay our bill.  I wish I knew before. Sign on door was small writing that I thought were part of the hour schedule."
    m = "One of the most flavourful places I've been to in a while great people and curries to die for my fav vegan spot to go to at the moment!"
    if len(text) == 1: text = eval(text)


    # pre-defined category
    label_list = ["food", "service", "price", "environment"]

    #object model
    model = sentiment_analyzer()

    #Running!!
    aspects,aspects_sentiment,aspects_category,category_sentiment = model.generate_sentiment(text)

    # Return value
    category_object = output(aspects,aspects_sentiment,aspects_category,category_sentiment,label_list)

    #Output
    print(text)
    for category in category_object:
        print(category.name,": ", category.sentiment)
        for aspect in category.aspects:
            print("\t",aspect.name,": ",aspect.sentiment)

    # Here you should think about a return type. Currently, we are doing category object
    return category_object

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default='a', type=str)
    args = parser.parse_args()
    play(text=args.text)



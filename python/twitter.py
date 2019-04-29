import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

API_Key = 'O0W6UBr5qeHziEvFXWJvLP4CV'
API_Secret_Key = 'WiccbjYV7WKU8guGJuOSSh6A7FQ86vl7eaJzeCzThBF0QcB5VX'
Access_Token = '1118268830956937217-Li3CRrn76O5pNnVqmU8kdw1IATpqjD'
Access_Token_Secret = 'FhHg2WXffLeIuCvsL8WTjKquRZUAY2ycgWX9kQ7yyC6YO'

class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    # class for streaming and processing live tweets

    def on_error(self, status):
        print(status)

if __name__ == "__main__":

        listener = StdOutListener()
        auth = OAuthHandler(API_Key, API_Secret_Key)
        auth.set_access_token(Access_Token, Access_Token_Secret)

        stream = Stream(auth, listener)
        stream.filter(track=['YG', 'Vampire Weekend', 'name of the album'])








from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://stockdata:dontuseany1122@stockcluster.kffpvum.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri)
print(client.admin.command('ping'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
    
    
    
db = client['AccelixData']
collection = db['Nifty_50_15min']

query_one = collection.find_one()
print(query_one)

query_all = collection.find()
for x in query_all:
    print(x)
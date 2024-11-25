import random
standard_prime = 524287

class Custom_HashTable:
    def __init__(self, m):
        self.prime_number = 524287
        self.m = m
        self.buckets = [[] for bucket in range(m)]
        self.a = self.compute_a()

    def insert(self, word):
        index = self.compute_hash(word)
        #print(index)
        self.buckets[index].append(word)

    def printer(self):
        for i, bucket in enumerate(self.buckets):
            print(f"Bucket {i}: {bucket}")

    def compute_a(self):
        number_chosen = random.randint(1, self.prime_number - 1)
        return number_chosen

    def compute_hash(self, x):
        x_val = 0
        for char in x:
            x_val += ord(char)
        #print(x_val)
        multiplier  =  self.a * x_val
        div = multiplier%self.prime_number
        hash_value = (div) % self.m
        return hash_value

    def read_from_file(self, filename):
        try:
            with open(filename, 'r', encoding='latin1') as file:
                for line in file:
                    word = line.strip()
                    if word:
                        self.insert(word)
        except UnicodeDecodeError:
            print(f"Issue spotted")
        except FileNotFoundError:
            print("File is not there")

def main() :
  new_hash_table = Custom_HashTable(10)
  words = [
      "airplane", "fox", "apple", "banana", "grape", "orange", "pear", "kiwi", "mango", "peach",
      "cherry", "lemon", "strawberry", "blueberry", "raspberry", "pineapple", "watermelon", "melon",
      "carrot", "broccoli", "tomato", "cucumber", "onion", "lettuce", "spinach", "potato", "sweetcorn",
      "caramel", "chocolate", "vanilla", "strawberry", "coffee", "tea", "milk", "juice", "soda", "water",
      "bread", "butter", "cheese", "yogurt", "egg", "bacon", "steak", "chicken", "pasta", "pizza",
      "sandwich", "hamburger", "hotdog", "fries", "cake", "cookie", "applepie", "chickenwings", "nachos", "popcorn",
      "cabbage", "lettuce", "spinach", "garlic", "cucumber", "eggplant", "zucchini", "pumpkin", "radish", "asparagus",
      "cantaloupe", "grapefruit", "lime", "apricot", "plum", "blackberry", "pomegranate", "pear", "avocado", "fig",
      "mandarin", "tangerine", "nectarine", "papaya", "coconut", "persimmon", "passionfruit", "dragonfruit", "kiwifruit",
      "aloe", "lemongrass", "ginger", "turmeric", "chili", "hotpepper", "blackpepper", "salt", "cumin", "cardamom",
      "rosemary", "oregano", "thyme", "basil", "parsley", "cilantro", "mint", "sage", "fennel", "lavender",
      "bamboo", "wheat", "barley", "corn", "sorghum", "oats", "rice", "quinoa", "spelt", "millet",
      "lion", "elephant", "tiger", "bear", "zebra", "giraffe", "hippopotamus", "kangaroo", "koala", "panda",
      "rabbit", "squirrel", "hedgehog", "mole", "fox", "wolf", "shark", "whale", "dolphin", "octopus",
      "penguin", "sealion", "seahorse", "eagle", "owl", "hawk", "parrot", "sparrow", "crow", "dove",
      "peacock", "rooster", "hen", "chicken", "duck", "turkey", "goose", "pelican", "bat", "raven",
      "butterfly", "bee", "ant", "ladybug", "grasshopper", "caterpillar", "fly", "mosquito", "cockroach", "beetle",
      "jellyfish", "starfish", "clam", "lobster", "crab", "shrimp", "snail", "worm", "slug", "tick",
      "squid", "worm", "beetle", "moth", "dragonfly", "flycatcher", "woodpecker", "swan", "peacock", "woodcock",
      "robin", "canary", "finch", "cardinal", "bald-eagle", "vulture", "buzzard", "falcon", "kite", "hawk",
      "cobra", "python", "boa", "rattlesnake", "viper", "anaconda", "gecko", "chameleon", "iguana", "lizard",
      "frog", "toad", "newt", "salamander", "crocodile", "alligator", "geese", "peafowl", "duckling", "chick",
      "cow", "sheep", "goat", "horse", "pig", "duck", "rabbit", "llama", "alpaca", "camel",
      "ox", "buffalo", "deer", "moose", "antelope", "bison", "elk", "swan", "sparrow", "mouse",
      "rat", "bat", "stoat", "ferret", "skunk", "weasel", "raccoon", "armadillo", "opossum", "beaver",
      "otter", "mink", "walrus", "seal", "platypus", "echidna", "shrew", "hedgehog", "rat", "mongoose",
      "pigeon", "parakeet", "cockatoo", "parrotfish", "goldfish", "turtle", "salmon", "trout", "catfish", "bass",
      "carp", "walleye", "pike", "tilapia", "perch", "mackerel", "cod", "halibut", "sardine", "tuna",
      "swordfish", "angelfish", "clownfish", "lionfish", "bass", "catfish", "salmon", "trout", "grouper"
  ]

  for word in words:
      new_hash_table.insert(word)

  new_hash_table.printer()
  print("This segment  is for the file reading ")
  new_hash =Custom_HashTable(50)
  new_hash.read_from_file('words.txt')
  new_hash.printer()

main()

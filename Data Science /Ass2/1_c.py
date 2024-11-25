import hashlib
import random

def resolve_hash(word):
    sha256_hash = hashlib.sha256(word.encode())
    result = sha256_hash.hexdigest()
    return result

def slice_last_4(hash_hex):
    return hash_hex[-4:]

def convert_to_index(last_4_digits):
    return int(last_4_digits, 16)

def generate_hash(word):
    hash_hex = resolve_hash(word)
    last_4_digits = slice_last_4(hash_hex)
    result = convert_to_index(last_4_digits)
    return result

class Custom_HashTable:
    def __init__(self, m):
        self.prime_number = 524287
        self.m = m
        self.buckets = [[] for bucs in range(m)]
        self.a = self.compute_a()

    def insert(self, word_hash):
        index = self.compute_hash(word_hash)
        self.buckets[index].append(word_hash)

    def compute_a(self):
        return random.randint(1, self.prime_number - 1)

    def compute_hash(self, word_hash):
        multiplier = self.a * word_hash
        div = multiplier % self.prime_number
        hash_value = div % self.m
        return hash_value

def flajolet_martin_estimate(words, m=500000):
    max_trailing_zeroes = 0
    hash_table = Custom_HashTable(m)

    for word in words:
        word_hash = generate_hash(word)
        hash_table.insert(word_hash)
        final_hash = hash_table.compute_hash(word_hash)
        bin_rep = bin(final_hash)
        trailing_zeroes = len(bin_rep) - len(bin_rep.rstrip('0'))
        max_trailing_zeroes = max(max_trailing_zeroes, trailing_zeroes)

    z = max_trailing_zeroes
    estimated_unique_count = (2 ** (z + 1)) / 2
    return estimated_unique_count

def read_and_estimate(filename):
    words = read_file(filename)
    if words:
        estimate = flajolet_martin_estimate(words)
        print(f"Estimated number of unique words: {estimate:.2f}")

def read_file(filename):
    words = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                word = line.strip()
                if word:
                    words.append(word)
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return words

filename = "words.txt"
read_and_estimate(filename)

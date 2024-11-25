import hashlib

def resolve_md5(word):
    md5_hash = hashlib.md5(word.encode())
    result = md5_hash.hexdigest()
    #print(result)
    return result

def slice_last_4(hash_hex):
    return hash_hex[-4:]

def convert_to_index(last_4_digits):
    return int(last_4_digits, 16)

def generate_hash(word):
    hash_hex = resolve_md5(word)
    last_4_digits = slice_last_4(hash_hex)
    result =  convert_to_index(last_4_digits)
    return result

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
    #print("This is the linkd lust of the words " , words)
    return words

def printer(words , hash_index):
  print(f"Word: {words}, Hash Index: {hash_index}")

def compute(words):
    for word in words:
        hash_index = generate_hash(word)
        printer(word , hash_index)

def read_and_hash_file(filename):
    words = read_file(filename)
    if words:
        compute(words)

def main():
  print("Code starts")
  read_and_hash_file('words.txt')
  print("Code finished")

main()

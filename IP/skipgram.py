import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
import os

# Disable WANDB for this example
os.environ["WANDB_DISABLED"] = "true"


data = {
    'milk': ['almond milk', 'coconut milk', 'oat milk', 'soy milk', 'cashew milk'],
    'butter': ['coconut oil', 'ghee', 'margarine', 'olive oil', 'avocado oil'],
    'sugar': ['agave syrup', 'honey', 'maple syrup', 'stevia', 'coconut sugar'],
    'flour': ['almond flour', 'oat flour', 'rice flour', 'coconut flour', 'quinoa flour'],
    'egg': ['flaxseed meal', 'applesauce', 'chia seeds', 'banana', 'silken tofu'],
    'salt': ['Himalayan pink salt', 'kosher salt', 'table salt', 'sea salt', 'iodized salt'],
    'beef': ['tofu', 'ground chicken', 'ground turkey', 'mushrooms', 'seitan'],
    'cheese': ['nutritional yeast', 'vegan cheese', 'cashew cheese', 'tofu cheese', 'almond cheese'],
    'cream': ['cashew cream', 'almond cream', 'coconut cream', 'soy cream', 'oat cream'],
    'honey': ['agave syrup', 'molasses', 'date syrup', 'maple syrup', 'brown rice syrup'],
    'carrot': ['beetroot', 'butternut squash', 'sweet potato', 'parsnip', 'radish'],
    'potato': ['turnip', 'cauliflower', 'sweet potato', 'parsnip', 'taro'],
    'chicken': ['seitan', 'tempeh', 'tofu', 'jackfruit', 'turkey'],
    'rice': ['barley', 'cauliflower rice', 'couscous', 'quinoa', 'bulgur wheat'],
    'pasta': ['spaghetti squash', 'rice noodles', 'zucchini noodles', 'quinoa pasta', 'lentil pasta'],
    'tomato': ['carrot puree', 'pumpkin', 'red pepper', 'beetroot', 'sundried tomatoes'],
    'onion': ['leeks', 'scallions', 'chives', 'shallots', 'spring onions'],
    'garlic': ['onion powder', 'garlic powder', 'shallots', 'ginger', 'green onions'],
    'lemon': ['lime', 'vinegar', 'orange juice', 'grapefruit juice', 'tamarind paste'],
    'vinegar': ['lemon juice', 'apple cider vinegar', 'white wine', 'lime juice', 'balsamic vinegar'],
    'oil': ['avocado oil', 'grapeseed oil', 'olive oil', 'canola oil', 'sunflower oil'],
    'basil': ['cilantro', 'oregano', 'thyme', 'parsley', 'rosemary'],
    'pepper': ['chili flakes', 'black pepper', 'white pepper', 'paprika', 'cayenne pepper'],
    'chocolate': ['white chocolate', 'cocoa powder', 'carob powder', 'dark chocolate', 'chocolate chips'],
    'yogurt': ['Greek yogurt', 'coconut yogurt', 'almond yogurt', 'soy yogurt', 'cashew yogurt'],
    'bread': ['pita', 'naan', 'tortilla', 'bagel', 'focaccia'],
    'mushrooms': ['zucchini', 'tofu', 'jackfruit', 'eggplant', 'artichokes'],
    'fish': ['chicken', 'shrimp', 'tempeh', 'tofu', 'jackfruit'],
    'tea': ['green tea', 'rooibos', 'herbal tea', 'black tea', 'peppermint tea'],
    'coffee': ['decaf coffee', 'tea', 'instant coffee', 'cocoa', 'chicory root'],
    'nuts': ['sunflower seeds', 'pumpkin seeds', 'chia seeds', 'flaxseeds', 'hemp seeds'],
    'banana': ['pumpkin puree', 'applesauce', 'mashed sweet potato', 'pear puree', 'zucchini puree'],
    'cucumber': ['radish', 'lettuce', 'celery', 'zucchini', 'jicama'],
    'spinach': ['kale', 'arugula', 'Swiss chard', 'bok choy', 'collard greens'],
    'apple': ['pear', 'plum', 'peach', 'nectarine', 'apricot'],
    'peanut butter': ['almond butter', 'cashew butter', 'sunflower seed butter', 'tahini', 'hazelnut spread'],
    'orange': ['tangerine', 'grapefruit', 'mandarin', 'clementine', 'lemon'],
    'coconut': ['shredded coconut', 'coconut milk', 'coconut cream', 'coconut butter', 'almond flakes'],
    'lettuce': ['spinach', 'arugula', 'kale', 'cabbage', 'radicchio'],
    'berries': ['blueberries', 'raspberries', 'blackberries', 'strawberries', 'cranberries'],
    'tofu': ['seitan', 'tempeh', 'jackfruit', 'mushrooms', 'chickpeas'],
    'almond milk': ['soy milk', 'coconut milk', 'oat milk', 'cashew milk', 'rice milk'],
    'mayonnaise': ['Greek yogurt', 'hummus', 'avocado', 'sour cream', 'mashed beans'],
    'celery': ['fennel', 'carrots', 'zucchini', 'cucumber', 'bell pepper'],
    'bell pepper': ['poblano pepper', 'jalapeño', 'Anaheim pepper', 'habanero', 'serrano pepper'],
    'corn': ['peas', 'zucchini', 'green beans', 'chickpeas', 'carrot'],
    'turkey': ['chicken', 'seitan', 'tempeh', 'tofu', 'duck'],
    'grapes': ['blueberries', 'cherries', 'raisins', 'cranberries', 'currants'] ,
    'anchovies': ['capers', 'olives', 'fish sauce', 'soy sauce', 'miso paste'],
    'bacon': ['tempeh bacon', 'seitan strips', 'mushrooms', 'eggplant bacon', 'coconut bacon'],
    'brown sugar': ['coconut sugar', 'maple sugar', 'honey', 'molasses', 'date sugar'],
    'buttermilk': ['yogurt and water mix', 'lemon juice and milk mix', 'vinegar and milk mix', 'coconut milk with lemon juice', 'soy milk with apple cider vinegar'],
    'cream cheese': ['cashew cheese', 'tofu cream cheese', 'yogurt', 'ricotta cheese', 'almond cream cheese'],
    'sour cream': ['Greek yogurt', 'plain yogurt', 'silken tofu', 'cashew cream', 'coconut cream'],
    'chili powder': ['paprika', 'cayenne pepper', 'red pepper flakes', 'hot sauce', 'chili paste'],
    'cinnamon': ['nutmeg', 'allspice', 'cardamom', 'cloves', 'pumpkin spice'],
    'coriander': ['cilantro', 'parsley', 'fennel seeds', 'caraway seeds', 'dill'],
    'dill': ['fennel fronds', 'tarragon', 'parsley', 'cilantro', 'chervil'],
    'ginger': ['galangal', 'cardamom', 'turmeric', 'allspice', 'nutmeg'],
    'lamb': ['ground beef', 'ground turkey', 'pork', 'tofu', 'seitan'],
    'maple syrup': ['honey', 'agave syrup', 'date syrup', 'molasses', 'coconut nectar'],
    'mayonnaise': ['avocado', 'Greek yogurt', 'hummus', 'sour cream', 'mashed beans'],
    'mozzarella': ['provolone', 'Swiss cheese', 'vegan mozzarella', 'tofu slices', 'cashew mozzarella'],
    'pork': ['chicken', 'turkey', 'beef', 'tofu', 'jackfruit'],
    'pumpkin': ['butternut squash', 'sweet potato', 'carrot', 'acorn squash', 'kabocha squash'],
    'raisins': ['dried cranberries', 'currants', 'dried blueberries', 'dates', 'dried cherries'],
    'soy sauce': ['tamari', 'coconut aminos', 'fish sauce', 'miso paste', 'liquid aminos'],
    'vanilla extract': ['almond extract', 'maple syrup', 'honey', 'vanilla bean paste', 'vanilla powder'],
    'whipping cream': ['coconut cream', 'cashew cream', 'soy cream', 'oat cream', 'half-and-half'],
    'wine': ['grape juice', 'apple cider', 'red wine vinegar', 'pomegranate juice', 'cranberry juice'],
    'yeast': ['baking powder', 'baking soda and acid mix', 'sourdough starter', 'self-rising flour', 'sparkling water'] ,
    'asparagus': ['broccoli', 'green beans', 'snap peas', 'zucchini', 'cauliflower'],
    'beets': ['radishes', 'turnips', 'carrots', 'sweet potatoes', 'parsnips'],
    'blue cheese': ['gorgonzola', 'feta cheese', 'goat cheese', 'vegan blue cheese', 'roquefort'],
    'brandy': ['whiskey', 'rum', 'apple cider', 'sherry', 'cognac'],
    'broccoli': ['cauliflower', 'Brussels sprouts', 'green beans', 'kale', 'asparagus'],
    'cauliflower': ['broccoli', 'zucchini', 'cabbage', 'parsnip', 'turnip'],
    'cream of tartar': ['lemon juice', 'white vinegar', 'citric acid', 'baking powder', 'yogurt'],
    'duck': ['chicken', 'turkey', 'goose', 'pheasant', 'tofu'],
    'evaporated milk': ['coconut milk', 'heavy cream', 'half-and-half', 'almond milk', 'oat milk'],
    'fennel': ['anise', 'celery', 'caraway seeds', 'parsley', 'dill'],
    'figs': ['dates', 'prunes', 'apricots', 'raisins', 'dried cherries'],
    'goat cheese': ['feta cheese', 'ricotta cheese', 'cream cheese', 'vegan cream cheese', 'queso fresco'],
    'grapefruit': ['orange', 'lemon', 'lime', 'tangerine', 'pomelo'],
    'ground turkey': ['ground chicken', 'ground beef', 'tofu crumbles', 'tempeh', 'seitan'],
    'jalapeño': ['serrano pepper', 'poblano pepper', 'banana pepper', 'chili flakes', 'cayenne pepper'],
    'kale': ['spinach', 'Swiss chard', 'collard greens', 'arugula', 'mustard greens'],
    'molasses': ['honey', 'maple syrup', 'brown sugar', 'dark corn syrup', 'date syrup'],
    'okra': ['zucchini', 'eggplant', 'green beans', 'asparagus', 'snap peas'],
    'paprika': ['chili powder', 'cayenne pepper', 'red pepper flakes', 'smoked paprika', 'ancho chili powder'],
    'peach': ['nectarine', 'plum', 'apricot', 'mango', 'pear'],
    'peas': ['edamame', 'snap peas', 'green beans', 'asparagus', 'zucchini'],
    'pomegranate': ['cranberries', 'raspberries', 'red currants', 'cherries', 'strawberries'],
    'prawns': ['shrimp', 'scallops', 'lobster', 'crab', 'tofu'],
    'radicchio': ['endive', 'arugula', 'romaine lettuce', 'kale', 'Swiss chard'],
    'ricotta': ['cottage cheese', 'cream cheese', 'Greek yogurt', 'tofu ricotta', 'mascarpone'],
    'rum': ['brandy', 'whiskey', 'bourbon', 'coconut water', 'vanilla extract'],
    'scallops': ['shrimp', 'prawns', 'lobster', 'tofu', 'king oyster mushrooms'],
    'sesame oil': ['olive oil', 'peanut oil', 'canola oil', 'avocado oil', 'coconut oil'],
    'shallots': ['onions', 'scallions', 'chives', 'garlic', 'red onion'],
    'shrimp': ['prawns', 'scallops', 'lobster', 'tofu', 'tempeh'],
    'star anise': ['fennel seeds', 'anise seed', 'allspice', 'cloves', 'cinnamon stick'],
    'thyme': ['oregano', 'rosemary', 'parsley', 'tarragon', 'marjoram'],
    'tuna': ['salmon', 'sardines', 'mackerel', 'jackfruit', 'tofu'],
    'vanilla bean': ['vanilla extract', 'vanilla powder', 'almond extract', 'maple syrup', 'vanilla bean paste'],
    'water chestnuts': ['jicama', 'celery', 'carrot slices', 'radishes', 'zucchini'],
    'white wine': ['vermouth', 'sherry', 'chicken broth', 'apple juice', 'grape juice'],
    'zucchini': ['cucumber', 'yellow squash', 'eggplant', 'asparagus', 'broccoli'] ,
    'almonds': ['cashews', 'pecans', 'walnuts', 'hazelnuts', 'pine nuts'],
    'apricots': ['peaches', 'nectarines', 'plums', 'apples', 'mango'],
    'artichokes': ['heart of palm', 'fennel', 'broccoli', 'zucchini', 'eggplant'],
    'avocado': ['hummus', 'guacamole', 'banana', 'tofu', 'mango'],
    'bacon': ['tempeh', 'tofu', 'seitan', 'vegan bacon', 'mushrooms'],
    'balsamic vinegar': ['red wine vinegar', 'apple cider vinegar', 'lemon juice', 'white wine vinegar', 'sherry vinegar'],
    'barbecue sauce': ['ketchup', 'teriyaki sauce', 'honey mustard', 'mole sauce', 'chili sauce'],
    'basil pesto': ['spinach pesto', 'sun-dried tomato pesto', 'arugula pesto', 'kale pesto', 'cilantro pesto'],
    'beef broth': ['vegetable broth', 'chicken broth', 'mushroom broth', 'bone broth', 'tomato broth'],
    'bitter melon': ['cucumber', 'eggplant', 'zucchini', 'kale', 'dandelion greens'],
    'black beans': ['kidney beans', 'pinto beans', 'garbanzo beans', 'lentils', 'chickpeas'],
    'blueberries': ['blackberries', 'raspberries', 'strawberries', 'cranberries', 'goji berries'],
    'bok choy': ['spinach', 'kale', 'Swiss chard', 'collard greens', 'cabbage'],
    'bread crumbs': ['crushed nuts', 'oats', 'quinoa', 'rice flour', 'cornmeal'],
    'brown sugar': ['maple syrup', 'honey', 'coconut sugar', 'molasses', 'agave syrup'],
    'buttermilk': ['yogurt', 'almond milk', 'coconut milk', 'lemon juice with milk', 'vinegar with milk'],
    'cabbage': ['lettuce', 'kale', 'spinach', 'collard greens', 'Swiss chard'],
    'canned pumpkin': ['sweet potato puree', 'butternut squash puree', 'carrot puree', 'apple puree', 'banana puree'],
    'capers': ['green olives', 'pickles', 'jalapeños', 'lemon zest', 'chopped parsley'],
    'caviar': ['tofu caviar', 'seaweed', 'lentils', 'avocado', 'mashed peas'],
    'chana dal': ['yellow split peas', 'moong dal', 'lentils', 'red lentils', 'kidney beans'],
    'cherry tomatoes': ['grape tomatoes', 'Roma tomatoes', 'plum tomatoes', 'sun-dried tomatoes', 'tomato paste'],
    'chili powder': ['paprika', 'cayenne pepper', 'smoked paprika', 'red pepper flakes', 'chipotle powder'],
    'coconut flour': ['almond flour', 'rice flour', 'chickpea flour', 'oat flour', 'cornstarch'],
    'coconut water': ['almond milk', 'coconut milk', 'fruit juice', 'rice milk', 'soy milk'],
    'cream cheese': ['vegan cream cheese', 'tofu cheese', 'cashew cheese', 'Greek yogurt', 'ricotta'],
    'cucumber': ['zucchini', 'celery', 'lettuce', 'radishes', 'jicama'],
    'curry paste': ['curry powder', 'turmeric', 'garam masala', 'red chili paste', 'ginger-garlic paste'],
    'dates': ['figs', 'raisins', 'prunes', 'dried apricots', 'dried cherries'],
    'eggplant': ['zucchini', 'chayote', 'mushrooms', 'kabocha squash', 'jackfruit'],
    'ghee': ['coconut oil', 'butter', 'avocado oil', 'olive oil', 'lard'],
    'ginger': ['ground ginger', 'galangal', 'turmeric', 'garlic', 'lemon zest'],
    'gravy': ['vegetable broth with cornstarch', 'mushroom gravy', 'tomato sauce', 'soy sauce', 'coconut aminos'],
    'hazelnuts': ['pecans', 'walnuts', 'almonds', 'macadamia nuts', 'cashews'],
    'hummus': ['guacamole', 'avocado spread', 'tofu spread', 'mashed beans', 'yogurt'],
    'jackfruit': ['tofu', 'seitan', 'mushrooms', 'tempeh', 'eggplant'],
    'kale chips': ['spinach chips', 'collard greens chips', 'cabbage chips', 'sweet potato chips', 'zucchini chips'],
    'kiwi': ['strawberries', 'blackberries', 'raspberries', 'mango', 'pineapple'],
    'lemon zest': ['lime zest', 'orange zest', 'grapefruit zest', 'lemon juice', 'vinegar'],
    'lemongrass': ['lemon zest', 'ginger', 'lime leaves', 'tarragon', 'basil'],
    'lime': ['lemon', 'vinegar', 'orange juice', 'tamarind paste', 'grapefruit juice'],
    'maple syrup': ['agave syrup', 'honey', 'molasses', 'brown rice syrup', 'date syrup'],
    'mint': ['basil', 'oregano', 'rosemary', 'thyme', 'parsley'],
    'miso paste': ['tamari sauce', 'soy sauce', 'coconut aminos', 'bragg liquid aminos', 'vegan Worcestershire sauce'],
    'mozzarella': ['vegan mozzarella', 'feta', 'goat cheese', 'ricotta', 'cottage cheese'],
    'mustard': ['honey mustard', 'yellow mustard', 'Dijon mustard', 'whole grain mustard', 'mustard powder'],
    'nutmeg': ['cinnamon', 'allspice', 'cloves', 'ginger', 'cardamom'],
    'oats': ['almond meal', 'coconut flour', 'chia seeds', 'flaxseeds', 'quinoa flakes'],
    'olives': ['capers', 'pickles', 'sun-dried tomatoes', 'green beans', 'anchovies'],
    'onion powder': ['garlic powder', 'chives', 'shallots', 'leeks', 'onion flakes'],
    'orange juice': ['lemon juice', 'lime juice', 'grapefruit juice', 'pineapple juice', 'apple juice'],
    'oregano': ['thyme', 'basil', 'marjoram', 'rosemary', 'sage'],
    'palm sugar': ['brown sugar', 'coconut sugar', 'maple syrup', 'molasses', 'honey'],
    'parmesan': ['vegan parmesan', 'nutritional yeast', 'cashew cheese', 'vegan feta', 'pecorino'],
    'parsnips': ['carrots', 'turnips', 'sweet potatoes', 'rutabaga', 'daikon radish'],
    'pea protein': ['soy protein', 'rice protein', 'hemp protein', 'lentil protein', 'chickpea protein'],
    'pineapple': ['mango', 'papaya', 'orange', 'peach', 'nectarine'],
    'pita bread': ['naan', 'tortilla', 'flatbread', 'baguette', 'focaccia'],
    'potatoes': ['sweet potatoes', 'yam', 'parsnips', 'rutabaga', 'cauliflower'],
    'pumpkin': ['butternut squash', 'sweet potato', 'acorn squash', 'carrot', 'beetroot'],
    'raspberries': ['blackberries', 'strawberries', 'blueberries', 'cranberries', 'goji berries'],
    'rice noodles': ['zucchini noodles', 'rice paper', 'soba noodles', 'udon noodles', 'egg noodles'],
    'saffron': ['turmeric', 'curry powder', 'paprika', 'ginger', 'cardamom'],
    'salt': ['kosher salt', 'sea salt', 'Himalayan pink salt', 'table salt', 'soy sauce'],
    'sour cream': ['Greek yogurt', 'vegan sour cream', 'cashew cream', 'tofu cream', 'coconut cream'],
    'spinach': ['kale', 'Swiss chard', 'collard greens', 'arugula', 'bok choy'],
    'tofu': ['tempeh', 'seitan', 'chickpeas', 'jackfruit', 'mushrooms'],
    'tomato paste': ['tomato puree', 'salsa', 'diced tomatoes', 'sun-dried tomatoes', 'roasted red peppers'],
    'turmeric': ['curry powder', 'ginger', 'mustard powder', 'saffron', 'cayenne pepper'],
    'vegan butter': ['margarine', 'coconut oil', 'olive oil', 'avocado oil', 'sunflower oil'],
    'vegetable broth': ['chicken broth', 'beef broth', 'mushroom broth', 'bone broth', 'water with seasonings'],
    'walnuts': ['almonds', 'pecans', 'cashews', 'hazelnuts', 'pine nuts'],
    'watercress': ['arugula', 'spinach', 'kale', 'dandelion greens', 'mustard greens'],
    'white chocolate': ['dark chocolate', 'milk chocolate', 'cocoa nibs', 'carob', 'vegan chocolate'],
    'yogurt': ['coconut yogurt', 'almond yogurt', 'soy yogurt', 'cashew yogurt', 'vegan yogurt']

}

# Tokenizer setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to generate masked sentences
def create_masked_sentences(data):
    sentences = []
    for ingredient, alternatives in data.items():
        for alt in alternatives:
            sentence = f"Instead of {ingredient}, you can use [MASK]."
            sentences.append((sentence, ingredient))  # Store the original word for evaluation
    return sentences

# Create masked sentences
masked_sentences = create_masked_sentences(data)

# Tokenization
def tokenize_data(sentences):
    input_texts = []
    labels = []
    for sentence, original_word in sentences:
        input_tokens = tokenizer.encode(sentence, add_special_tokens=True)
        input_texts.append(input_tokens)
        label_tokens = tokenizer.encode(original_word, add_special_tokens=True)
        labels.append(label_tokens)
    return input_texts, labels

input_data, output_data = tokenize_data(masked_sentences)

class MaskedLanguageDataset(Dataset):
    def __init__(self, input_data, output_data, max_len=20):
        self.input_data = input_data
        self.output_data = output_data
        self.max_len = max_len

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_text = self.input_data[idx]
        output_text = self.output_data[idx]

        input_tokens = tokenizer.encode_plus(
            input_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt'
        )

        output_tokens = tokenizer.encode_plus(
            output_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt'
        )

        input_text_tensor = input_tokens['input_ids'].squeeze(0)
        output_text_tensor = output_tokens['input_ids'].squeeze(0)

        # Return as a dictionary
        return {
            'input_ids': input_text_tensor,
            'labels': output_text_tensor
        }

# Collate function for DataLoader
def collate_fn(batch):
    input_tensors, output_tensors = zip(*batch)
    input_tensors = torch.stack(input_tensors)
    output_tensors = torch.stack(output_tensors)
    
    return {
        'input_ids': input_tensors,
        'labels': output_tensors
    }

from torch.utils.data import DataLoader, random_split

dataset = MaskedLanguageDataset(input_data, output_data)

train_size = int(0.6 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Model setup
model = BertForMaskedLM.from_pretrained('bert-base-uncased')


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= train_dataset,
    eval_dataset= val_dataset,
)

trainer.train()

# Save the model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")

# Use the trained model for inference
def predict_ingredient(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
        predicted_token_id = predictions[0, mask_token_index].argmax(dim=-1)
        predicted_token = tokenizer.decode(predicted_token_id)
    return predicted_token

# Example prediction
sentence = "Instead of milk, you can use [MASK]."
predicted_ingredient = predict_ingredient(sentence)
print(f"Predicted ingredient: {predicted_ingredient}")

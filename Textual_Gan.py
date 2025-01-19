import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

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
    'yeast': ['baking powder', 'baking soda and acid mix', 'sourdough starter', 'self-rising flour', 'sparkling water'] ,    'asparagus': ['broccoli', 'green beans', 'snap peas', 'zucchini', 'cauliflower'],
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


categories = list(data.keys())
substitutes = [item for sublist in data.values() for item in sublist]
encoder = LabelEncoder()
substitute_encoded = encoder.fit_transform(substitutes)

class TextDataset(Dataset):
    def __init__(self, substitutes, encoder, seq_length=5):
        self.substitutes = substitutes
        self.encoder = encoder
        self.seq_length = seq_length

    def __len__(self):
        return len(self.substitutes) - self.seq_length

    def __getitem__(self, idx):
        sequence = self.substitutes[idx:idx + self.seq_length]
        sequence_encoded = self.encoder.transform(sequence)  
        return torch.tensor(sequence_encoded)


class Generator(nn.Module):
    def __init__(self, latent_dim, embedding_dim, vocab_size, seq_length):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 128)
        self.lstm = nn.LSTM(128, embedding_dim, batch_first=True)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.seq_length = seq_length

    def forward(self, z):
        x = torch.relu(self.fc(z))  # Feed noise through a fully connected layer
        x = x.unsqueeze(1).repeat(1, self.seq_length, 1)  # Repeat for the sequence length
        x, _ = self.lstm(x)  # LSTM layer
        x = self.fc_out(x)  # Output layer to match vocab size
        return x


class Discriminator(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(Discriminator, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embeddings(x)  # Get word embeddings
        x, _ = self.lstm(x)  # LSTM layer
        x = x.mean(dim=1)  # Global average pooling (to reduce sequence length)
        x = self.fc_out(x)  # Final output layer
        return torch.sigmoid(x)  # Output probability of real or fake

# Step 4: Train the GAN

# Set parameters
latent_dim = 100  # Latent space dimensionality
embedding_dim = 50  # Embedding size for each word
vocab_size = len(encoder.classes_)  # Vocabulary size (number of unique substitutes)
seq_length = 5  # Length of sequences to generate
num_epochs = 1000  # Number of epochs for training

# Create dataset and dataloader
dataset = TextDataset(substitutes, encoder, seq_length)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize models
generator = Generator(latent_dim, embedding_dim, vocab_size, seq_length)
discriminator = Discriminator(embedding_dim, vocab_size)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# Loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    for real_data in dataloader:
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        optimizer_d.zero_grad()

        # Real data
        outputs = discriminator(real_data)
        loss_d_real = criterion(outputs, real_labels)
        loss_d_real.backward()

        # Generate fake data
        z = torch.randn(batch_size, latent_dim)  # Latent space noise
        fake_data = generator(z).argmax(dim=2)  # Generate fake sequences (word indices)

        # Fake data
        outputs = discriminator(fake_data.detach())
        loss_d_fake = criterion(outputs, fake_labels)
        loss_d_fake.backward()

        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()

        # Want discriminator to think fake data is real
        outputs = discriminator(fake_data)
        loss_g = criterion(outputs, real_labels)
        loss_g.backward()

        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {loss_d_real.item() + loss_d_fake.item()}, Loss G: {loss_g.item()}")

# Step 5: Generate new substitutes
z = torch.randn(1, latent_dim)  # Sample random noise
generated_data = generator(z)
generated_substitute = generated_data.argmax(dim=2)  # Get the word with the highest probability
generated_substitute_text = encoder.inverse_transform(generated_substitute.squeeze().cpu().numpy())
print(f"Generated Substitute: {generated_substitute_text}")

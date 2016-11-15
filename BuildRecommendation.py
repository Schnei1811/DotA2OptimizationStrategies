import numpy as np
import tensorflow as tf
import operator
import heapq

def datacreation(input_data):
	input_data[selectedhero-1] = 1
	input_data[radhero1+112] = 1
	input_data[radhero2+112] = 1
	input_data[radhero3+112] = 1
	input_data[radhero4+112] = 1
	input_data[radhero5+112] = 1
	input_data[direhero1+225] = 1
	input_data[direhero2+225] = 1
	input_data[direhero3+225] = 1
	input_data[direhero4+225] = 1
	input_data[direhero5+225] = 1	
	return input_data

def deep_neural_network_model(data, hidden_size,n_classes,n_features):
	n_nodes_hl1 = int(hidden_size)
	n_nodes_hl2 = int(hidden_size)

	hidden_1_layer = {'f_fum':n_nodes_hl1,
					  'weight':tf.Variable(tf.random_normal([n_features, n_nodes_hl1])),
					  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'f_fum':n_nodes_hl2,
					  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	output_layer = {'f_fum':None,
					'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
					'bias':tf.Variable(tf.random_normal([n_classes])),}

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
	l1 = tf.nn.relu(l1)
	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
	l2 = tf.nn.relu(l2)
	output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']
	return output

def PredictDeepNeuralNetwork(input_data):
	x = tf.placeholder('float', [None, n_features])
	prediction = deep_neural_network_model(x,hidden_size,n_classes,n_features)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		saver.restore(sess,"SavedParameters/DNNmodel.ckpt")
		result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[input_data]}),1)))
		Probabilities = prediction.eval(feed_dict={x: [input_data]},session=sess)
		print(result)

	Probabilities = Probabilities[0]
	Probabilities = Probabilities[1:]
	index, value = max(enumerate(Probabilities), key=operator.itemgetter(1))
	print(Item[index+1])
	x = np.where(Probabilities>=heapq.nlargest(6,Probabilities.flatten())[-1])
	x = x[0]
	return x

radhero1 = 2
radhero2 = 39
radhero3 = 53
radhero4 = 13
radhero5 = 43
direhero1 = 52
direhero2 = 112
direhero3 = 14
direhero4 = 105
direhero5 = 10
selectedhero = 2

Hero = {1:'Antimage',2:'Axe',3:'Bane',4:'Bloodseeker',5:'Crystal Maiden',6:'Drow Ranger',7:'Earthshaker',8:'Juggernaut',9:'Mirana',10:'Morphling',
		11:'Shadow Fiend',12:'Phantom Lancer',13:'Puck',14:'Pudge',15:'Razor',16:'Sand King',17:'Storm Spirit',18:'Sven',19:'Tiny',20:'Vengeful Spirit',
		21:'WindRanger',22:'Zeus',23:'Kunkka',24:'Blank',25:'Lina',26:'Lion',27:'Shadow Shaman',28:'Slardar',29:'Tidehunter',30:'Witch Doctor',
		31:'Lich',32:'Riki',33:'Enigma',34:'Tinker',35:'Sniper',36:'Necrophos',37:'Warlock',38:'Beastmaster',39:'Queen of Pain',40:'Venomancer',
		41:'Faceless Void',42:'Wraith King',43:'Death Prophet',44:'Phantom Assassin',45:'Pugna',46:'Templar Assassin',47:'Viper',48:'Luna',49:'Dragon Knight',50:'Dazzle',
		51:'Clockwerk',52:'Leshrac',53:'Natures Prophet',54:'Lifestealer',55:'Dark Seer',56:'Clinkz',57:'Omniknight',58:'Enchantress',59:'Huskar',60:'Night Stalker',
		61:'Brood Mother',62:'Bounty Hunter',63:'Weaver',64:'Jakiro',65:'Batrider',66:'Chen',67:'Spectre',68:'Ancient Apparition',69:'Doom',70:'Antimage',
		71:'Spirit Breaker',72:'Gyrocopter',73:'Alchemist',74:'Invoker',75:'Silencer',76:'Outworld Devourer',77:'Lycan',78:'BrewMaster',79:'Shadow Demon',80:'Lone Druid',
		81:'Chaos Knight',82:'Meepo',83:'Treant',84:'Ogre Magi',85:'Undying',86:'Rubick',87:'Disruptor',88:'Nyx Assassin',89:'Naga Siren',90:'Keeper of the Light',
		91:'Wisp',92:'Visage',93:'Slark',94:'Medusa',95:'Troll Warlord',96:'Centaur Warrunner',97:'Magnus',98:'Timbersaw',99:'Bristleback',100:'Tusk',
		101:'Skywrath Mage',102:'Abaddon',103:'Elder Titan',104:'Legion Commander',105:'Techies',106:'Ember Spirit',107:'Earth Spirit',108:'Abyssal Underlord',109:'TerrorBlade',110:'Pheonix',
		111:'Oracle',112:'Winter Wyvern',113:'Arc Warden'}

Item = {1:'Blink',2:'Blades of Attack',3:'Broadsword',4:'Chainmail',5:'Claymore',6:'Helm of Iron Will',7:'Javelin',8:'Mithril Hammer',9:'Platemail',10:'Quarter Staff',
		11:'Quelling Blade',12:'Ring of Protection',13:'Gauntlets',14:'Slippers',15:'Mantle',16:'Branch',17:'Belt of Strength',18:'Boots of Elves?',19:'Robe',20:'Circlet',
		21:'Ogre Axe',22:'Blade of Alacrity',23:'Staff of Wizardry',24:'Ultimate Orb',25:'Gloves',26:'LifeSteal?',27:'Ring of Regen',28:'Sobi Mask?',29:'Boots',30:'Gem',
		31:'Cloak',32:'Talisman of Evasion',33:'Cheese',34:'Magic Stick',35:'Magic Wand',36:'Ghost Scepter',37:'Clarity',38:'Healing Flask',39:'Dust',40:'Bottle',
		41:'Observer Ward',42:'Sentry Ward',43:'Tango',44:'Courier',45:'TP Scroll',46:'Boots of Travel',47:'Phase Boots',48:'Demon Edge',49:'Eagle Song',50:'Reaver',
		51:'Sacred Relic',52:'Hyperstone',53:'Ring of Health',54:'Void Stone',55:'Mystic Staff',56:'Energy Booster',57:'Point Booster',58:'Vitality Booster',59:'Power Treads',60:'Hand of Midas',
		61:'Oblivion Staff',62:'Pers?',63:'Poor Mans Shield',64:'Bracer',65:'Wraith Band',66:'Null Talisman',67:'Mekanism',68:'Vladmirs Offering',69:'Flying Courier',70:'Buckler',
		71:'Ring of Basiluis',72:'Pipe',73:'Urn of Shadows',74:'Headdress',75:'Scythe of Vyse',76:'Orchid',77:'Cyclone?',78:'Force Staff',79:'Dagon',80:'Neconomicon',
		81:'Ultimate Scepter?',82:'Refresher Orb',83:'Assault Cuirass',84:'Heart',85:'Black King Bar',86:'Shivas Guard',87:'Bloodstone',88:'Sphere?',89:'Vanguard',90:'Blade Mail',
		91:'Soul Booster',92:'Hood of Defiance',93:'Devine Rapier',94:'Monkey King Bar',95:'Radiance',96:'Butterfly',97:'Greater Crit?',98:'Basher',99:'Battlefury',100:'Manta',
		101:'Lesser Crit?',102:'Arlmet of Mordiggeon',103:'Invis Sword?',104:'Sange and Yasha',105:'Satanic',106:'Mjollnir',107:'Skaadi',108:'Sange',109:'Helm of the Dominator',110:'Maelstrom',
		111:'Desolater',112:'Yasha',113:'Mask of Madness',114:'Diffusal Blade',115:'Ethereal Blade',116:'Soul Ring',117:'Arcane Boots',118:'Orb of Venom',119:'Stout Shield',120:'Anicent Djaggo?',
		121:'Medallion of Courage',122:'Smoke of Deceit',123:'Veil of Discord',124:'Necronomicon 2',125:'Necronomicon 3',126:'Diffusal Blade 2',127:'Dagon 2',128:'Dagon 3',129:'Dagon 4',130:'Dagon 5',
		131:'Rod of Atos',132:'Abyssal Blade',133:'Heavens Halberd',134:'Ring of Aquila',135:'Tranquil Boots',136:'Shadow Amulet',137:'Enchanted Mango',138:'Ward Dispenser?',139:'Boots of Travel 2',140:'Lotus Orb',
		141:'Solar Crest',142:'Octarine Core',143:'Guardian Greaves',144:'Aether Lens',145:'Dragon Lance',146:'Faerie Fire',147:'Iron Talon',148:'Blight Stone',149:'Tango(Shared)',150:'Crimson Guard',
		151:'Wind Lace',152:'Bloodthorn',153:'Moon Shard',154:'Silver Edge',155:'Echo Sabre',156:'Glimmer Cape',157:'Tome of Knowledge',158:'Hurriance Pike',159:'Infused Raindrop'}

Predict = {1:'Radiant Victory',2:'Dire Victory'}

n_features = 339-1
n_classes = 159+1
hidden_size = 1000

print("\nRadiant Team:\n",Hero[radhero1],",",Hero[radhero2],",",Hero[radhero3],",",Hero[radhero4],",",Hero[radhero5])
print("\nDire Team:\n",Hero[direhero1],",",Hero[direhero2],",",Hero[direhero3],",",Hero[direhero4],",",Hero[direhero5])

print("\nSelected Hero:\n",Hero[selectedhero])

input_data = np.zeros((n_features,),dtype=np.int)
input_data = datacreation(input_data)
dnnpredict = PredictDeepNeuralNetwork(input_data)

print(dnnpredict)

print("\nItem Recommendations:\n",Item[dnnpredict[0]+1],Item[dnnpredict[1]+1],Item[dnnpredict[2]+1],Item[dnnpredict[3]+1],Item[dnnpredict[4]+1],Item[dnnpredict[5]+1])
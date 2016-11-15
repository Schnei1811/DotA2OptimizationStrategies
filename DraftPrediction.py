import numpy as np
import pandas as pd
import pickle

def datacreation(input_data):
	input_data[radhero1-1] = 1
	input_data[radhero2-1] = 1
	input_data[radhero3-1] = 1
	input_data[radhero4-1] = 1
	input_data[radhero5-1] = 1
	input_data[direhero1+112] = 1
	input_data[direhero2+112] = 1
	input_data[direhero3+112] = 1
	input_data[direhero4+112] = 1
	input_data[direhero5+112] = 1	
	return input_data

def PredictRandomForest(input_data):
	input_data = input_data.reshape(1,-1)
	rfclf = pd.read_pickle('SavedParameters/RFpickle.pickle')
	rfpredict = int(rfclf.predict(input_data))
	return rfpredict

def PredictSimpleNeuralNetwork():
	X = np.array([input_data])
	fmin = pd.read_pickle('SavedParameters/SNNpickle.pickle')
	theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (num_features + 1)], (hidden_size, (num_features + 1))))
	theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (num_features + 1):], (num_classes, (hidden_size + 1))))
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
	snnPredict = np.array(np.argmax(h, axis=1))
	snnPredict = snnPredict[0]
	return snnPredict

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

radhero1 = 3
radhero2 = 9
radhero3 = 32
radhero4 = 63
radhero5 = 43
direhero1 = 73
direhero2 = 113
direhero3 = 22
direhero4 = 67
direhero5 = 87

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

Predict = {1:'Radiant Victory',2:'Dire Victory'}

num_features = 227
num_classes = 2
hidden_size = 1000

print("\nRadiant Team:\n",Hero[radhero1],",",Hero[radhero2],",",Hero[radhero3],",",Hero[radhero4],",",Hero[radhero5])
print("\nDire Team:\n",Hero[direhero1],",",Hero[direhero2],",",Hero[direhero3],",",Hero[direhero4],",",Hero[direhero5])

input_data = np.zeros((num_features-1,),dtype=np.int)
input_data = datacreation(input_data)
rfpredict = PredictRandomForest(input_data)

input_data = np.zeros((num_features,),dtype=np.int)
input_data = datacreation(input_data)
snnpredict = PredictSimpleNeuralNetwork()

print("\nRandom Forest Predicted", Predict[rfpredict])
print("\nSimple Neural Network Predicted", Predict[int(snnpredict)])
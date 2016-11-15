import numpy as np
import pandas as pd
import pickle

print("Enter 10 Hero Numbers (1-113):")
hero1 = int(input("Hero 1 "))
hero2 = int(input("Hero 2 "))
hero3 = int(input("Hero 3 "))
hero4 = int(input("Hero 4 "))
hero5 = int(input("Hero 5 "))
hero6 = int(input("Hero 6 "))
hero7 = int(input("Hero 7 "))
hero8 = int(input("Hero 8 "))
hero9 = int(input("Hero 9 "))
hero10 = int(input("Hero 10 "))
heroselected = int(input("Recommend Build for Which Hero?" ))

#hero1 = 11
#hero2 = 12
#hero3 = 13
#hero4 = 14
#hero5 = 15
#hero6 = 16
#hero7 = 17
#hero8 = 18
#hero9 = 19
#hero10 = 20
#heroselected = 11

def datacreation(input_data, heroid, radiant,dire):
	input_data[heroselected+heroid] = 1
	input_data[hero1+radiant] = 1
	input_data[hero2+radiant] = 1
	input_data[hero3+radiant] = 1
	input_data[hero4+radiant] = 1
	input_data[hero5+radiant] = 1
	input_data[hero6+dire] = 1
	input_data[hero7+dire] = 1
	input_data[hero8+dire] = 1
	input_data[hero9+dire] = 1
	input_data[hero10+dire] = 1	
	return input_data

def insertzeros(input_data):
	input_data = np.insert(input_data,0,0)
	input_data = np.insert(input_data,0,0)
	input_data = np.insert(input_data,0,0)
	input_data = np.insert(input_data,0,0)
	input_data = np.insert(input_data,0,0)
	return input_data

def checkrules(dataname,rfpredict,numskill):
	prevprob = np.load("modelparameters/%s/rfprobscores.npy" %dataname)
	highestprob = str(np.unravel_index(np.argmax(prevprob), prevprob.shape))
	highestprob = int(str(highestprob[1]))+1
	if highestprob == rfpredict:
		if numskill['skill5'] < maxskill['max5']:
			rfpredict = 5
		elif numskill['skill2'] < maxskill['max2']:
			rfpredict = 2
		elif numskill['skill3'] < maxskill['max3']:
			rfpredict = 3
		elif numskill['skill4'] < maxskill['max4']:
			rfpredict = 4
		else:
			rfpredict = 1
	else:
		rfpredict = highestprob
	return rfpredict

def skillpredict(i,input_data,numskill,maxskill):
	input_data = np.array([input_data])
	dataname = "lvl%s" %i
	rfclf = pd.read_pickle('modelparameters/%s/RFpickle.pickle'%dataname)
	rfpredict = int(rfclf.predict(input_data))

	if rfpredict == 5 and numskill['skill5'] >= maxskill['max5'] or rfpredict ==5 and numskill['skill5'] == 1 and 6<i<11 or rfpredict == 5 and numskill['skill5'] == 2 and i<16:
		rfpredict = checkrules(dataname,rfpredict,numskill)
	elif rfpredict == 4 and numskill['skill4'] >= maxskill['max4']:
		rfpredict = checkrules(dataname,rfpredict,numskill)
	elif rfpredict == 3 and numskill['skill3'] >= maxskill['max3']:
		rfpredict = checkrules(dataname,rfpredict,numskill)
	elif rfpredict == 2 and numskill['skill2'] >= maxskill['max2']:
		rfpredict = checkrules(dataname,rfpredict,numskill)
	elif rfpredict == 1 and numskill['skill1'] >= maxskill['max1']:
		rfpredict = checkrules(dataname,rfpredict,numskill)

	numskill['skill%s'%rfpredict] = numskill['skill%s'%rfpredict] + 1
	return rfpredict

lvls = 26
i=1
heroid = 0
num_features = 339
radiant = 113
dire = 226
maxskill = {'max1':1,'max2':1,'max3':1,'max4':1,'max5':0}
numskill = {'skill1':0,'skill2':0,'skill3':0,'skill4':0,'skill5':0}
input_data = np.zeros((num_features,),dtype=np.int)
input_data = datacreation(input_data,heroid,radiant,dire)

Skills = {'RfPred1':0,'RfPred2':0,'RfPred3':0,'RfPred4':0,'RfPred5':0,'RfPred6':0,'RfPred7':0,'RfPred8':0,
			'RfPred9':0,'RfPred10':0,'RfPred11':0,'RfPred12':0,'RfPred13':0,'RfPred14':0,'RfPred15':0,'RfPred16':0,
			'RfPred17':0,'RfPred18':0,'RfPred19':0,'RfPred20':0,'RfPred21':0,'RfPred22':0,'RfPred23':0,'RfPred24':0,'RfPred25':0}

Skills["RfPred%s"%i] = skillpredict(i,input_data,numskill,maxskill)

while i < lvls:	
	if i == 3:
		maxskill['max2'] = maxskill['max2']+ 1
		maxskill['max3'] = maxskill['max3']+ 1
		maxskill['max4'] = maxskill['max4']+ 1
	if i == 5:
		maxskill['max2'] = maxskill['max2']+ 1
		maxskill['max3'] = maxskill['max3']+ 1
		maxskill['max4'] = maxskill['max4']+ 1
	if i == 6:
		maxskill['max5'] = maxskill['max5']+ 1
	if i == 7:
		maxskill['max2'] = maxskill['max2']+ 1
		maxskill['max3'] = maxskill['max3']+ 1
		maxskill['max4'] = maxskill['max4']+ 1
	if i == 11:
		maxskill['max5'] = maxskill['max5']+ 1
	if i == 16:
		maxskill['max5'] = maxskill['max5']+ 1

	if i > 1:
		input_data = insertzeros(input_data)
		input_data[int(Skills["RfPred%s"%str(i-1)])-1] = 1
		Skills["RfPred%s"%i] = skillpredict(i,input_data,numskill,maxskill)
	heroid = heroid +5
	num_features = num_features+5
	radiant = radiant+5
	dire = dire + 5
	i = i+1

print("\nRecommended Build for ",heroselected)
print(Skills['RfPred1'],Skills['RfPred2'],Skills['RfPred3'],Skills['RfPred4'],Skills['RfPred5'],Skills['RfPred6'],Skills['RfPred7'],
	Skills['RfPred8'],Skills['RfPred9'],Skills['RfPred10'],Skills['RfPred11'],Skills['RfPred12'],Skills['RfPred13'],Skills['RfPred14'],
	Skills['RfPred15'],Skills['RfPred16'],Skills['RfPred17'],Skills['RfPred18'],Skills['RfPred19'],Skills['RfPred20'],Skills['RfPred21'],
	Skills['RfPred22'],Skills['RfPred23'],Skills['RfPred24'],Skills['RfPred25'])
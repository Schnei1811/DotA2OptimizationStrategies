# Dota2 Optimization Strategies

Included here are three optimization strategies based on data scaped from DotaBuff.com using the Python module Beautiful Soup.

Winner Prediction uses a Random Forest and Simple Neural Network to predict the winner of a match based solely on the draft. Both machine learning algoirthms are trained on a onehot representation of the Radiant and Dire team composition. The simple neural network can predict the match outcome with 72% accuracy. The user can input the team composition and use the trained model parameters to predict any games outcome.

Skill Recommender uses a Random Forest to predict the order in which a specific hero should level their skills based on the hero composition of both teams. Using build orders scraped from winning games on DotaBuff.com the program uses Random Forest to predict what skill a hero should choose for each level based on a data set of onehot team compositions. Included are hard coded conditions to ensure it does not recommend skills that are not allowed within the rules of the game (ie. more than 4 of a skill).

Build Recommendation uses a MultiLayer Perceptron to predict the 6 optimal end game items for a specific hero based on the draft and trained on high level winning games from DotaBuff.com. Considering team compositions in a onehot format, the deep neural network is trained on the 159 possible items purchaseable in Dota 2. Rather than returning a discrete classification, the deep neural networks expresses each query as a confidence for each of the 159 items. The top 6 items are recommended for the player.

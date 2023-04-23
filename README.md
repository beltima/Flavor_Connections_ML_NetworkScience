# Flavor_Connections_ML_NetworkScience

Final Project for Machine Learning For Network Science

M.Sc. in Data Sciences and Business Analytics 

CentraleSupélec 

Co-Author: 
- [@Aiza]
- [@Alinehbg] (https://github.com/Alinehbg)
- [@EmmaHongW] (https://github.com/EmmaHongW)

## ABSTRACT
In this project, we propose a machine learning-based approach to explore the similarity between the ingredients of different cuisines and determine their pairing of flavors. We construct ingredient networks for each cuisine by representing ingredients as nodes and co-occurrences as edges. We employ machine learning’s algorithm to model the relationships between the ingredients, and use it to generate feature importance scores for each ingredient. By analyzing these scores, we identify the key flavor components of each cuisine, and compare them to find similarities between different cuisines. We then use this information to recommend dishes based on their similarity in terms of flavor pairing. Our approach is effective in identifying flavor similarities between cuisines, and can be used to recommend dishes that are likely to be enjoyed by individuals based on their culinary preferences. This approach also has the potential to enhance the culinary experience for individuals by providing tailored dish recommendations based on their flavor preferences.

## Methodology and results
Our project is based on data from Yong-Yeol Ahn's work [1], including 1530 food ingredients, flavor compounds and 36,781 edges. We also have a database of 7,000 recipes.

### Preprocessing
The first step was to disaggregate the recipe database of 7,000 elements to have a list of ingredients. This step involves breaking down each recipe in the database into a list of its constituent ingredients. For example, if a recipe calls for "pasta with tomato sauce and basil", this step would break it down into three ingredients: pasta, tomato sauce, and basil. We homogenize the ingredient list by performing a lemmatization, a regex and splitting the words. We also removed words that bring low information such as ‘low fat’, ‘gluten free’, etc. 

Then we matched ingredients from the recipes to ingredients in a graph, and filtered out those with more than three missing matches. We made sure that the functions chosen also catch some spelling differences and make appropriate substitutions. Because recipes had a high variability, we standardized the ingredient names and made them consistent across the entire dataset. For example, the words : 'rib','chuck','sirloin' and 'steak' were replaced with ‘beef’. Finally, we choose to perform a function that makes additional substitutions for certain ingredients that were not initially matched correctly. Using the compound database, we matched each ingredient to its corresponding flavor and removed the ingredients that do not have a match in the flavor dataset. 

We ended up with a dataset with columns for each ingredient, as well as a dataset projected into the ingredient space in the flavor network. Some recipes had to be discarded as their ingredients are not listed in the flavor network.
We perform weight scaling of ingredients depending on the recipe in which they are. This step involves assigning weights to each ingredient based on its quantity in the recipe. For example, if a recipe calls for 2 cups of pasta and 1 cup of tomato sauce, pasta would have a weight of 2 and tomato sauce would have a weight of 1. For this step, we used the TF-IDF "Term Frequency-Inverse Document Frequency" method.  The TF-IDF method is used to normalize the flavor matrix to adjust for the relative importance of each compound in each recipe.

### Backbone
In order to create the nodes and edges, we calculated all the combinations possible between the different ingredients. 
The nodes are the top 200 ingredients. 

The edges were created by counting the number of recipes where the ingredient lives. The ingredients with the highest degrees were the ones present in the highest number of recipes - here onions are in 18 205 recipes, followed by garlic with 17 465 recipes.

The importance of each edge was given by the count of the pair (ingredient 1, ingredient 2): for example, the pair (almond, apricot) was counted 7 times which is less than the pairs (almond, anis) counted 12 times.

The one-sided ratio for each ingredient pair is determined by dividing the number of co-occurrences by the frequency of the source node.

Then the other types of weights are calculated using the ingredient-pair frequency-inverse recipe frequency (IF-IRF) method. Firstly, the recipe counts for each ingredient in the ingredient pairs are retrieved from the graph information. Then, the combined recipe count for each ingredient pair is calculated by summing the recipe counts of the two ingredients. The ingredient-pair frequency (IF) is computed as [1 + log10(1 + IF)) * log10(total recipes / recipe frequency]. The inverse recipe frequency (IRF) is obtained by taking the logarithm of the total number of recipes divided by the combined recipe count. The IF and IRF values are then multiplied to obtain the IF-IRF weight. Additionally, a log (base e) of IF-IRF is also calculated by replacing the log (base 10) with natural logarithm in the IF and IRF formulas.

The IF-IRF weight calculated with log (base e) is then used to extract the backbone graph. The threshold is set as 60% of the maximum weight, the edge of the pair which has a weight below the threshold will be removed, resulting in the extraction of a backbone graph that retains only edges with weights above the threshold.

The backbone graph is visualized as follows:

<!-- ![Figure 1. Backbone of the ingredient network](/images/Backbone%20of%20the%20ingredient%20network.png) -->
<p align="center">
  <img src="/images/Backbone%20of%20the%20ingredient%20network.png" alt="Figure 1. Backbone of the ingredient network">
</p>


Each node denotes an ingredient. Two ingredients are connected if they share a significant number of flavor compounds. Here, in this graph, the full network is too dense to be informative. We still used the full network in the rest of our work.

### Cuisine Classification
We classified the different cuisines based on the ingredient matrix and on the flavor matrix. 

#### Cuisine classification with ingredients
We used ingredient information as features. We apply the following classifiers in order  to classify the cuisines:
-	Logistic Regression
-	Support Vector Machine
-	RandomForest
-	Multinomial Naive Bayes
-	XGBoost

Table 1. Evaluation of the performance of the classifiers

|Classifier |	Accuracy|
|-----------|---------|
|Logistic Regression |	69.3%|
|Support Vector Machine |	70.1%|
|RandomForest |	68.4%|
|Multinomial Naive Bayes |	63.8%|
|XGBoost	| 68.8%|

We also performed hyperparameter tuning in order to prevent overfitting and achieve the highest performance possible.
SVM is the classifier that allows the best classification of the recipes. Using ingredient information as features, we were able to classify recipes into regional cuisines with 71% accuracy.

#### Cuisine classification with flavor 
Then, we used the flavors as features. We apply the following classifiers in order  to classify the cuisines:
-	Logistic Regression
-	Support Vector Machine

Table 2. Evaluation of the performance of the classifiers

|Classifier |	Accuracy|
|-----------|---------|
|Logistic Regression | 66.2%|
|Support Vector Machine	| 65.2%|


Using flavor profile as features, our classification accuracy using logistic regression is only 66%, suggesting more overlap in the flavor space.
### Cuisine clustering
We performed t-SNE clustering and plotted on a dataframe of recipes based on their regional cuisine. t-SNE is commonly used for visualizing high-dimensional data in a lower-dimensional space (usually 2D or 3D). We choose this algorithm because it is a non-linear dimensionality reduction technique that aims to preserve the local structure of the high-dimensional data points in the lower-dimensional space. 

We first separated the cuisines into two groups: cuisines that are alike and cuisines that are different. This first ‘manual’ clustering was performed in order to better visualize the clusters - as we have a high number of cuisines. This grouping was done based on their cultural background. For example, the cuisines that we assumed were alike are cuisines from North America (Mexican, Cuban, Hawaiian, etc).

#### Cuisines that are alike
In this group, we find the following cuisines: 'Southern & Soul Food, American', 'American', 'Southwestern, American, Mexican', 'Cajun & Creole, American', 'Southwestern, American', 'Cajun & Creole, Southern & Soul Food, American', 'Hawaiian, American', 'Chinese, Asian'.  

We performed three measure distances: Jaccard, cosine and hamming. Jaccard and cosine have both poor performances: they struggled to differentiate the regional cuisines of North America - which may mean they share similarities.   

The hamming distance performed better as we could visualize more defined clusters. 
#### Cuisines that are different
In this group we find the following cuisines : 'Indian, Asian', 'Mexican', 'Greek', 'Cajun & Creole, American', 'Chinese, Asian', 'Italian' and 'Irish'.
##### Clustering on ingredients
In this group we find the following cuisines : 'Indian, Asian', 'Mexican', 'Greek', 'Cajun & Creole, American', 'Chinese, Asian', 'Italian' and 'Irish'. Such as the section before, we performed the three same distance measures. 
With these new cuisines, we can better visualize the differentiation between each cuisine. However, it appears to be difficult to properly cluster the Irish cuisine (in dark orange) from the others. This could be explained by the fact that the Irish cuisine has an influence on other american cuisines - as the Irish immigrated to America in the 19th century. 

The hamming distance is struggling more with Greek and Italian cuisines (beige and red). It is also difficult to distinguish between Chinese, Indian and Mexican cuisines.  
##### Clustering based on flavors
To perform clustering based on the flavor compounds, we used the following distance measure (as the three methods used before were not enough to distinguish between cuisines): hamming, Jaccard, cosine, euclidean, sqeuclidean and Minkowski. 

While it is beautiful to look at (especially for the Jaccard distance), the clustering based on flavor does not allow to distinguish between cuisines.

We can suggest that the flavors can not be clustered to a specific cuisine since there are ingredients shared through them all, like tomatoes and garlic. 
### Dish recommendation
Based on the ingredients present in a recipe, we built a function that returns the closest recipes based on the cuisine clustering. The closest recipe is calculated using the cosine similarity. As an example, if the user chooses as an input a Mexican dish, the function can return either a similar Mexican dish or a Cuban one. The user chooses if he prefers a similar dish from the same original cuisine or not. 

Using the example given in the graph above, the closest recipe from Chipotle Kiwi Tequila Carne Asada is the crock pot cuban shredded pork (Cuban cuisine) with a similarity of 0.84 followed by another cuban dish, the Churrasco Skirt Steak with a similarity of 0.835. 
## Evaluation
Regarding the classification of the cuisines, we plotted the confusion matrix in order to better visualize the performance of our classification model. The rows represent the true labels and the columns represent the predicted labels. The diagonal of the confusion matrix shows the number of correct classifications, while the off-diagonal elements indicate the misclassifications. 

We managed to see some similarities, for example, the Mediterranean cuisine and the Greek cuisine appear to share the same ingredients and thus ‘confuse’ our classification model. 

When analyzing the dish recommendation, we may face some strangeness. In the previous example, a similar dish of the Chipotle Kiwi Tequila Carne Asada is Sunset Margarita. The user needs to keep in mind that his dish recommendation is ingredient- based, and both recipes contain Tequila, orange juice, and lime juice. It is worth noting that the output of our dish recommendation should be treated cautiously - for obvious reasons. A solution that can be implemented is to add weights to ingredients that are more relevant than others. We performed a weight scaling based on the quantity but on the relevance.  For example onions, garlic, olive oil should have lower weights since they are present in most recipes. 

The flavor network does not provide information on the concentration of each compound - a compound is considered as present above a certain threshold. The concentration of each compound plays an important role in taste and flavor and thus in dish recommendation.  
## Conclusions
The proposed machine learning-based approach is effective in identifying and recommending dishes based on the similarity of flavors between cuisines, and has the potential to enhance the culinary experience for individuals by providing tailored dish recommendations based on their flavor preferences. However, one must keep in mind that human taste perception is complex, influenced by other sensory experiences such as smell and temperature.  

## References
[1] Ahn, YY., Ahnert, S., Bagrow, J. et al. (2011) Flavor network and the principles of food pairing. Sci Rep 1, 196. https://doi.org/10.1038/srep00196

[2] Teng, C.-Y., Lin, Y.-R., & Adamic, L. A. (2012). Recipe recommendation using ingredient networks. In Proceedings of the 4th Annual ACM Web Science Conference (pp. 298-307). https://doi.org/10.1145/2380718.2380757

[3] Jain, A., N K, R., & Bagler, G. (2015). Analysis of Food Pairing in Regional Cuisines of India. PloS one, 10(10), e0139539. https://doi.org/10.1371/journal.pone.0139539

[4] Arellano-Covarrubias, A., Gómez-Corona, C., Varela, P., & Escalona-Buendía, H. B. (2019). Connecting flavors in social media: A cross cultural study with beer pairing. Food research international (Ottawa, Ont.), 115, 303–310. https://doi.org/10.1016/j.foodres.2018.12.004

[5] Varshney, K. R., Varshney, L. R., Wang, J., & Myers, D. (2013). Flavor pairing in medieval European cuisine: A study in cooking with dirty data. arXiv preprint arXiv:1307.7982. https://doi.org/10.48550/arXiv.1307.7982 

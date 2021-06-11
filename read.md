Relevant Product Recommendation search Engine
	
Objective :This application is developed with the intention to maximize sales through e-commerce website of Home Depot.
Architecture & Mechanism:
It is designed with the  combination of Natural Language processing and Kmeans Clustering (unsupervised technique).
The data consist of  Home depot Products Id, Product Title and Product Description. In different csv files.
Using Term frequency and Inverse Document frequency and word net lemmatization   (TFIDF) technique ,product description column was cleaned and preprocessed .Then based on majority of feature counts clusters groups  were made using Kmeans clustering by keeping relevant products in same cluster .
The search engine predicts the cluster group based on user entry and displays cluster which act like a filter.
By selecting the item in the cluster will display top 10 trending products related to it..
Eg. Let say user types bulb as search object , then the system will predict cluster group corresponding to bulb which contains bulb as well as other items related to it. 
And Then by selecting the item in cluster will display   top 10 trending products corresponding to it.
Public Application Link:
https://share.streamlit.io/siddheshm14/product_recommendation/main/relevantsearch.py
 

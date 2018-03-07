# Capstone: Baltimore City Crime Prediction

# Introduction

<p style='text-align: justify;'>The objective of the project is to predict as accurately as possible, using different machine learning algorithms, whether a crime, particularly a serious one, will take place given certain spatial and temporal parameters.</p>

<p style='text-align: justify;'>In light of the impracticality of stationing policemen and the cost and sensitivity of setting up surveillance cameras on every street, an accurate prediction will allow us to allocate more efficiently, based on the severity of the crime, the crime fighting resources available to us.</p>

<p style='text-align: justify;'>Just to illustrate the point, if there were only 5 streets in a state and the state has 2 policemen and 3 cameras available for deployment to prevent burglaries and homicide, it would be more beneficial to have the cameras set up on streets where burglaries are more likely to occur and the policemen patrol where homicide is more likely to occur. This is because the damage cause by theft can be remedied almost completely should the culprit be apprehended. The same cannot be said of homicide as its consequences are irreversible and the urgency of prevention far outweigh that of any burglary.</p>

<p style='text-align: justify;'>For the purposes of this project, we will be using crime data of Baltimore City in the United States, Maryland and exploring different machine learning models in an attempt to build a model that is able to predict, with reasonable likelihood, the of occurrence of each type of crime.</p>

## Acknowledgements

<p style='text-align: justify;'>The crime data of Baltimore City in the United States of America, Maryland for the years 2012 to 2017 (<strong>‘Raw Dataset’</strong>) was obtained from Kaggle at <a href="https://www.kaggle.com/sohier/crime-in-baltimore/data">this</a> weblink.</p> 

<p style='text-align: justify;'>All acknowledgements and disclaimers given in the link above in relation to the Raw Dataset similarly apply.</p> 

# The Dataset

## Metadata

<p style='text-align: justify;'>The Raw Dataset comprises 276529 records of crime from 2012 to 2017 and 15 different variables. The table below sets out the metadata of the Raw Dataset:</p>

| Variable Name        | Description           | Datatype  |
| :------------------- |:----------------------|:----------|
| CrimeDate|Date of occurrence of the crime.| Object|
| CrimeTime|Time of occurrence of the crime|Object|
| CrimeCode|The crime code reflects which subgroup a particular crime belongs within each crime type.| Object   |
| Location| The street at which the crime took place.| Object    |
| Description|The type of crime that took place. There is a total of 15 crime types.| Object     |
| Inside/Outside| Whether the crime took place in the open or under some form of shelter.| Object|
| Weapon|The type of weapon that was used in the carrying out of the crime.| Object    |
| Post| - | Float64     |
| District| The district in which the crime took place.| Object|
| Neighborhood| The neighbourhood in which the crime took place.| Object     |
| Longitude| The longitudinal coordinates of the place of occurrence of the crime.| Float64     |
| Latitude|The latitudinal coordinates of the place of occurrence of the crime. | Float64   |
| Location 1| A combination of the longitude and latitude of the occurrence of the crime.| Object|
| Premise| The setting in which the crime took place.| Object|
| Total Incidents| The number of incidents of crime that took place.| Int64|

# At A Glance

## Null Values

| Variable Name        | Null Values           | 
| :------------------- |:----------------------|
| CrimeDate|0|
| CrimeTime|0|
| CrimeCode|0|
| Location|2207|
| Description|0|
| Inside/Outside|10279|
| Weapon|180952|
| Post|224|
| District|80|
| Neighborhood|2740|
| Longitude|2204|
| Latitude|2204|
| Location 1|2204|
| Premise| 10757|
| Total Incidents| 0|

## First 2 Rows

<div style="overflow-x:auto;">
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>CrimeDate</th>
        <th>CrimeTime</th>
        <th>CrimeCode</th>
        <th>Location</th>
        <th>Description</th>
        <th>Inside/Outside</th>
        <th>Weapon</th>
        <th>Post</th>
        <th>District</th>
        <th>Neighborhood</th>
        <th>Longitude</th>
        <th>Latitude</th>
        <th>Location 1</th>
        <th>Premise</th>
        <th>Total Incidents</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>09/02/2017</td>
        <td>23:30:00</td>
        <td>3JK</td>
        <td>4200 AUDREY AVE</td>
        <td>ROBBERY - RESIDENCE</td>
        <td>I</td>
        <td>KNIFE</td>
        <td>913.0</td>
        <td>SOUTHERN</td>
        <td>Brooklyn</td>
        <td>-76.60541</td>
        <td>39.22951</td>
        <td>(39.2295100000, -76.6054100000)</td>
        <td>ROW/TOWNHO</td>
        <td>1</td>
      </tr>
      <tr>
        <th>1</th>
        <td>09/02/2017</td>
        <td>23:00:00</td>
        <td>7A</td>
        <td>800 NEWINGTON AVE</td>
        <td>AUTO THEFT</td>
        <td>O</td>
        <td>NaN</td>
        <td>133.0</td>
        <td>CENTRAL</td>
        <td>Reservoir Hill</td>
        <td>-76.63217</td>
        <td>39.31360</td>
        <td>(39.3136000000, -76.6321700000)</td>
        <td>STREET</td>
        <td>1</td>
      </tr>
    </tbody>
  </table>
</div>

# EDA

## Dropping of Rows and Columns

| Item        | Action           |
| :------------------- |:----------------------|
| Location 1|The information contained in this column is repeated from the 'Latitude' and 'Longitude' columns. As it did not provide any more information than we already  had, this column was similarly dropped.|
| Total Incidents|Because all values in this column are 1, it does not provide any meaningful information with which we can work with. The column was therefore dropped.|
| Post|It could not be ascertained what the values in this column represented. The column was therefore dropped.|
| Other Null Values|Since the remaining null values represent only a small proportion of the dataset, all rows containing null values were dropped.|

## Data Cleaning

### Weapon

<p style='text-align: justify;'>More than half of this column were null values. These columns were therefore filled with 'NONE' and an assumption made that these were crimes where no weapons were used to cause physical injury.</p>

### Inside/Outside

<p style='text-align: justify;'>This column originally contained 4 types of values, namely 'Inside', 'Outside','I' and 'O'.</p>

<p style='text-align: justify;'>Upon a visual cross check against values in the 'Premise' column, it was concluded that 'I', as with 'Inside', indicated that the crime took place indoors while 'O' and 'Outside' indicated that the crime took place outdoors.</p>

<p style='text-align: justify;'>The column values were therefore converted to '0' for 'Outside' and 'O' and '1' for 'Inside' and 'I'.</p>

## Re-categorising Crime Type

### Original Categories

<p style='text-align: justify;'>There are 15 different crime types set out in the 'Description' column of the dataframe. The bar plot below is a plot of the value counts of each class of crime.</p>

![barplot](https://zhongiekka.github.io/img/crime_pd_value_counts.png)

<p style='text-align: justify;'>On top of the multiclass issue, as can bee seen from the above visualisation, the dataset is also imbalanced in that some crimes occur a lot more frequently than others (e.g. there are 58246 incidents of larceny compared to 1299 incidents of murder and 1548 incidents of rape).</p>

### Re-categorization

<p style='text-align: justify;'>One way of handling this will be to combine classes of crimes that are similar in nature. This will help to reduce both the imbalance and the number of classes within the dependent variable. The combinations are as follow:</p>

#### Class 0: COMMON ASSAULT, ASSAULT BY THREAT

<p style='text-align: justify;'>These 2 can be classed together as they are all classified under the umbrella crime of assault. The crime of assault actually includes battery as well, so verbal threats and physical contact are both captured by the same law.</p>

#### Class 1: LARCENY, LARCENY FROM AUTO

<p style='text-align: justify;'>Larceny and Larceny From Auto can be classed together as they relate to theft of personal belongings.</p>

#### Class 2: AUTO THEFT, ROBBERY - CARJACKING

<p style='text-align: justify;'>Autotheft and Robbery - Carjacking can be classed together as they relate to theft of cars.</p>

#### Class 3: ROBBERY - COMMERCIAL, RESIDENCE, STREET
       
<p style='text-align: justify;'>These 3 can be classed together since they relate to robbery. Car robbery has been classed with auto theft instead of other robberies as there have been enough instances of theft of cars for it to be a class of its own.</p>

#### Class 4: BURGLARY

<p style='text-align: justify;'>There are sufficient instances of burglary for it to be a class of its own. There are also no other classes of crimes in the dataset which are of a similar nature.</p>

#### Class 5: AGG. ASSAULT, HOMICIDE, ARSON, RAPE, SHOOTING

<p style='text-align: justify;'>These 5 crimes are classed together even though they are not similar in nature since the severity of these crimes are extremely high. As opposed to the other crimes above, victims of crimes these crimes are unlikely to be able to be restored to the position they were in before the crime.</p>

#### Bar Plot of Recategorised Crimes 

![barplot](https://zhongiekka.github.io/img/crime_pd_value_counts_grouped.png)

<p style='text-align: justify;'>The exact value counts of the re-categorized crimes are set out below for reference:</p>

| Crime Type        | Value Count           |
| :------------------- |:----------------------|
| 0|92773|
| 1|26965|
| 2|46387|
| 3|23305|
| 4|40947|
| 5|32741|

## Feature Engineering

### Distance From Closest Police Station

<p style='text-align: justify;'>Based on a Google search, there are 19 police stations in Baltimore City. The coordinates of each of these police stations were extracted from Google Maps and the distance between each crime and police station was calculated in kilometers with the shortest distance being extracted and included in the dataframe as a new column labelled 'Closest Police Station(km)'.</p>

### Others

<p style='text-align: justify;'>Each of the month, day and time of occurrence of crimes were also extracted from the Raw Dataset. With regard to time of occurrence of the crimes, each day has been split into 8 parts of 3 hours each such that instead of having a continuous variable ranging from 00:00 to 23:59, the time of occurrence of each crime is a categorical value ranging from 1 to 8.</p>

## Relationships

<p style='text-align: justify;'>The different graphs below have been plotted to assist in visualising different aspects of the data so as to flag out any patterns or relationships.</p>

### Crime Occurrence by Month

![barplot](https://zhongiekka.github.io/img/crime_pd_month_vs_description_grouped.png)

### Crime Occurrence by District

![barplot](https://zhongiekka.github.io/img/crime_pd_district_vs_description_grouped.png)

### Heatmap

![heatmap](https://zhongiekka.github.io/img/crime_pd_heatmap.png)

<p style='text-align: justify;'>Based on the heatmap, it can be seen that most of the features show little to no correlation. The features that are correlated are 'District' and 'Latitude', where there is relatively strong negative correlation of -0.58. All other variables have a Pearson correlation value of under 0.2.</p>

### Radviz

![RadvizOriginal](https://zhongiekka.github.io/img/crime_pd_radviz_original.png)

<p style='text-align: justify;'>The independent variables form an even unit circle and every datapoint is linked to each independent variable by a spring. The stiffness of each spring for every datapoint is represented by the value of that particular variable for that datapoint.</p>

<p style='text-align: justify;'>As can be seen from the plot above, there is no obvious 2-dimensional clustering between different crimes. There are also many overlaps.</p>

## Geographical Plots

### Background

<p style='text-align: justify;'>The Raw Dataset provided around 260 difference neighborhoods. Based on <a href="https://data.baltimorecity.gov/Neighborhoods/Neighborhoods-Shape/ysi8-7icr">neighborhood data</a> provided by the City of Baltimore a recent map of the neighborhoods of Baltimore City is set out below:</p>

![neighborhoods](https://zhongiekka.github.io/img/geopandas_neighborhoods.png)

<p style='text-align: justify;'>Hoever, some neighborhoods may be too small if they were to be considered meaningfully. An alternative way that the city is partitioned is by different <a href="https://data.baltimorecity.gov/Neighborhoods/Community-Statistical-Areas-Shape/uga5-5yms">community statistical areas</a>, which are larger than neighborhoods. In the map below, community statistical areas (outlined in blue) are larger than the different neighborhoods (outlined in red).</p>

![neighborhoods](https://zhongiekka.github.io/img/geopandas_nhoods_overlay_csa.png)

<p style='text-align: justify;'>We will try to map the occurrence of some of the crimes to give a better idea of how spread out they are.</p>

### Geographical Mapping of Crimes

<p style='text-align: justify;'>The below is a map of all crimes in the dataset. This is not a meaningful visualisation as there are too many overlaps. We shall plot some of the crimes individually just to see if things are clearer.</p>

![neighborhoods](https://zhongiekka.github.io/img/geopandas_all_crimes.png)

#### Larceny

<p style='text-align: justify;'>Let's take a look at larceny, being the most frequently occurring crime in the dataset. When compared with the plots of homicide and rape further below, it can be seen that there are no particular clusters where each crime takes place. The main difference is the density of occurrence. This makes it more difficult for each crime to be accurately classified purely based on geographical factors as there may be many overlaps. </p>

![neighborhoods](https://zhongiekka.github.io/img/geopandas_larceny.png)

### Homicide

![neighborhoods](https://zhongiekka.github.io/img/geopandas_homicide.png)

### Rape

![neighborhoods](https://zhongiekka.github.io/img/geopandas_rape.png)

## One Hot Encoding

### Categorical Variables

<p style='text-align: justify;'>Other than columns reflecting GPS coordinates and the 'Closest Police Station(km)' column, all other columns have categorical values.</p>

<p style='text-align: justify;'>In order for training models to be built effectively, all variables with categorical values in the dataframe have to be dummified or the algorithms may read them as continuous values. One hot encoding will create new columns for every class of each variable. The values of these new columns will be binary.</p>

<p style='text-align: justify;'>This wil result in the dataframe having 32 columns (include the 'Description' column even though it has been represented by the newly allocated crime codes).</p>

<p style='text-align: justify;'>The radviz plot of the relationship between each dummified variable is shown below:</p>

![Radviz](https://zhongiekka.github.io/img/crime_pd_radviz.png)

# Preparation for Prediction

## Independent and Dependent Variables

### Dependent Variable

<p style='text-align: justify;'>Since the objective of the project is to predict whether a particular crime will take place, the dependent variable which we are aiming to predict will be the crime type, in this case represented by the crime code.</p>

### Independent Variables

Everything else except for 'Description'.

## Training Data, Test Data and Holdout Data

### Holdout Set

<p style='text-align: justify;'>25% of the entire processed dataset (the <strong>'Main Dataset'</strong>) will be set aside to be used as a holdout set (the <strong>'Holdout Set'</strong>) to ensure that there is no leakage of information in the training models and to prevent any overfitting.</p>

### Train-Test Split

<p style='text-align: justify;'>Of the remaining 75% of the Main Dataset, 70% will be used to train each model and 30% will be used to test if those models perform well.</p>

<p style='text-align: justify;'>All splitting will be stratified to ensure that each class is correctly represented.</p>

# Planning

## Blueprint Moving Forward

<p style='text-align: justify;'>As mentioned above, two problems presented by the dataset are (1) the number of classes of crimes (being 15 classes originally); and (2) the imbalance in occurrences between the classes of crimes.</p>

<p style='text-align: justify;'>To tackle these 2 issues, the crimes have been regrouped, with crimes of similar nature being grouped together and serious crimes being grouped together, such that only 6 classes of crimes remain. While this served to reduce the number of classes and the imbalance between classes, it did not eradicate the issues.</p>

<p style='text-align: justify;'>Further, grouping all serious crimes together may not bring us closer to our objective of allocating crime fighting resources more efficiently since, amongst the crimes deemed serious, different approaches may be adopted to combat each crime (e.g. the number of occurrences of rape and aggravated assault may be reduced by increasing police patrol in an area but fire engines should also be on standby if the crime in question is arson). The attempt to tackle this issue was conducted in 2 stages.</p>

### Stage 1

<p style='text-align: justify;'>The model trained in this stage seeks to classify whether a crime belongs into any of the 6 classes of crimes in the Main Dataset, whether serious or not, based on the independent variables.</p>

<p style='text-align: justify;'>Some machine learning models have inbuilt hyperparameter options that allow the algorithm to handle imbalance while others do not. In assessing which algorithm works best, each algorithm capable of handling imbalance will be trained with 2 sets of data, namely, the Main Dataset and the resampled dataset, where the minority classes are randomly oversampled while the majority classes are randomly undersampled to achieve a balanced dataset.</p>

<p style='text-align: justify;'>Models that are not capable of handling imbalanced data will be trained solely on the resampled data.</p>

<p style='text-align: justify;'>The results between all trained models will be compared and the best model will be used to classify crime data from a Holdout Set. Crimes classified as serious will then be run through the second model.</p>

### Stage 2

<p style='text-align: justify;'>Any crime predicted as serious (class 5) from Stage 1 will then be run through the second model, which aims to classify which of the 4 classes of serious crimes the crime actually is.</p>

<p style='text-align: justify;'>It is recognised that there may be misclassification of serious crimes in the first model such that some crimes are wrongly classified as serious when they are not. To cater for such misclassification, the second model will not merely be trained to recognise serious crimes. A proportionate number of non-serious crimes will be included in the training model to help the machine classify crimes more accurately.</p>

<p style='text-align: justify;'>As with the model building procedure in Stage 1, random sampling will be conducted on the training data and the models will be trained with balanced and/or imbalanced data. The results will similarly be compared and the best performing model chosen to classify data in the Holdout Set.</p>

<p style='text-align: justify;'>Practical applications: If the state has enough resources, it should investigate all crimes classified as serious by the first model. However, it may not be practical to make such an assumption. The second model will therefore help to narrow down the crimes where more resources should be allocated to in relation to prevention.</p>

# Predictive Models

## Models used

### Logistic Regression

<p style='text-align: justify;'>Logistic regression is a statistical method that calculates the probability of an outcome based on one or more independent variables. In cases of binary classification, the sigmoid function is applied the softmax function is applied for multiclass classification. For more information on the sigmoid and softmax functions, click <a href="http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/">here</a>. </p>

#### Multilinear

<p style='text-align: justify;'>When multi_class = multilinear is selected, the softmax function will be used to calculate the probability that a instance of crime falls within each of the 6 classes of crimes and returns the class that has the highest probability.</p>

<p style='text-align: justify;'>It should be noted that 'class_weights' is another hyperparameter that can be tuned in sklearn's logistic regression. According to the documentation, if the mode 'balanced' is selected, weights inversely proportional to class frequencies of the dependent variables will be applied to each class respectively. </p>

<p style='text-align: justify;'>Where the data is imbalanced and no resampling has been conducted, class_weights will be set to 'balanced'.</p>
    
#### One versus Rest

<p style='text-align: justify;'>When multi_class = ovr is selected, the model compares, in a binary fashion, each class against every other class collectively. In essence, it makes 6 different binary classifications and again, returns the class that has the highest probability.</p>

<p style='text-align: justify;'>class_weight is similarly set to 'balanced' where the dataset is imbalanced.</p>

### Random Forest Classifier

<p style='text-align: justify;'>A random forest classifier is an ensemble classifier that generates a number of decision tree classifiers and aggregates the votes of each tree before returning the final classification.</p>

<p style='text-align: justify;'>In brief, a decision tree classifier divides the entire set of data into smaller and smaller groups, with the whole dataset being the root and each division being referred to as a node. The final node is termed as a leaf. </p>

<p style='text-align: justify;'>If a division results in less impurity in each node as compared to the root or previous node, the decision tree classifier will make that division. The division is iterated until no further division can be made or certain hyperparameter conditions are fulfilled. </p>

<p style='text-align: justify;'>The underlying concept behind each division is referred to as information gain. For more information, click <a href="https://medium.com/machine-learning-101/chapter-3-decision-trees-theory-e7398adac567">here</a>.</p>

<p style='text-align: justify;'>It should be noted that the algorithm used by decision trees are 'greedy' in nature and will they will make a division based on the most information gained in the next node without considering a different permutation that may lead to higher information gain overall.</p>

### AdaBoost Classifier

<p style='text-align: justify;'>Similar to Random Forest Classifier, the AdaBoost Classifier is an ensemble classsifier that combines several weak classifier algorithms (in the current case the base classifiers are decision trees). </p>

<p style='text-align: justify;'>With each iteration of the base classifier, weights are attached to the training data with larger weights being attached to misclassified datapoints from the previous iteration of the classifier. This increases the chances that the misclassified datapoints will be chosen to train the next iteration of the classifier.</p>

<p style='text-align: justify;'>This is process is repeated for the selected number of estimators and the weights are adjusted with each iteration. Note that weights are attached to each iteration of the classifier as well based on the accuracy of that iteration. </p>

<p style='text-align: justify;'>In terms of predictions, the independent variables are passed through the model and a vote is taken by the estimators on how the dependent variable should be classified.</p>

<p style='text-align: justify;'>For more information on the AdaBoost Classifier, click <a href="https://medium.com/machine-learning-101/https-medium-com-savanpatel-chapter-6-adaboost-classifier-b945f330af06">here</a>.</p>

## Model 1 Training Results

### Before Any Resampling of Majority and Minority Classes
<div style="overflow-x:auto;">
  <div>
    <div>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th colspan="3"><strong>Avg/Total</strong></th>
            <th colspan="3"><strong>Class 0</strong></th>
            <th colspan="3"><strong>Class 1</strong></th>
            <th colspan="3"><strong>Class 2</strong></th>
            <th colspan="3"><strong>Class 3</strong></th>
            <th colspan="3"><strong>Class 4</strong></th>
            <th colspan="3"><strong>Class 5</strong></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td></td>
            <td><strong>Precision</strong></td>
            <td><strong>Recall</strong></td>
            <td><strong>f1</strong></td>
            <td><strong>Precision</strong></td>
            <td><strong>Recall</strong></td>
            <td><strong>f1</strong></td>
            <td><strong>Precision</strong></td>
            <td><strong>Recall</strong></td>
            <td><strong>f1</strong></td>
            <td><strong>Precision</strong></td>
            <td><strong>Recall</strong></td>
            <td><strong>f1</strong></td>
            <td><strong>Precision</strong></td>
            <td><strong>Recall</strong></td>
            <td><strong>f1</strong></td>
            <td><strong>Precision</strong></td>
            <td><strong>Recall</strong></td>
            <td><strong>f1</strong></td>
            <td><strong>Precision</strong></td>
            <td><strong>Recall</strong></td>
            <td><strong>f1</strong></td>
          </tr>
          <tr>
            <td><strong>Logistic Regression (multilinear)</strong></td>
            <td>0.67</td>
            <td>0.58</td>
            <td>0.57</td>
            <td>0.73</td>
            <td>0.25</td>
            <td>0.37</td>
            <td>0.32</td>
            <td>0.74</td>
            <td>0.44</td>
            <td>0.93</td>
            <td>0.93</td>
            <td>0.93</td>
            <td>0.54</td>
            <td>0.41</td>
            <td>0.47</td>
            <td>0.48</td>
            <td>0.92</td>
            <td>0.63</td>
            <td>0.80</td>
            <td>0.62</td>
            <td>0.70</td>
          </tr>
          <tr>
            <td><strong>Logistic Regression (ovr)</strong></td>
            <td>0.68</td>
            <td>0.57</td>
            <td>0.54</td>
            <td>0.75</td>
            <td>0.18</td>
            <td>0.29</td>
            <td>0.31</td>
            <td>0.77</td>
            <td>0.44</td>
            <td>0.93</td>
            <td>0.93</td>
            <td>0.93</td>
            <td>0.52</td>
            <td>0.42</td>
            <td>0.47</td>
            <td>0.46</td>
            <td>0.95</td>
            <td>0.62</td>
            <td>0.80</td>
            <td>0.59</td>
            <td>0.68</td>
          </tr>
          <tr>
            <td><strong>Random Forest Classifier</strong></td>
            <td>0.70</td>
            <td>0.70</td>
            <td>0.70</td>
            <td>0.71</td>
            <td>0.66</td>
            <td>0.68</td>
            <td>0.39</td>
            <td>0.43</td>
            <td>0.41</td>
            <td>0.92</td>
            <td>0.93</td>
            <td>0.93</td>
            <td>0.64</td>
            <td>0.40</td>
            <td>0.49</td>
            <td>0.62</td>
            <td>0.84</td>
            <td>0.71</td>
            <td>0.79</td>
            <td>0.74</td>
            <td>0.76</td>
          </tr>
          <tr>
            <td><strong>AdaBoost Classifier</strong></td>
            <td>0.69</td>
            <td>0.70</td>
            <td>0.69</td>
            <td>0.64</td>
            <td>0.81</td>
            <td>0.72</td>
            <td>0.42</td>
            <td>0.19</td>
            <td>0.26</td>
            <td>0.93</td>
            <td>0.93</td>
            <td>0.93</td>
            <td>0.64</td>
            <td>0.42</td>
            <td>0.51</td>
            <td>0.65</td>
            <td>0.67</td>
            <td>0.66</td>
            <td>0.80</td>
            <td>0.72</td>
            <td>0.76</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</div>       

## Handling Imbalance Via Random Sampling

<p style='text-align: justify;'>To handle the imbalance, a new dataset has been created from resampled data with a balanced number of data points in each class.</p>

<p style='text-align: justify;'>After performing a value count of the number of datapoints in each class, the median number of datapoints was found to be roughly 30000. Random oversampling and undersampling was therefore performed on the different classes such that each class had 30000 datapoints.</p>

### After Random Resampling of Majority and Minority Classes
<div style="overflow-x:auto;">
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th colspan="3"><strong>Avg/Total</strong></th>
        <th colspan="3"><strong>Class 0</strong></th>
        <th colspan="3"><strong>Class 1</strong></th>
        <th colspan="3"><strong>Class 2</strong></th>
        <th colspan="3"><strong>Class 3</strong></th>
        <th colspan="3"><strong>Class 4</strong></th>
        <th colspan="3"><strong>Class 5</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
      </tr>
      <tr>
        <td><strong>Logistic Regression (multilinear)</strong></td>
        <td>0.64</td>
        <td>0.65</td>
        <td>0.63</td>      
        <td>0.45</td>
        <td>0.24</td>
        <td>0.31</td>
        <td>0.53</td>
        <td>0.75</td>
        <td>0.62</td>
        <td>0.90</td>
        <td>0.93</td>
        <td>0.92</td>
        <td>0.63</td>
        <td>0.43</td>
        <td>0.51</td>
        <td>0.60</td>
        <td>0.92</td>
        <td>0.72</td>
        <td>0.75</td>
        <td>0.63</td>
        <td>0.68</td>
      </tr>
      <tr>
        <td><strong>Logistic Regression (ovr)</strong></td>
        <td>0.64</td>
        <td>0.65</td>
        <td>0.62</td>
        <td>0.46</td>
        <td>0.22</td>
        <td>0.29</td>
        <td>0.53</td>
        <td>0.77</td>
        <td>0.63</td>
        <td>0.90</td>
        <td>0.93</td>
        <td>0.92</td>
        <td>0.61</td>
        <td>0.44</td>
        <td>0.51</td>
        <td>0.60</td>
        <td>0.93</td>
        <td>0.73</td>
        <td>0.75</td>
        <td>0.60</td>
        <td>0.67</td>
      </tr>
      <tr>
        <td><strong>Random Forest Classifier</strong></td>
        <td>0.80</td>
        <td>0.79</td>
        <td>0.79</td>
        <td>0.68</td>
        <td>0.46</td>
        <td>0.55</td>
        <td>0.68</td>
        <td>0.85</td>
        <td>0.75</td>
        <td>0.94</td>
        <td>0.93</td>
        <td>0.93</td>
        <td>0.86</td>
        <td>0.78</td>
        <td>0.82</td>
        <td>0.73</td>
        <td>0.90</td>
        <td>0.80</td>
        <td>0.91</td>
        <td>0.84</td>
        <td>0.87</td>
      </tr>
      <tr>
        <td><strong>AdaBoost Classifier</strong></td>
        <td><strong>0.81</strong></td>
        <td><strong>0.81</strong></td>
        <td><strong>0.81</strong></td>
        <td>0.64</td>
        <td>0.56</td>
        <td>0.60</td>
        <td>0.75</td>
        <td>0.81</td>
        <td>0.78</td>
        <td>0.95</td>
        <td>0.93</td>
        <td>0.94</td>
        <td>0.88</td>
        <td>0.83</td>
        <td>0.85</td>
        <td>0.74</td>
        <td>0.88</td>
        <td>0.80</td>
        <td>0.92</td>
        <td>0.85</td>
        <td>0.89</td>
      </tr>
    </tbody>
  </table>
</div> 

## Model 2 Training Results

<p style='text-align: justify;'>In preparing the training data for the second model, instead of grouping the serious crimes together, all non-serious crimes are grouped as a class. However, as there are more instances of non-serious crimes as compared with serious crimes, the number non-serious crimes was therefore undersampled in such that as a group, their numbers were balanced with the number of each class of serious crimes.</p>

### After Random Resampling of Majority and Minority Classes
<div style="overflow-x:auto;">
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th colspan="3"><strong>Avg/Total</strong></th>
        <th colspan="3"><strong>Class 0</strong></th>
        <th colspan="3"><strong>Class 1</strong></th>
        <th colspan="3"><strong>Class 2</strong></th>
        <th colspan="3"><strong>Class 3</strong></th>
        <th colspan="3"><strong>Class 4</strong></th>
        <th colspan="3"><strong>Class 5</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
      </tr>
      <tr>
        <td><strong>Logistic Regression (multilinear)</strong></td>
        <td>0.58</td>
        <td>0.58</td>
        <td>0.56</td>
        <td>0.64</td>
        <td>0.34</td>
        <td>0.45</td>
        <td>0.44</td>
        <td>0.26</td>
        <td>0.32</td>
        <td>0.63</td>
        <td>0.80</td>
        <td>0.71</td>
        <td>0.66</td>
        <td>0.84</td>
        <td>0.74</td>
        <td>0.50</td>
        <td>0.78</td>
        <td>0.61</td>
        <td>0.58</td>
        <td>0.46</td>
        <td>0.51</td>
      </tr>
      <tr>
        <td><strong>Logistic Regression (ovr)</strong></td>
        <td>0.57</td>
        <td>0.58</td>
        <td>0.54</td>
        <td>0.65</td>
        <td>0.29</td>
        <td>0.40</td>
        <td>0.45</td>
        <td>0.24</td>
        <td>0.31</td>
        <td>0.63</td>
        <td>0.82</td>
        <td>0.71</td>
        <td>0.65</td>
        <td>0.87</td>
        <td>0.74</td>
        <td>0.49</td>
        <td>0.79</td>
        <td>0.61</td>
        <td>0.58</td>
        <td>0.44</td>
        <td>0.50</td>
      </tr>
      <tr>
        <td><strong>Random Forest Classifier</strong></td>
        <td>0.88</td>
        <td>0.88</td>
        <td>0.88</td>
        <td>0.86</td>
        <td>0.64</td>
        <td>0.73</td>
        <td>0.92</td>
        <td>0.97</td>
        <td>0.95</td>
        <td>0.92</td>
        <td>1.00</td>
        <td>0.95</td>
        <td>0.90</td>
        <td>1.00</td>
        <td>0.95</td>
        <td>0.82</td>
        <td>0.95</td>
        <td>0.88</td>
        <td>0.88</td>
        <td>0.74</td>
        <td>0.81</td>
      </tr>
      <tr>
        <td><strong>AdaBoost Classifier</strong></td>
        <td><strong>0.91</strong></td>
        <td><strong>0.91</strong></td>
        <td><strong>0.91</strong></td>
        <td>0.84</td>
        <td>0.74</td>
        <td>0.79</td>
        <td>0.95</td>
        <td>0.97</td>
        <td>0.96</td>
        <td>0.99</td>
        <td>1.00</td>
        <td>0.99</td>
        <td>0.98</td>
        <td>1.00</td>
        <td>0.99</td>
        <td>0.83</td>
        <td>0.93</td>
        <td>0.88</td>
        <td>0.88</td>
        <td>0.84</td>
        <td>0.86</td>
      </tr>
    </tbody>
  </table>
</div>

# Holdout Set Testing Results

## Model 1

<div style="overflow-x:auto;">
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th colspan="3"><strong>Avg/Total</strong></th>
        <th colspan="3"><strong>Class 0</strong></th>
        <th colspan="3"><strong>Class 1</strong></th>
        <th colspan="3"><strong>Class 2</strong></th>
        <th colspan="3"><strong>Class 3</strong></th>
        <th colspan="3"><strong>Class 4</strong></th>
        <th colspan="3"><strong>Class 5</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
      </tr>
      <tr>
        <td><strong>Random Forest Classifier</strong></td>
        <td>0.69</td>
        <td>0.64</td>
        <td>0.63</td>      
        <td>0.74</td>
        <td>0.40</td>
        <td>0.52</td>
        <td>0.34</td>
        <td>0.64</td>
        <td>0.45</td>
        <td>0.93</td>
        <td>0.93</td>
        <td>0.93</td>
        <td>0.56</td>
        <td>0.44</td>
        <td>0.49</td>
        <td>0.53</td>
        <td>0.90</td>
        <td>0.67</td>
        <td>0.81</td>
        <td>0.71</td>
        <td>0.75</td>
      </tr>
      <tr>
        <td><strong>AdaBoost Classifier</strong></td>
        <td><strong>0.81</strong></td>
        <td><strong>0.78</strong></td>
        <td><strong>0.78</strong></td>
        <td>0.85</td>
        <td>0.63</td>
        <td>0.73</td>
        <td>0.53</td>
        <td>0.73</td>
        <td>0.61</td>
        <td>0.96</td>
        <td>0.95</td>
        <td>0.95</td>
        <td>0.78</td>
        <td>0.74</td>
        <td>0.76</td>
        <td>0.64</td>
        <td>0.93</td>
        <td>0.76</td>
        <td>0.92</td>
        <td>0.81</td>
        <td>0.86</td>
      </tr>
    </tbody>
  </table>
</div>

## Model 2

<div style="overflow-x:auto;">
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th colspan="3"><strong>Avg/Total</strong></th>
        <th colspan="3"><strong>Class 0</strong></th>
        <th colspan="3"><strong>Class 1</strong></th>
        <th colspan="3"><strong>Class 2</strong></th>
        <th colspan="3"><strong>Class 3</strong></th>
        <th colspan="3"><strong>Class 4</strong></th>
        <th colspan="3"><strong>Class 5</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
        <td><strong>Precision</strong></td>
        <td><strong>Recall</strong></td>
        <td><strong>f1</strong></td>
      </tr>
      <tr>
        <td><strong>Randome Forest Classifier</strong></td>
        <td>0.70</td>
        <td>0.48</td>
        <td>0.54</td>      
        <td>0.87</td>
        <td>0.50</td>
        <td>0.64</td>
        <td>0.08</td>
        <td>0.20</td>
        <td>0.12</td>
        <td>0.79</td>
        <td>0.61</td>
        <td>0.69</td>
        <td>0.13</td>
        <td>0.72</td>
        <td>0.22</td>
        <td>0.30</td>
        <td>0.65</td>
        <td>0.41</td>
        <td>0.06</td>
        <td>0.03</td>
        <td>0.04</td>
      </tr>
      <tr>
        <td><strong>AdaBoost Classifier</strong></td>
        <td>0.71</td>
        <td>0.47</td>
        <td>0.53</td>
        <td>0.88</td>
        <td>0.48</td>
        <td>0.62</td>
        <td>0.09</td>
        <td>0.15</td>
        <td>0.11</td>
        <td>0.79</td>
        <td>0.55</td>
        <td>0.65</td>
        <td>0.13</td>
        <td>0.73</td>
        <td>0.22</td>
        <td>0.30</td>
        <td>0.73</td>
        <td>0.42</td>
        <td>0.07</td>
        <td>0.05</td>
        <td>0.06</td>
      </tr>
      <tr>
        <td><strong>Logistic Regression (multinomial)</strong></td>
        <td><strong>0.71</strong></td>
        <td><strong>0.42</strong></td>
        <td><strong>0.48</strong></td>
        <td>0.89</td>
        <td>0.40</td>
        <td>0.55</td>
        <td>0.08</td>
        <td>0.20</td>
        <td>0.11</td>
        <td>0.80</td>
        <td>0.79</td>
        <td>0.80</td>
        <td>0.12</td>
        <td>0.84</td>
        <td>0.20</td>
        <td>0.32</td>
        <td>0.69</td>
        <td>0.44</td>
        <td>0.08</td>
        <td>0.04</td>
        <td>0.06</td>
      </tr>
    </tbody>
  </table>
</div>

# Conclusion

<p style='text-align: justify;'>In summary, the best performing model in Model 1 performed relatively well and was able to classify crimes with a minimum recall score of 63%. This means that for each class of crime, the model was able to correctly capture at least 63% of them. </p>

<p style='text-align: justify;'>If we shift our focus to the classification of serious crimes, where the urgency of prevention is higher than that of other crimes, it should be noted that the first model managed to achieve a recall score of 81%, meaning it was able to capture 81% of all crimes that are classified as serious in the Holdout Set.</p>

<p style='text-align: justify;'>As regards the models for Stage 2, while the f1 score appears relatively low across all models, that can be attributed to the particularly low precision and recall scores of class 5, which represents the agglomoration of all non-serious crimes. If we look more closely at the best performing model, we can see that it managed to achieve a recall score of 0.79 in relation to rape cases, 0.84 in relation to arson and 0.69 in relation to shooting cases.</p>

<p style='text-align: justify;'>It can be concluded that the models above are able to reasonably classify crimes, with particular attention on serious crimes. The models could, however, possibly be improved if more data could be gathered in relation to each crime, such as the zoning of the land on which the crime took place, whether there are any CCTV cameras operating in the vicinity of the scene of crime or the distance to the closest CCTV camera, and the frequency of police patrols in that area.</p>

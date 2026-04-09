### Machine Learning

The following are the main types of Machine Learning:
- Supervise Learning (mostly used in real-life applications)
- Unsupervised Learning
- Recommender System
- Reinforcement Learning

Having tools is not enough to build a good Machine Learning system, 
You have to practice equally well and know when to apply them, and how to apply them

## Supervised Learning
- Key characteristic that we give our learning algorithms examples to learn from, and include the right answers
- Second is that by giving correct pairs of input X and desired output Y, the learning algorithm eventually learns 
to take input only without the output label Y and gives accurate result, prediction, and guess
- Learning X with input and being able to give the expected output Y
  X -> Y

Examples: 
<img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/dab9b471-86d4-47b4-aad3-31cf9947a7e4" />

- In all examples above, we first train our model with inputs X and output Y for correct answers, so that it can learn.
- After the model has learned using X and Y pairs, it then takes fresh input only and tries to give the output Y itself through what it has learned.

## Regression (Type of Supervised Learning)

<img width="900" height="400" alt="image" src="https://github.com/user-attachments/assets/908bf713-cd66-4250-9eec-180cae2a5c56" />

- Above is an example of a house prediction model.
- We made the model through X, Y pairs, i.e., inputs, outputs.
- Now there could be a price prediction through a simple straight line, which is a simple function.
- Fitting a straight line isn't the only learning algorithm you can use.
- Or the house price can be predicted by a more complex function denoted by a curve line.
- Selecting the function has to be decided through special understanding.
- Where prediction comes in, that type of Supervised learning is known as Regression (Predicting Something, a number)
- One thing you see is how to get an algorithm to systematically choose the most appropriate line or curve or other thing to fit to this data.

## Classification (Type of Supervised Learning)

### 1. Example of Classification with Single Input and Output

<img width="900" height="400" alt="image" src="https://github.com/user-attachments/assets/2e29eaf7-974a-44b6-a2bf-e65f47c191bf" />

- Imagine we have to detect Breast Cancer and we are building a machine learning system for that.
- Using the patient's medical records and history to feed to the system
- If a tumor is dangerous and cancerous, we call it malignant.
- If the tumor is not that dangerous than we call it begnin
- Imagine our dataset has the size of the tumor shown on the right side in the table,
  along with it, we label begnin with 0 and malignant with 1.
- X axis shows tumor size, Y axis shows tumor is 0 (begnin) , 1 (malignant)
- Now, Classification is different from Regression as it will make a prediction
  based on a set of possible results, unlike Regression, which has an infinite number of possible results list and it has
  to answer from them. That's why the classification problem  has a small set of answers and is different from regression.
  Regression will result in the form of a number. Classification can be either text or a number from the possible results.

### 2. Example: Plotting the Graph on a Straight Line

<img width="919" height="166" alt="image" src="https://github.com/user-attachments/assets/adb90761-a184-49a2-80bc-2ced3b8b2a22" />

  - This dataset can be plotted on a straight line as well.
  - Now imagine the black mark, the patient falls here, would it be malignant or benign?
  - Classification can have more than 2 possible results (classes, categories)

### 3. Example with Multiple Inputs

<img width="781" height="496" alt="image" src="https://github.com/user-attachments/assets/1667ca62-d134-42d9-8fcb-e8148afda5d9" />

- Now in this Scenario, it is possible there could be more than one input required in a problem.
- Here we have age and tumour size as input, now how will we classify it is benign or malignant?
- Keeping the pink circle as the identification area of the tumor.
- In case of multiple inputs, there has to be a boundary line for differentiation, as the yellow line shown in
  the graph. This boundary line helps us to identify that the tumor is benign. This is how
  we classify the result if there are multiple inputs by having a boundary to separate the likelihood of a result.

## Unsupervised Learning

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/6688a3e1-d74e-44cf-a7d5-1a085ea1f163" />

- In Supervised Learning, we have a set of right answers
- This is another type of machine learning model: unsupervised learning
- In this type, we don't have a labelled dataset, unlike supervised learning
  We don't give a set of possible output results to the model and the problem.
- Unsupervised learning only has a set of Inputs.
- As in the graph above, such a dataset being unlabelled and results possible to group with the similar ones
  is known as a Clustering Algorithm.
- In the clustering algorithm of Unsupervised Learning, the dataset is unlabeled.
- Similar Results are grouped in different clusters and then the point on the graph predicts the result based on the cluster.
- We ask the algorithm to figure out all by itself what's interesting or what patterns or structures there might be in this data


### 1. Example: Google News

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/1661cf69-4d27-4081-adc4-43e2ca012adc" />

- Google News works by clustering similar data.
- Looks for stories, news and clusters the similar ones.
- As we can see in the picture above, the search results of Panada, twin, and zoo are matching
  and showing in one cluster
- The Clustering algorithm itself groups similar words and news in one cluster and answers accordingly.

### 1. Example: Micro Detection
<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/523a3063-17a7-436d-9a71-dad5183f4b53" />

### 3. Example: Customer Group Segmentation

<img width="700" height="600" alt="image" src="https://github.com/user-attachments/assets/f4193144-83a3-4cd5-8720-5c17819078c8" />

- This is a clustering that our team used to try to better serve our community as we're trying to figure out what the major categories of learners are in the DeepLearning.ai community.


### Other Types of Unsupervised Learning
<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/5ac2e9a1-c519-4a65-b561-9d6892add0c7" />


## Supervised Learning:

### Linear Regression

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/444e7c44-04c2-4b88-9543-fddef6ecd551" />

- Linear Regression is one main type of Regression problem where we get straightline fit for predicting the result.
- Any Supervised model that predicts a number as a result is known as a regression problem.
- We give labelled dataset as inputs x,y, and we get a straight line fit on the graph to predict the value based on linear Regression
- Assume we have to predict the price of a house based on the size of the house (x-axis) , price on (y-axis)
- Price denotes the most recently built house, which means we have trained data on possible results. The right answers are given in the dataset
- Using the graph, we are prediciting price of a house with 1250cm sq ft based on a straight line fit our model can depict, which is $ 220k.
- Another type of Supervised learning is a classification model; there are only a small number of outputs.

### Terminologies

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/3a7650f6-07fe-47d1-bc0c-2cda7ec1912f" />

###  Training Set
    - The data used to train the model is known as the training set.

### Input Variable/Feature 
   - The standard notation to denote inputs or input variables is lower case x, known as a feature or, in the case of multiple inputs, as a feature set.

### Target Set
  - The standard notation to denote inputs or output variables is lower case y, known as the target variable
  -  To denote all the possible training examples, we use "m" meaning all the training examples like (x,y) => (40,90), (90,300), (600,450)
  - In the picture above, i in superscript doesn't represent power, it shows the "ith" training set. This is not exponent


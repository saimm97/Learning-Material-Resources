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
- Now Classification is different from Regression as it will make a prediction
  based on a set of possible results, unlike Regression, which has an infinite number of results list and it has
  to answer from them. That's why Classification problem  has small set of answers and is different from regression.
  Regression will result in form of a number. Classification can be either text or a number from possible results.

  

### 2. Example as Plotting the Graph on a Straight line

<img width="919" height="166" alt="image" src="https://github.com/user-attachments/assets/adb90761-a184-49a2-80bc-2ced3b8b2a22" />

  - This dataset can be plotted on a straightline as well.
  - Now imagine the black mark, the patient falls here, would it be malignant or benign?
  - Classification can have more than 2 possible results (classes,category)



### 3. Example with Multiple Inputs

<img width="781" height="496" alt="image" src="https://github.com/user-attachments/assets/1667ca62-d134-42d9-8fcb-e8148afda5d9" />

- Now in this Scenario, it is possible there could be more than one input required in a problem.
- Here we have age and tumour size as input, now how will we classify it is benign or malignant?
- Keeping the pink circle as the identification area of tumor.
- In case of multiple inputs, there has to be a boundary line for differentiation, as the yellow line shown in
  the graph. This boundary line helps us to identify that the tumor is benign. This is how
  we classify the result if there are multiple inputs by having a boundary to separate the likelihood of a result.

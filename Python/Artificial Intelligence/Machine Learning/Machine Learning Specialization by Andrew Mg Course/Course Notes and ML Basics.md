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

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/1debd617-2f09-4bea-b5a7-900606d0a942" />


###  Training Set
    - The data used to train the model is known as the training set.

### Input Variable/Feature 
   ```
   - The standard notation to denote inputs or input variables is lower case x, known as a feature or, in the case of multiple inputs, as a feature set.
  ```

### Target Set
  ```
  - The standard notation to denote inputs or output variables is lower case y, known as the target variable.
  -  To denote all the possible training examples, we use "m" meaning all the training examples like (x,y) => (40,90), (90,300), (600,450)
  - In the picture above, i in superscript doesn't represent power, it shows the "ith" training set. This is not an exponent.
  ```

<img width="2543" height="1258" alt="image" src="https://github.com/user-attachments/assets/ae95888d-d73f-4db3-8d9c-f8919bb54887" />

## Cost Function

### Squared Error Cost Function 

<img width="1977" height="996" alt="image" src="https://github.com/user-attachments/assets/ba71f82a-fb14-4c73-921f-0362333d91ca" />

$$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2$$

Reading it piece by piece:
- y(i) = the *actual* value (the real data point)
- y^​(i)=fw,b​(x(i)) = the *predicted* value (where the line says it should be)
   
- $f_{w,b}(x^{(i)}) - y^{(i)}$ → the error (prediction minus actual) for one example
- $\left(\dots\right)^2$ → square it, so errors are always positive and big errors count more
- $\sum_{i=1}^{m}$ → add up over all $m$ training examples
- $\frac{1}{m}$ → average it
- the extra $\frac{1}{2}$ → a convenience that makes the calculus cleaner later (when you take derivatives for gradient descent, the 2 cancels out)
## The goal
 
> Find $w, b$ such that $\hat{y}^{(i)}$ is close to $y^{(i)}$ for all points.
- In other words, choose the slope and intercept that make J(w,b)J(w,b)
J(w,b) as small as possible. A small cost means the line fits the data well; a large cost means predictions are far off. The next step in the course (gradient descent) is the algorithm that actually finds those best values of ww
w and bb
b.
 
"Making J as small as possible" just means: **find the slope $w$ and intercept $b$ that make the line hug the data as closely as it can.**
 
## The simplified case (just w)
 
If we drop $b$ (set $b = 0$), the line is forced through the origin and there is only **one** knob to turn: $w$. This makes the idea easy to picture because $J$ now depends on a single variable:
 
$$f_w(x) = wx \qquad J(w) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_w(x^{(i)}) - y^{(i)}\right)^2$$
 
## The picture: a bowl
 
Try different slopes:
 
- Too shallow a slope → line misses the points → $J$ is large
- Too steep a slope → line overshoots the points → $J$ is large
- Just right → line goes through the middle of the points → $J$ is smallest
If you plot $J$ on the vertical axis against $w$ on the horizontal axis, you get a **U-shaped (bowl) curve**:
 
```
 J(w)
  │ \                         /
  │  \                       /
  │   \                     /
  │    \                   /
  │     \_               _/
  │       \_           _/
  │         \__     __/
  │            \___/   ← minimum (best w)
  └─────────────┴────────────── w
              w* (best fit)
```
 
### Worked example
 
Data points: $(1,1)$, $(2,2)$, $(3,3)$. The best slope is $w = 1$.
 
| w   | line fit        | J(w)  |
|-----|-----------------|-------|
| 0   | flat, misses all| 2.33  |
| 0.5 | too shallow     | 0.58  |
| 1   | passes through all | 0.00 |
| 1.5 | too steep       | 0.58  |
| 2   | way too steep   | 2.33  |
 
As $w$ moves toward 1, the line snaps onto the points and $J$ drops to the bottom of the bowl. At exactly $w = 1$, the line hits every point and $J = 0$.
 
## So what does "minimize J" mean?
 
It means: **roll the ball to the bottom of that bowl.** Find the $w$ (and in the full version, $b$ too) sitting at the lowest point, because that is the setting where your predictions are closest to reality across all your data.
 
Squaring the errors is exactly *why* you get a smooth bowl instead of a jagged mess — squaring makes the cost grow gently as you drift from the best fit, giving the optimizer a clean downhill path to follow.
 
That "rolling downhill to the bottom" is literally the next algorithm: **gradient descent**. It starts at some random $w$, checks which way is downhill, and takes steps toward the minimum until it reaches the bottom of the bowl.
 


# Visualizing the Cost function


<img width="1828" height="898" alt="image" src="https://github.com/user-attachments/assets/95d99cda-19fa-44bf-a51a-8ec7fb8a5ba6" />


# Model, Parameters, Cost function, Objective

<img width="1859" height="977" alt="image" src="https://github.com/user-attachments/assets/74c224b9-0bf1-41c7-8cce-7354d80b5091" />

- Here we will see how f(w,b) => function of x  relates to J(w) => function of w,b
- Formula f(x) = wx + b , we assume w + 0.06 , b = 50
- f(x) = 0.06w + 50
- When we had b = 0, and w had values, we got the shape of a bowl for J(w,b)
- Though the plot becomes a bit complex

<img width="1671" height="973" alt="image" src="https://github.com/user-attachments/assets/ec9cf91f-b9f8-4d66-979a-2ce3b3e9bf8d" />


- Now, in this plot, we have values for w and b both.
- For example, if w = -10, b = -15 than height of the surface above  these two would be the J(w= -10, b= -15

<img width="1749" height="970" alt="image" src="https://github.com/user-attachments/assets/b5468fdc-6369-4d42-8401-42a23db2548a" />

- Another way to plot this function is to use a contour plot.
- Overall we can see all these points fit terribly and away from the best first value of J.
- We use contour plots when there are multiple inputs.

**Top-left plot (f_w,b):** This shows your data — house prices vs. size in feet². The red X's are training examples. The three colored lines (blue, orange, green) are three different candidate models f(x) = wx + b. Notice all three lines fit the data terribly — they have negative slopes and sit way below the data points. These are bad parameter choices, chosen to illustrate a point.

**Top-right plot (contour plot):** This is the key one. The cost function J(w, b) depends on two variables, w and b. To plot a function of two inputs, you'd normally need 3D, but a contour plot collapses it into 2D — like a topographic map.

A contour plot works by drawing lines that connect points of equal height (equal cost J). Each ellipse/ring is one "elevation" of cost:
- Points on the same ring have the **same** value of J
- The center of the rings is the **lowest** cost — the minimum, where the best (w, b) lives
- Rings far from the center mean **high** cost (bad fit)

The three colored X's mark where the three lines from the left plot sit in (w, b) space. They're all far from the center, in the outer rings — which is exactly why they fit the data so poorly. The dashed lines just project each point down to read off its w and b coordinates.

**Bottom plot (3D surface):** This is the *same* cost function J, but shown in full 3D as a bowl-shaped surface. The contour plot above is literally this bowl viewed from directly overhead, with the rings being slices at different heights. The bottom (minimum of the bowl) corresponds to the center of the contour rings. The three colored X's sit high up on the walls of the bowl — again showing high cost.

**The big idea:** Minimizing the cost means moving toward the center of the contour rings / the bottom of the bowl. That's what gradient descent does — it walks "downhill" on this surface to find the (w, b) that best fits the data.


## Code for Linear Regression Cost Formula with one Variable

### Computing Cost
The term 'cost' in this assignment might be a little confusing, since the data is housing cost data. Here, cost is a measure of how well our model is predicting the target price of the house. The term 'price' is used for housing data.

The equation for cost with one variable is:
  $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$ 
 
where 
  $$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}$$
  
- $f_{w,b}(x^{(i)})$ is our prediction for example $i$ using parameters $w,b$.  
- $(f_{w,b}(x^{(i)}) -y^{(i)})^2$ is the squared difference between the target value and the prediction.   
- These differences are summed over all the $m$ examples and divided by `2m` to produce the cost, $J(w,b)$.  
>Note, in lecture summation ranges are typically from 1 to m, while code will be from 0 to m-1.

### Gradient Descent

<img width="700" height="700" alt="image" src="https://github.com/user-attachments/assets/581f55c8-9fe2-4925-93be-6ea8c7cc5056" />

- Gradient Descent is a function that can be used to minimize any function, not only a linear function
- Gradient Descent applies to more general functions or other cost functions having 2 parameters
- Imagine we have a cost function J = J(w1..w2...w3 wn), our goal is to minimize cost J over parameters (w1, w2,,wn, b). Gradient Descent can be used to minimize it
- What we will do is start with some initial values of w and b. As this is linear regression, we can take w=0, b=0.
- By taking small guesses, we will use gradient descent to minimize the cost.
- We will continue to change the value of w,b at the point till the J(w,b) settles near the minimum.

<img width="900" height="800" alt="image" src="https://github.com/user-attachments/assets/6d6c7e81-ecbc-49a6-9b56-d4412570deba" />





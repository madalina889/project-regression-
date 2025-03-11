# project-regression-
Supervised machine learning and linear regression 
1 class with new teacher 
We’re focusing on regression part , supervised has labels unsure. l. Doesn’t . 
We have regression and classification .
Classification is a method to predict .
The regression has a line thats goes in the middle of the points  , using that it’s gone became the model , we make the line through the most of points in the plot and that’s gone be the regression line .
Between a classifier and another which is the line that best separates them , , that’s what classification does . When is binary is classification , when it’s only one thing prediction is regression like regional temp. But if is is gone rain or not is classification ,2 things to work on . Everything that turns out that’s binary is classification .
Machine learning problem : regression and classification 
Understand the difference between.
First step Is model.fit (x,y) and that way we find the linear regression 
The predict 
Each point has an amount of error 
X = df_sales[‘ads].values …. Is capital X is big 
Close to the line there is air between the line and the data and that can be calculated , we calculate the distance between the points and regression line .
After the prediction we evaluated this is whats coming out of the third added column , the second is the prediction. 
How we calculate the air 

If I have a category of happy or not happy 
get_dummies() 
Beside the training set (we create the model and fit the data) , once we have that model we predict the testing set 
random_state = 42 : the number we choose is because we want to have the same randomization to work on the same project , we match our codes that way model
Model.score is showing how good is our model of prediction , in med for cancer is I reach the 0.78 is good couse is like .

Chat GPT explanation of cod(breaking down codes : 
 
By having the info and doing the scatter plot , I’m asked to do : # Reformat data of the independent variable X as a single-column array
X = df_salary['years_experience'].values.reshape(-1,1)

# Display sample data
X[:5]
Why? Context:
When you're working with data in machine learning, especially for tasks like regression, you often work with two sets of data:
Independent variable (X): This is the input data. In your case, it’s years_experience.
Dependent variable (y): This is the output data. Typically, this would be something like the salary of employees, but you didn’t provide that here, so I’m assuming it’s something like that.
The purpose of reshaping the data is to make sure it matches the format that machine learning algorithms expect. Let’s break it down step-by-step.
What does reshape(-1, 1) do?
You have a column (years_experience) that contains data like this:
csharp
Copy
[1.1, 1.3, 1.5, 2.0, 2.2]
This is a 1D array (a single list of values). But many machine learning algorithms, such as linear regression, expect the input data to be in a 2D array format. This is because they expect the data to be structured like a "table" where:
Each row corresponds to a data point.
Each column corresponds to a feature or variable.
When you use reshape(-1, 1), it turns this 1D array into a 2D column vector. The -1 means "figure out how many rows to make based on the data," and the 1 means "make sure there's only one column."
This is a 2D array, where each value from the original list is now in its own row, and there’s only one column. This is exactly the format that models like linear regression expect for their input.
Understanding X.shape
The .shape attribute of a NumPy array (or similar data structures) tells you the dimensions of the array. Specifically:
First number: This is the number of samples or data points (i.e., how many individual rows you have).
Second number: This is the number of features (i.e., how many columns of data each sample has).
# Create a model with scikit-learn
model = LinearRegression()
What is Linear Regression?
Linear Regression is a supervised machine learning algorithm used to model the relationship between a dependent variable (often called the target variable, y) and one or more independent variables (also called predictors or features, X). In your case:
X: Independent variable, which is years of experience.
y: Dependent variable, which is salary (this is the target you want to predict).
# Fit the data into the model
model.fit(X, y)
 After doing the model we fit the data of x and y . 
The Linear Regression Model:
The simple linear regression equation looks like this:
y
=
β
0
+
β
1
⋅
X
y=β0 +β1 ⋅X
Where:# Display the y-intercept


y is the predicted value (salary, in your case).
X is the input value (years of experience).
β₀ is the intercept(intercetta) (the value of y when X = 0).
β₁ is the slope(pendenza) (how much y changes when X increases by one unit).
Why Use Linear Regression?
Linear regression helps answer questions like:
How does the years of experience affect salary? Linear regression will help quantify that relationship.
What salary would you predict for someone with a given amount of experience?

After fitting the model :# Display the slope
print(f"Model's slope: {model.coef_}”)
# Display the y-intercept
print(f"Model's y-intercept: {model.intercept_}")

# Create a copy of the original data
df_salary_predicted = df_salary.copy()

# Add a column with the predicted salary values
df_salary_predicted["salary_predicted"] = predicted_y_values

# Display sample data
df_salary_predicted.head()
model.predict(X): This line uses the trained linear regression model to make predictions for each value of X (the years of experience in your dataset).
X is your input data (the years of experience for each person).
The model.predict() function calculates the predicted salary values based on the relationship the model learned during training (the intercept and slope).
predicted_y_values will be an array of predicted salaries corresponding to each value of X (years of experience).
Evaluation metrics that define how good is the model 
# Import relevant metrics - score, r2, mse, rmse - from Scikit-learn from sklearn.metrics import mean_squared_error, r2_score
# Compute the metrics for the linear regression model score = round(model.score(X, y, sample_weight=None),5) r2 = round(r2_score(y, predicted_y_values),5) mse = round(mean_squared_error(y, predicted_y_values),4) rmse = round(np.sqrt(mse),4) # Print relevant metrics. print(f"The score is {score}.") print(f"The r2 is {r2}.") print(f"The mean squared error is {mse}.") print(f"The root mean squared error is {rmse}.”)
mean_squared_error: This function calculates the Mean Squared Error (MSE) between the actual values (y) and the predicted values (predicted_y_values). MSE measures the average of the squares of the errors (the differences between actual and predicted values). A lower MSE indicates a better fit of the model.
r2_score: This function computes the R² score (also called coefficient of determination), which is a measure of how well the model explains the variation in the data. R² ranges from 0 to 1 (for regression models). A higher R² value indicates that the model does a better job of explaining the variance in the dependent variable (y).
Again, round(..., 5) rounds the result to 5 decimal places.


# The first three years look like significant outliers. # Reduce the DataFrame to just the "Year" and "Total" columns # and only from 2003 onwards df = pd.DataFrame(df_electricity.loc[df_electricity["Year"]>=2003,["Year","Total"]])\ .reset_index().drop(columns="index") df.head() : explain why the code is built like that
ChatGPT said:
df_electricity["Year"] >= 2003: This part of the code is a boolean filter that checks for rows where the "Year" is greater than or equal to 2003. It essentially creates a mask that is True for all rows where the "Year" value is 2003 or later.
df_electricity.loc[...]: The .loc[] function is used to filter the DataFrame based on the condition provided inside the brackets.
The condition df_electricity["Year"] >= 2003 ensures that only the rows where the year is 2003 or later are selected.
The second part, ["Year", "Total"], ensures that only the columns "Year" and "Total" are kept for the filtered rows.
After this operation, you're left with a DataFrame that contains only the rows with years starting from 2003 and only the columns "Year" and "Total"
Activity 5 ,review some point : 
pd.get_dummies  is used to convert variables into dummy/indicator variables (also called one-hot encoding . Let’s say I have a column of car_type and the rows are sedan ,SUV, TRUCK . SEDAN ,; ALL OF THIS IS GONE BECAME car_type_sedan Car_type_Suv etc all of them columns , the function 0 , 1 is gone be assign to each category is assigned.
pd.get_dummies() converts categorical variables in your DataFrame into numerical columns using one-hot encoding.
This is often required for machine learning models that can't directly handle categorical data.
 NOW WE HAVE :
# Use Pandas .astype("category").cat.codes for single column category encoding
columns_to_encode = ["make",
                     "fuel-type",
                     "aspiration",
                     "body-style",
                     "drive-wheels",
                     "engine-location",
                     "engine-type",
                     "fuel-system"]

# Copy car_data
car_data_cat_codes = car_data.copy()

# Loop through columns_to_encode and convert the columns to category codes
for column in columns_to_encode:
    car_data_cat_codes[column] = car_data_cat_codes[column].astype("category").cat.codes
When you convert this column to a categorical type and access .cat.codes, pandas will assign an integer to each category, and then encode the values in the column accordingly.
Convert the column to category type.
Assign integer codes to each unique category.
The resulting codes will replace the original categories.

In the context of machine learning and data processing, encoding refers to the process of converting non-numeric data(such as categorical data) into a numeric format that machine learning algorithms can understand.
One-Hot Encoding: Creates binary (0 or 1) columns for each unique category. (What pd.get_dummies() does).
Label Encoding (Category Encoding): Assigns each unique category a numeric code. (What .cat.codes does).
One-hot encoding is particularly useful when you need to convert categorical features (like car make, fuel type, etc.) into a numeric form that machine learning algorithms can use for analysis or modeling.
Key Points of LabelEncoder:
Label Encoding: This assigns an integer value to each unique category in a categorical column.
For example, if you have a column with values like "Red", "Green", and "Blue", label encoding might convert them into integers like 0, 1, and 2.
Non-Ordinal Categorical Data: Even though label encoding works for non-ordinal data (i.e., no inherent order), it introduces a risk that the algorithm might treat the encoded numbers as ordered (which isn't always the case). For non-ordinal categorical data, one-hot encoding is often preferred, but label encoding can still be used in some cases.

Summary of What Each Step Does: (one hot encoder)
OneHotEncoder(handle_unknown='ignore'): Initializes the encoder to handle unknown categories by ignoring them.
fit(): Learns the unique categories in the specified columns.
transform(): Converts the categorical data into one-hot encoded format (sparse matrix).
The result (car_data_ohe) is the one-hot encoded data in a sparse matrix form, which is memory efficient.
Useful for Ordinal Data: If there is a meaningful order to the categories (e.g., "Low", "Medium", "High"), label encoding will assign increasing integers to reflect that order.
Why Use LabelEncoder?
It is simpler and more memory-efficient than one-hot encoding when dealing with columns that have a small number of unique categories.
DIFFERENCES BETWEEN THOSE KIND OF VARIABLES WHICH ARE ALL TECHNIQUES TO CONVERT THE VARIABLES INTO A NUMERICAL FORMAT THAT MACHINE LEARNING MODELS CAN USE. 
1. Label Encoding:
Label Encoding assigns an integer (or label) to each unique category in a categorical column. The integer value is simply a numeric label for each category.
How it Works:
Each unique value in a categorical column is assigned a numeric code (usually starting from 0).
This is suitable for ordinal categorical data (where there is a meaningful order between the categories) because the numeric values might represent a ranking or hierarchy.
LIKE : SUV , SUDAN , TOYOTA, SUDAN , SUV : 0 ,1 ,2,1 .. ALL TOYOTA ARE 2 ECC.
2. One-Hot Encoding:
One-Hot Encoding creates a binary column for each category in a categorical column. Each column represents a category, and the value for each column is 1 if the original value is that category, and 0 if it is not.
How it Works:
For each unique category, create a new column.
For each row, mark 1 in the column corresponding to the category for that row, and 0 in the other columns.
3. Dummy Variables (or Dummy Encoding):
Dummy Variable Encoding is very similar to One-Hot Encoding, but it typically excludes one category from the encoding. This helps avoid the dummy variable trap (multicollinearity). The dummy variable trap occurs when one category can be perfectly predicted by the others, which can affect some models, such as linear regression.
How it Works:
Like one-hot encoding, dummy variables create a new column for each category.
However, it excludes one category (usually the first or last category) to serve as the "reference" category.
The model can use the other columns to determine the presence of that excluded category.
SO HAS TO COLUMNS AS THE HOT ENCODING T=BUT INSTEAD OF PUTTING TOYOTA SUV AND TRUCK ,PITS JUST SUB=V AND TRUCK AND WHEN BOTH VALUES ARE 0 AND 0 MEANS THAT IS TOYOTA THERE 
GOING TO ACTIVITY 7 
# Use the Sklearn `train_test_split()` function to split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
This split helps to assess how well the model generalizes to new, unseen data, which is crucial for evaluating its true performance and avoiding overfitting.
2. Why Use train_test_split()?
The train_test_split() function from scikit-learn is a convenient way to randomly split your dataset into these two sets.
Training Data (X_train, y_train): The model uses this data to learn the patterns in the data. The features(X_train) and the corresponding target values (y_train) are used to train the machine learning model.
Testing Data (X_test, y_test): This data is not seen by the model during training. After training the model on the training data, you use the testing set to evaluate how well the model can predict new data that it hasn't seen before. It allows you to measure the model's performance and assess its generalization ability.
Why: to avoid overfitting , to see how good is the model to evaluate , real-world simulation.
What to do then to see if the model is correct ? : looking  for the model score (r2 -Coefficient of Determination).
Training Score vs. Testing Score: If the R² score on the test set is similar to the R² score on the training set, this indicates that the model is generalizing well.

Class 2 
Variance and Bias are models of the predictions .
Bias means the model that build around the points is not one line , Is like when is trying to get how many points it can by going on the directions of what would be a regression line .
Overfitting, undercutting   , k-fold cross-validation :split the data , dataset is gone be decided in 4 
P-values : hypothesis testing , way to form a question where you can affirm or negate the question. You’re not sayin if is better or worst but just if there is a change in the data. P values are based on the treat we talked about, what it matters? Should I keep it or let it go ? Like if I have a lot of columns I use p values to see how the columns relate to the target , to what I really want to answer to . I don’t need all 20 columns to answer to my simple question. P value is gone help select . Once we generate values per column how do we reed them? We are looking for the top part and less variance . Then we do the score of the first model with all columns and the model of the selected columns . I do the r2 and I see that before was negative which is useless then we increase the score by selecting columns .
We need to get rid of the useless columns and p values method helps us do it . 
lr = sm.OLS(y_train, X_train).fit(
is using Statsmodels (sm) to perform Ordinary Least Squares (OLS) regression.
Another point can be that if the number is bad of the data can be also be that the data is trash , is not good ,not making sense to work on it.
Regularization : we want also to try other algorithms that deals with overfit .
I want to minimize the squared residential of the regression all the time and there is also the lasso regression model to do it .
We put in the code ridge (ridge regression model) you can put multiple alfas in there and see what alfas is the best for the regression . What this alfa means , I=here the ridge alpha 0.01 and alpha 0.001 etc is like tasting the alfas as we were doing with clusters , in this case to see what alpha is better .
VIF deals with multicollinearity which is bad 
Regularization : a way of get rid of overfitting .


# Create a function to calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
This code defines a function calc_vif to calculate the Variance Inflation Factor (VIF) for each feature in a given dataset X. The VIF is used to detect multicollinearity among the independent variables in a regression model. High VIF values suggest that a predictor variable is highly correlated with other predictor variables, which can make the model unstable.
Let's break down the code line by line:
1. Importing the variance_inflation_factor function:
python
Copy
from statsmodels.stats.outliers_influence import variance_inflation_factor
This line imports the variance_inflation_factor function from the statsmodels library, which is used to compute the VIF for a given variable. variance_inflation_factor takes two inputs:
The design matrix (which includes all the independent variables in your regression model).
The index of the variable for which the VIF needs to be calculated.
2. Defining the function calc_vif:
python
Copy
def calc_vif(X):
Here, the function calc_vif is defined, and it takes one parameter X, which is expected to be a Pandas DataFrame containing the independent variables (predictors) of your regression model.
3. Creating an empty DataFrame for VIF results:
python
Copy
    vif = pd.DataFrame()
This line creates an empty Pandas DataFrame vif where the results will be stored. This DataFrame will eventually contain two columns:
variables: The names of the independent variables.
VIF: The corresponding VIF values for each variable.
4. Filling the DataFrame with variable names and VIF values:
python
Copy
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
X.columns: This gets the names of the columns (variables) in the DataFrame X, and assigns them to the variablescolumn in the vif DataFrame.
variance_inflation_factor(X.values, i): This calculates the VIF for each variable i. The X.values converts the DataFrame to a NumPy array, which is what variance_inflation_factor needs. The i is the index of the variable for which the VIF is being calculated.
The list comprehension [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]iterates over each column index i of X (from 0 to X.shape[1] - 1, where X.shape[1] is the number of columns in X) and calculates the VIF for each variable.
The resulting list of VIF values is assigned to the VIF column in the vif DataFrame.
5. Returning the VIF DataFrame:
python
Copy
    return(vif)
Finally, the function returns the vif DataFrame, which now contains the names of the variables and their corresponding VIF values.

The function calc_vif takes a dataset X, computes the VIF for each variable in X, and returns a DataFrame containing the variable names and their respective VIF values. The VIF helps to detect multicollinearity by indicating how much the variance of a regression coefficient is inflated due to correlations with other variables in the model.



CLASS 3 
WORKFLOW: loading data and cleaning it by removing nulls (get_dummies on datasets) : splitting data in test and training ; selectinfg best model ; training data : are the codes we always use.
Pipeline is inside sklearn and through this you can say what you want to obtain. 

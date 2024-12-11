import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st


st.markdown("# Welcome to my LinkedIn user prediction app!")

st.markdown("#### Please enter some features below to see if someone is a LinkedIn user")

def clean_sm (x):
    x = np.where(x == 1,
                 1,
                 0)
    return x


s = pd.read_csv("social_media_usage.csv")


#Create the new dataframe
ss = pd.DataFrame({
        "sm_li": s["web1h"].apply(clean_sm),
        "income":np.where(s["income"]>9, np.nan, s["income"]),
        "education": np.where(s["educ2"] >8, np.nan, s["educ2"]),
        "parent": np.where(s["par"] == 1, 1, 0),
        "marital": np.where (s["marital"] == 1, 1, 0),
        "age": np.where (s["age"]>98, np.nan, s["age"]),
        "female": np.where(s["gender"] ==2, 1,0)
        })
    #Drop null values
ss = ss.dropna()
    
    #isolate the dependent and the prediction features
    
y = ss["sm_li"]
X = ss[["income", "education", "parent", "marital", "age", "female"]]
    #train the model on all the data
    
lr = LogisticRegression(class_weight = "balanced")
lr.fit(X, y)


    #new data
f1 = st.selectbox("Select your income level:", 
            options = ["Less than $10,000",
                        "$10,000 to less than $20,000",
                        "$20,000 to less than $30,000",
                        "$30,000 to less than $40,000",
                        "$40,000 to less than $50,000",
                        "$50,000 to less than $75,000",
                        "$75,000 to less than $100,000",
                        "$100,000 to less than $150,000",
                        "$150,000 or more"])

if f1 == "Less than $10,000":
    f1 = 1
elif f1 == "$10,000 to less than $20,000":
    f1 = 2
elif f1 == "$20,000 to less than $30,000":
    f1 = 3
elif f1 == "$30,000 to less than $40,000":
    f1 = 4
elif f1 == "$40,000 to less than $50,000":
    f1 = 5
elif f1 == "$50,000 to less than $75,000":
    f1 = 6
elif f1 == "$75,000 to less than $100,000":
    f1 = 7
elif f1 == "$100,000 to less than $150,000":
    f1 = 8
else:
    f1 = 9



f2 = st.selectbox("Select your highest level of education:",
            options = ["Less than highschool",
                        "Did not finish highschool",
                        "Graduated high school",
                        "Some college, no degree",
                        "Two-year associates degree from college or university",
                        "Four-year college or university degree / Bachelor's degree",
                        "Some postgraduate schooling, no postgraduate degreee"
                        "Postgraduate or professional degree, including master's, doctorate, medical, or law"])

if f2 == "Less than highschool":
    f2 = 1
elif f2 == "Did not finish highschool":
    f2 = 2
elif f2 == "Graduated high school":
    f2 = 3
elif f2 ==  "Some college, no degree":
    f2 = 4
elif f2 == "Two-year associates degree from college or university":
    f2 = 5
elif f2 == "Four-year college or university degree / Bachelor's degree":
    f2 = 6
elif f2 == "Some postgraduate schooling, no postgraduate degreee":
    f2 = 7
else: 
    f2 = 8


f3 = st.selectbox("Are you a parent with a child under 18 living with you?:", ["Yes", "No"])
f4 = st.selectbox("Are you married?", ["Yes", "No"])
f5 = st.number_input("Enter age here:", min_value =  18, max_value=98, value= 45)
f6 = st.selectbox("Are you female?", ["Yes", "No"])

#Change Yes/No to 1/0

f3 = 1 if f3 == "Yes" else 0
f4 = 1 if f4 == "Yes" else 0
f6 = 1 if f6 == "Yes" else 0



newdata = pd.DataFrame({
        "income": [f1],
        "education": [f2],
        "parent": [f3],
        "marital": [f4],
        "age": [f5],
        "female": [f6]
    })

#prediction function
def predictions(newdata):
    
    
    category = lr.predict(newdata)
    probs = lr.predict_proba(newdata)
    

    return probs, category

    
    



#create a prediction button
if st.button ("Predict"):
    probs, category = predictions(newdata)
    cat_str = "Yes" if category == 1 else "No"

    st.write(f"Is this person a predicted LinkedIn user? {cat_str}" )
    st.write(f"The probability this person is a LinkedIn user is {round(probs[0][1],2)}. ")
    st.write(f"The probability they are not one is {round (probs[0][0],2)}.")




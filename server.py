import streamlit as st
import pickle
import numpy as np


# method to load the saved model
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# load the model
data = load_model()

#this will help to encorde the string inputs to binary 
linear_reg = data["model"]
le_Gender = data["le_Gender"]
le_Married = data["le_Married"]
# le_Dependents = data["le_Dependents"]
le_Education = data["le_Education"]
le_Self_Employed = data["le_Self_Employed"]


# method to predict loan eligebility
def main():
    st.title("Loan Eligibility Classification")
    
    st.write(""" ### We need some information to classify the loan Eligibility """)
    
    Gender = (
        "Male",
        "Female"
    )
    
    Married = (
        "Yes",
        "No"
    )
    
    Dependants = (
        "0",
        "1",
        "2",
        "3"
    )
    
    Education = (
        "Graduate",
        "Not Graduate"
    )
    
    Self_Employed = (
        "Yes",
        "No"
    )
    
    Property_Area = (
        "Rural",
        "Urban",
        "Semiurban"
    )
    
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        gender = st.selectbox("Gender", Gender)
        married_status = st.selectbox("Are you Married", Married)
        self_Employed = st.selectbox("Self Employed status", Self_Employed)
        property_Area = st.selectbox("Your property area", Property_Area)
        loanamountterm= st.number_input('Loan Amount term', 0, 400)
        credithistory= st.number_input('Credit history', 0, 1)
    with col2:
        dependants = st.selectbox("How many dependants", Dependants)
        education = st.selectbox("Education status", Education)
        applicantIncome= st.number_input('Applicant income', 0, 50000)
        coapplicantIncome= st.number_input('Co Applicant income', 0, 50000)
        loanamount = st.number_input('Loan Amount', 0, 50000)
        
    # action the button to start the prediction
    ok = st.button("Classify Loan Eligibility")
    if ok:
        # convert the property area to number, since models dont accept string 
        if property_Area == 'Urban':
            property_Area = 0
        if property_Area == 'Rural':
            property_Area = 2
        if property_Area == 'Semiurban':
            property_Area = 1

        X = np.array([[gender, married_status, dependants, education, self_Employed, applicantIncome, coapplicantIncome, loanamount, loanamountterm, credithistory, property_Area ]])
        
        # here the encoding is done from strings to the format that the model accepts 
        X[:, 0] = le_Gender.transform(X[:,0])
        X[:, 1] = le_Married.transform(X[:,1])
        # X[:, 2] = le_Dependents.transform(X[:,2])
        X[:, 3] = le_Education.transform(X[:, 3])
        X[:, 4] = le_Self_Employed.transform(X[:, 4])
        X = X.astype(float)
        
        # we use the linear regression model to predict
        loan_eligebility = linear_reg.predict(X)
        status = np.round(loan_eligebility[0])
        
        #translate the prediction result 
        if status == 1.0:
            status = 'Eligible'
        else:
            status = 'Ineligible'
        st.subheader(f"You are ** {status} ** for the loan")
main()

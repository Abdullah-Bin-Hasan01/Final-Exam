import pandas as pd
import numpy as np
import pickle
import gradio as gr

# load the model 
with open('final_model.pkl','rb') as file:
    model = pickle.load(file)
    
#main logic

''' Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area
 '''
COLUMNS = ["Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area"]

def predict_loan(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area):
    # Ensure correct types
    input_df = pd.DataFrame({
        "Gender":[Gender],
        "Married":[Married],
        "Dependents":[Dependents],
        "Education":[Education],
        "Self_Employed":[Self_Employed],
        "ApplicantIncome":[int(ApplicantIncome)],
        "CoapplicantIncome":[int(CoapplicantIncome)],
        "LoanAmount":[int(LoanAmount)],
        "Loan_Amount_Term":[int(Loan_Amount_Term)],
        "Credit_History":[int(Credit_History)],
        "Property_Area":[Property_Area],
    })[COLUMNS]  # enforce column order

    # Feed the entire pipeline
    pred = model.predict(input_df)[0]  
    return f'Predicted Loan Status: {pred}'


# inputs 

''' 
Gender:
['Male' 'Female' nan]

Married:
['No' 'Yes' nan]

Education:
['Graduate' 'Not Graduate']

Self_Employed:
['No' 'Yes' nan]

Property_Area:
['Urban' 'Rural' 'Semiurban']

Loan_Status:
['Y' 'N']
'''

inputs = [


     gr.Dropdown(choices=['Male','Female'], label='Enter your Gender : '),

     gr.Dropdown(choices=['Yes','No'], label='Enter your Married Status : '),

     gr.Dropdown(choices=['0','1','2','3+'], label='Enter your Dependents : '),

     gr.Dropdown(choices=['Graduate','Not Graduate'], label='Enter your Education : '),

     gr.Dropdown(choices=['Yes','No'], label='Are you Self_Employed ? :  '),

     gr.Number(label='Enter ApplicantIncome : ', precision=0),

     gr.Number(label='Enter CoapplicantIncome : ', precision=0),

     gr.Number(label='Enter LoanAmount : ', precision=0),

     gr.Number(label='Enter Loan_Amount_Term : ', precision=0),

     gr.Number(label='Enter Credit_History : ', precision=0),

     gr.Dropdown(choices=['Urban','Rural','Semiurban'], label='Enter your Property_Area : ')
]


#interface
app = gr.Interface(
    fn=predict_loan,
    inputs=inputs,
    outputs='text',
    title='Predicted Loan Status ( Y = Yes, N = No)'
)
#launch
app.launch()

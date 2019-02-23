import pandas as pd
import sys

################# RECEIVING INPUT DATA WHICH IS TO BE PREDICTED ##########
if(len(sys.argv) != 2):
    print("USGAE--> python Prediction_File.py <PATH-OF-EXCEL-FILE>")
    exit()
##########################################################################
    

################# DESERIALIZING PREVIOUSLY CREATED MODEL ##################
import pickle
file = open('BANK_TRANSACTION_USING_NAIVE_BAYES.pkl', 'rb')
clf = pickle.load(file)
file.close()
##########################################################################


################ READING THE INPUT FILE ##############################
try:
    data = pd.read_excel(sys.argv[1])   
except:
    print("Error in Loading The Excel File. Recheck the Path Provided")
    
########################################################################


############### PREPROCESSING THE TEXT #######################
def preprocess_text(df,column):
    import re
    for i in range(len(df)):
        ######  REMOVING SPECIAL CHARACTERS
        df.loc[i,column]  = re.sub(r'\W',' ',str(df.loc[i,column]))
    
        ######  REMOVING ALL SINGLE CHARACTERS
        df.loc[i,column]  = re.sub(r'\s+[a-zA-Z]\s+',' ',str(df.loc[i,column]))
    
        ######  REMOVING MULTIPLE SPACES WITH SINGLE SPACE
        df.loc[i,column]  = re.sub(r'\s+',' ',str(df.loc[i,column]))
        
    return df


input_column  = 'Particulars'
output_column = 'Model Level Classification'
data = preprocess_text(data,input_column)
###############################################################

################# PERFORMING BAG OF WORDS #############
X = data.loc[:,input_column]
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_counts   = count_vect.fit_transform(X).toarray()


######################################################

################ PERFORMING PREDICTION ###############
predict = clf.predict(X_counts)
data[output_column] = predict
data.to_excel('OUTPUTFILE.xlsx')
print('Model Prediction Successful. Please view file OUTPUTFILE.xlsx for output')
######################################################













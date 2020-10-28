import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

heart = pd.read_csv('dataset.csv')

X = heart.drop(['target'], axis=1).values
y = heart['target'].values

scale = StandardScaler()
X = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size = 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)

model = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state = 2)
model.fit(X_train, y_train)

print(model.score(X_train, y_train))

pickle.dump(model,open('final_model', 'wb'))

page_bg_img = '''
<style>
body {
background-image: url("https://image.freepik.com/free-vector/watercolour-background-with-light-blue-stains_23-2148525000.jpg");
background-size: cover;
}
</style>
'''


st.markdown(page_bg_img, unsafe_allow_html=True)

html_temp = """
    <div style="background-color:#FF0000  ;padding:10px">
    <h1 style="color:white;text-align:center;">Heart Failure Prediction</h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.write("""**1. Select Age :**""")
age = int(st.slider('', 0, 100, 50))
st.write("""**You selected this option **""", age)


st.write("""**2. Select Gender :**""")
sex = int(st.selectbox("(1=Male, 0=Female)", ["1", "0"]))
st.write("""**You selected this option **""", sex)

st.write("""**3. Select Chest Pain Type :**""")
cp = int(st.selectbox("(0=Typical Angina, 1=Atypical Angina, 2=Non-anginal Pain, 3=Asymptomatic)", ["0", "1", "2", "3"]))
st.write("""**You selected this option **""", cp)

st.write("""**4. Select Resting Blood Pressure :**""")
trestbps = int(st.slider('In mm/Hg unit', 0, 200, 100))
st.write("""**You selected this option **""", trestbps)

st.write("""**5. Select Serum Cholestrol :**""")
chol = int(st.slider('In mg/dl unit', 0, 600, 300))
st.write("""**You selected this option **""", chol)

st.write("""**6. Fasting Blood Sugar > 120 mg/dl :**""")
fbs = int(st.selectbox("(0=True, 1=False)", ["0", "1"]))
st.write("""**You selected this option **""", fbs)

st.write("""**7. Resting Electrocardiographic Results :**""")
restecg = int(st.selectbox("(0=Normal, 1=Having ST-T, 2=Hypertrophy)", ["0", "1", "2"]))
st.write("""**You selected this option **""", cp)

st.write("""**8. Maximum Heart Rate Achieved :**""")
thalach = int(st.slider('', 0, 220, 110))
st.write("""**You selected this option **""", thalach)

st.write("""**9. Pain In Chest While Exercise :**""")
exang = int(st.selectbox("(1=Yes, 0=No)", ["1", "0"]))
st.write("""**You selected this option **""", exang)

st.write("""**10. ST Depression Induced By Exercise Relative To Rest :**""")
oldpeak = float(st.slider('', 0.0, 10.0, 5.0))
st.write("""**You selected this option **""", oldpeak)

st.write("""**11. Slope Of Peak Exercise ST Segment :**""")
slope = int(st.selectbox("(0=Upsloping, 1=Flat, 2=Downsloping)", ["0", "1", "2"]))
st.write("""**You selected this option **""", slope)

st.write("""**12. Number Of Major Vessels Colored By Fluorosopy :**""")
ca = int(st.selectbox("", ["0", "1", "2", "3"]))
st.write("""**You selected this option **""", ca)

st.write("""**13. Thal :**""")
thal = int(st.slider('3=Normal; 6=Fixed Defect; 7=Reversible Defect', 0, 7, 3))
st.write("""**You selected this option **""", thal)

safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Your heart is safe</h2>
       </div>
    """
danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your heart is in danger</h2>
       </div>
    """

lst = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
df = pd.DataFrame(lst, columns=['Age','Sex','Cp','Trestbps','Chol','Fbs','Restecg','Thalach','Exang','Oldpeak','Slope','Ca','Thal'])

prediction = model.predict_proba(df)[:, 1]
predict = np.round(prediction, 2)
pred = int(predict*100)

if st.button("Predict"):
    st.success('The probability of heart failure is {}%'.format(pred))
    if pred > 50.0:
        st.markdown(danger_html,unsafe_allow_html=True)
    else:
        st.markdown(safe_html,unsafe_allow_html=True)
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def preprocess_datetime_columns(solar):
    l = solar.columns.str.contains("date", case=False, regex=True)
    col = solar.columns
    for i in range(len(l)):
        if l[i]:
            if pd.api.types.is_datetime64_ns_dtype(solar[col[i]]):
                continue
            else:
                # Add %S to the format to handle seconds (":00")
                solar[col[i]] = pd.to_datetime(solar[col[i]], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    m, dom, h, mini = [], [], [], []
    for i in solar.columns:
        if pd.api.types.is_datetime64_ns_dtype(solar[i]):
            # Use dt accessor to extract components
            solar['MONTH'] = solar[i].dt.month
            solar['MONTH_DAY'] = solar[i].dt.day
            solar['HOUR'] = solar[i].dt.hour
            solar['MINUTE'] = solar[i].dt.minute

            # Drop the original datetime column
            solar = solar.drop(columns=[i])

    return solar


def convert_categorical_to_dummies(data):
        for col in data.columns:
            if data[col].dtype == "object":
                categories = data[col].unique()
                if len(categories) < len(data):
                    data = pd.get_dummies(data, columns=[col])
                else:
                    continue

        for col in data.columns:
            if data[col].dtype == "bool":
                data[col] = data[col].astype(int)
        return data
    

def standard_scalar(columns, solar):
        scaler = StandardScaler()
        for col in columns:
            if solar[col].dtype == object:
                solar[col] = pd.to_numeric(solar[col], errors='coerce')

            if solar[col].dtype in [int, float]:
                solar[col] = scaler.fit_transform(solar[col].values.reshape(-1, 1))
        return solar

def feature_reduction(x_train, y_train):
    rf_classifier = RandomForestClassifier(random_state = 42)
    param_grid = {"max_features":list(range(1, len(x_train.columns) + 1))}
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1, error_score="raise")
    grid_search.fit(x_train, y_train)
    best_max_features = grid_search.best_params_['max_features']
    best_rf_classifier = RandomForestClassifier(max_features=best_max_features, random_state=42)
    best_rf_classifier.fit(x_train, y_train)
    feature_importance = best_rf_classifier.feature_importances_
    feature_names = x_train.columns
    df_feature_importance = pd.DataFrame(data={'Feature': feature_names, 'Importance': feature_importance})
    df_feature_importance['Importance'] = round(df_feature_importance['Importance'], 2)
    df_feature_importance = df_feature_importance.sort_values(by="Importance", ascending=False)
    return df_feature_importance

def cross_val(model, x, y):
    k_fold = 10
    kf = KFold(n_splits= k_fold, shuffle= True, random_state= 42)
    scores = cross_val_score(model, x, y, cv = kf, scoring = 'accuracy', error_score= 'raise')
    st.write(scores.mean())

def Naive_Bayes(x_train, y_train, x_test, y_test):
    nb_classifier = GaussianNB()
    nb_classifier.fit(x_train, y_train)
    y_pred = nb_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def decision_tree_class(x_train, y_train, x_test, y_test):
    dtr = DecisionTreeClassifier()
    dtr.fit(x_train, y_train)
    y_pred = dtr.predict(x_test)
    clrep = classification_report(y_test, y_pred, output_dict = True)
    return y_pred
    
def logistic_regression(x_train, y_train, x_test, y_test):
    logi = LogisticRegression()
    logi.fit(x_train, y_train)
    y_pred = logi.predict(x_test)
    clrep = classification_report(y_test, y_pred, output_dict = True)
    return y_pred

def AdaBoostClass(x_train, y_train, x_test, y_test):
    ada = AdaBoostClassifier(n_estimators=50, random_state=44)
    ada.fit(x_train, y_train)
    y_pred = ada.predict(x_test)
    clrep = classification_report(y_test, y_pred, output_dict = True)
    return y_pred

def pred_logistic_reg(x_train, y_train, data):
    logi = LogisticRegression()
    logi.fit(x_train, y_train)
    y_pred = logi.predict(data)
    return y_pred

def pred_decision_tree_class(x_train, y_train, data):
    dtr = DecisionTreeClassifier()
    dtr.fit(x_train, y_train)
    y_pred = dtr.predict(data)
    return y_pred

def pred_AdaBoostClass(x_train, y_train, data):
    ada = AdaBoostClassifier(n_estimators=50, random_state=44)
    ada.fit(x_train, y_train)
    y_pred = ada.predict(data)
    return y_pred

def confusion_mat(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot= True, cmap= "Blues")
    plt.xlabel("Predicted labels", fontsize=14)
    plt.ylabel("True labels", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    st.pyplot(plt)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    st.write("ACCURACY: ",accuracy)

def check_file_columns(df, defined_columns):
    # Read the uploaded file
    
    # Get the column names from the uploaded file
    file_column_names = df.columns.tolist()
    
    # Check if all defined column names are present in the file
    missing_columns = [col for col in defined_columns if col not in file_column_names]
    
    if missing_columns:
        st.write("The file does not contain the following columns:", missing_columns)
        return 0
    else:
        st.write("The file contains all defined columns.")
        return 1

def main():
    st.title("Solar Data Processor WebApp")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="1")

    if uploaded_file is not None:
        solar_data = pd.read_csv(uploaded_file)
        df = solar_data.copy()
        st.subheader("Declaring Target Variable")
        target = st.selectbox("Select Tagret variable", df.columns)

        st.subheader("Original Data:")
        st.write(solar_data)

        # Perform preprocessing using SolarDataProcessor
        solar_data = preprocess_datetime_columns(solar_data)

        # Get the columns to be standardized from user input
        columns_to_standardize = st.multiselect("Select columns to be standardized:", solar_data.columns)

        # Perform standardization
        solar_data = standard_scalar(columns_to_standardize, solar_data)
        solar_data = convert_categorical_to_dummies(solar_data)
        st.subheader("Processed Data:")
        st.write(solar_data)

        # Add the train-test split code snippet here
        st.markdown("---")

        st.subheader("EDA( Exploration Data Analysis )")

        if st.button("Show Correlation Heatmap (Processed Data)"):
            power_sensor = solar_data.drop(columns=[target])
            corr = power_sensor.corr(method='spearman')
            fig = px.imshow(corr)
            st.plotly_chart(fig)
        
        X = st.selectbox("X-axis", solar_data.drop(columns=[target]).columns)
        Y = st.selectbox("Y-axis", solar_data.drop(columns=[target, X]).columns)

        plot_options = ["Line Plot","Scatter Plot"]
        selected_plot = st.selectbox('Select plot to show:', plot_options)
        if selected_plot == "Scatter Plot":
            # fig, ax = plt.subplots(figsize=(15, 8))
            # ax.Scatter(solar_data[X], solar_data[Y])
            # df = pd.DataFrame([solar_data[X], solar_data[Y]], columns = [X, Y])
            # st.line_chart(df)
            fig, ax = plt.subplots(figsize=(15, 8))
            solar_data.plot(x=X, y=Y, style='.', ax=ax)
            st.pyplot(fig)
        
        elif selected_plot == "Line Plot":
            fig, ax = plt.subplots(figsize=(15, 8))
            sns.lineplot(x = solar_data[X], y=solar_data[Y], ax=ax)
            st.pyplot(fig)

        st.markdown("---")
        
        x_train, x_test, y_train, y_test = train_test_split(solar_data.drop(columns = [target], inplace = False), solar_data[target], test_size=0.20, random_state=88)
        st.subheader("Display Important Features:")
        df = feature_reduction(x_train, y_train)
        st.write(df)
        st.write("Bar Plot:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Feature', y='Importance', data=df)
        plt.xticks(rotation=90)
        st.pyplot(fig)
        st.markdown("---")

        st.subheader("Machine Learning")
        selected_features = st.multiselect("Select columns on which you want to train the model.", solar_data.drop(columns = [target]).columns)
        x_train, x_test, y_train, y_test = train_test_split(solar_data[selected_features], solar_data[target], test_size = 0.2, random_state= 42)

        # Add the machine learning model selection and implementation code here
        model_options = ['Logistic Regression', 'Decision Tree Classification', 'Ada Boost Classification']
        selected_model = st.selectbox('Select a machine learning model:', model_options)

        if selected_model == 'Logistic Regression':
            if st.button("Run Logistic Regression" ,key = "logistic reg"):
                y_pred_lr = logistic_regression(x_train, y_train, x_test, y_test)
                # st.write("Accuracy: ", r2_score)
                confusion_mat(y_test, y_pred_lr)
            
            if st.button("Cross-Validation", key = "cv"):
                model = LogisticRegression()
                cross_val(model, solar_data[selected_features], solar_data[target])

        elif selected_model == 'Decision Tree Classification':
            if st.button("Run Decision Tree Classification" , key="Decision Tree"):
                y_pred_dtr = decision_tree_class(x_train, y_train, x_test, y_test)
                confusion_mat(y_test, y_pred_dtr)
            
            if st.button("Cross-Validation", key = "cv"):
                model = DecisionTreeClassifier()
                cross_val(model, solar_data[selected_features], solar_data[target])
        
        elif selected_model == 'Ada Boost Classification':
            if st.button("Run Ada Boost Classifier", key = "Ada Boost"):
                y_pred_ada = AdaBoostClass(x_train, y_train, x_test, y_test)
                confusion_mat(y_test, y_pred_ada)
            
            if st.button("Cross-Validation", key = "cv"):
                model = AdaBoostClassifier()
                cross_val(model, solar_data[selected_features], solar_data[target])

        pred_option = ["Single Value Prediction", "DataSet Prediction"]
        r = st.radio("Select one option", pred_option)   
        if r == "Single Value Prediction":     
            st.subheader("Single Value Prediction")
            a = []
            st.write("NOTE: \n Enter the value in the same format as that in the dataset")
            for i in selected_features:
                b = st.number_input("{}".format(i), min_value= None, max_value= None)
                a.append(b)
            b = [a]
            data = pd.DataFrame(b, columns = [selected_features])
            d = standard_scalar(data.columns, data)
            st.write(data)
            st.markdown("#### Prediction: ")
            if selected_model == "Logistic Regression":
                prediction = pred_logistic_reg(x_train, y_train, data)  
                st.write(prediction)
            
            elif selected_model == "Decision Tree Classification":
                prediction = pred_decision_tree_class(x_train, y_train, data)  
                st.write(prediction)
            
            elif selected_model == "Ada Boost Classification":
                prediction = pred_AdaBoostClass(x_train, y_train, data)
                st.write(prediction)

        elif r == "DataSet Prediction":
            st.subheader("Dataset Prediction")
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            df = pd.read_csv(uploaded_file)
            df = preprocess_datetime_columns(df)
            st.write(df.head())
    
            if uploaded_file is not None:
                # Check if the uploaded file contains defined column names
                check = check_file_columns(df, selected_features)
                if check == 0:
                    st.write("Please upload a file with the required columns.")
                
                if check == 1:
                    st.markdown("#### Prediction: ")

                    if selected_model == "Logistic Regression":
                        prediction = pred_logistic_reg(x_train, y_train, df[selected_features])
                        st.write(prediction)
            
                    elif selected_model == "Decision Tree Classification":
                        prediction = pred_decision_tree_class(x_train, y_train, df[selected_features])
                        st.write(prediction)

                    elif selected_model == "Ada Boost Classification":
                        prediction = pred_AdaBoostClass(x_train, y_train, df[selected_features])
                        st.write(prediction)
                            
if __name__ == "__main__":
    main()
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    ''' Read file '''
    f = pd.read_csv('BankChurners.csv')

    ''' Drop useless columns '''
    f = f.drop(columns=['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2', 'CLIENTNUM'], axis=1)

    ''' Show percentage of existing and attrited customers '''
    print(f['Attrition_Flag'].value_counts(normalize=True))
    print(f.Income_Category.value_counts(normalize=True))

    ''' Plot '''
    f.groupby(['Education_Level', 'Attrition_Flag']).size().unstack().plot(kind='bar', stacked=True)
    f.groupby(['Card_Category', 'Attrition_Flag']).size().unstack().plot(kind='bar', stacked=True)
    f.groupby(['Income_Category', 'Attrition_Flag']).size().unstack().plot(kind='bar', stacked=True)
    f.groupby(['Marital_Status', 'Attrition_Flag']).size().unstack().plot(kind='bar', stacked=True)
    plt.show()

    ''' Encode data '''
    le = LabelEncoder()
    f['Attrition_Flag'] = le.fit_transform(f['Attrition_Flag'])
    f['Gender'] = le.fit_transform(f['Gender'])
    f['Marital_Status'] = le.fit_transform(f['Marital_Status'])
    f['Card_Category'] = le.fit_transform(f['Card_Category'])
    f['Education_Level'] = le.fit_transform(f['Education_Level'])
    f['Income_Category'] = le.fit_transform(f['Income_Category'])

    ''' Show importance of features with Pearson Correlation '''
    plt.figure()
    cor = f.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

    ''' X & Y '''
    X = f.drop(columns=['Attrition_Flag'])
    Y = f['Attrition_Flag']

    ''' train & test data '''
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=.3)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    ''' Model train '''
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    rf.fit(train_x, train_y)
    pred = rf.predict(test_x)
    print('Accuracy:')
    print(accuracy_score(test_y, pred))



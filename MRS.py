import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
music_data = pd.read_csv('music.csv')
X= music_data.drop(columns=['genre'])
y = music_data['genre']
model = DecisionTreeClassifier()
model.fit(X, y)
predictions= model.predict([ [21,1], [22, 0]  ])
predictions
joblib.dump(model ,)

output 


array(['HipHop', 'Dance'], dtype=object)

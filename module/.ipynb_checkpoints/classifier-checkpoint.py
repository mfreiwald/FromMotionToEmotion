from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

rf10 = RandomForestClassifier(
        n_estimators=800, 
        max_features="auto", 
        max_depth=30, 
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=False,
        n_jobs=-1, 
        random_state=1)

rf5 = RandomForestClassifier(
        n_estimators=1800, 
        max_features="auto", 
        max_depth=40, 
        min_samples_split=5,
        min_samples_leaf=1,
        bootstrap=True,
        n_jobs=-1, 
        random_state=1)

rf1 = RandomForestClassifier(
        n_estimators=400,
        max_features="auto",
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        bootstrap=True,
        n_jobs=-1, 
        random_state=1)

dummyM = DummyClassifier(strategy="most_frequent", random_state=1)
dummyS = DummyClassifier(strategy="stratified", random_state=1)
dummyU = DummyClassifier(strategy="uniform", random_state=1)
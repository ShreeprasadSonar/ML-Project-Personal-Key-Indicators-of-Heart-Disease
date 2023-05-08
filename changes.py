# iv) XGBoost

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class XGBoostRegressor:
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.trees = []

    def _log_odds(self, p):
        p = np.clip(p, 1e-5, 1 - 1e-5)
        return np.log(p / (1 - p))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        y = (y == y.unique()[0]).astype(int)
        pred = np.full(y.shape, y.mean())
        for i in range(self.n_estimators):
            grad = y - self._sigmoid(self._log_odds(pred))
            tree = self._fit_tree(X, grad)
            pred += self.learning_rate * self._predict_tree(X, tree)
            self.trees.append(tree)

    def predict_proba(self, X):
        pred = np.full(X.shape[0], 0.5)
        for tree in self.trees:
            pred += self.learning_rate * self._predict_tree(X, tree)
        return np.vstack([1 - pred, pred]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def _get_split(self, X, grad):
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        for feature in X.columns:
            values = X[feature].unique()
            for threshold in values:
                left = grad[X[feature] < threshold].mean()
                right = grad[X[feature] >= threshold].mean()
                gain = (left ** 2 + right ** 2) / 2 - ((left + right) / 2) ** 2
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _fit_tree(self, X, grad, depth=1):
        if depth > self.max_depth or len(X) < 2:
            return self._log_odds(grad.mean())
        feature, threshold = self._get_split(X, grad)
        tree = {
            'feature': feature,
            'threshold': threshold,
            'left': self._fit_tree(X[X[feature] < threshold], grad[X[feature] < threshold], depth + 1),
            'right': self._fit_tree(X[X[feature] >= threshold], grad[X[feature] >= threshold], depth + 1),
        }
        return tree

    def _predict_tree(self, X, tree):
        if isinstance(tree, float):
            return tree
        return np.where(X[tree['feature']] < tree['threshold'],
                        self._predict_tree(X, tree['left']),
                        self._predict_tree(X, tree['right']))
        
def XGBoostClassifier(Xtrn, ytrn, Xtst, ytst, n_estimators=100, max_depth=6):
    print("start XGBoostRegressor")
    xg= XGBoostRegressor(max_depth=max_depth, n_estimators=n_estimators)
    xg.fit(Xtrn, ytrn)
    print("fit XGBoostRegressor done")
    y_pred=xg.predict(Xtst)
    
    tst_acc = accuracy_score(ytst, y_pred)
    tst_precision, tst_recall, tst_f1, support  = precision_recall_fscore_support(ytst, y_pred, average='weighted')
    print(f"Accuracy: {tst_acc}")
    print(f"Precision: {tst_precision}")
    print(f"Recall: {tst_recall}")
    print(f"F1: {tst_f1}")
    
    xgb = XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.1)
    print("fit XGBClassifier done")
    xgb.fit(Xtrn, ytrn)

    y_pred_xgb = xgb.predict(Xtst)

    tst_acc = accuracy_score(ytst, y_pred_xgb)
    tst_precision, tst_recall, tst_f1, support  = precision_recall_fscore_support(ytst, y_pred_xgb, average='weighted')
    print(f"Accuracy: {tst_acc}")
    print(f"Precision: {tst_precision}")
    print(f"Recall: {tst_recall}")
    print(f"F1: {tst_f1}")
    
#XGBoostClassifier(X_train, y_train, X_test, y_test) 




print(len(df_normalized.columns))

features_to_drop = ["HeartDisease", "BMI", "AlcoholDrinking", "MentalHealth", "GenHealth", "SleepTime", "Asthma", "Sex_Female", "Sex_Male",
       "Race_American Indian/Alaskan Native", "Race_Asian", "Race_Black",
       "Race_Hispanic", "Race_Other", "Race_White"]
print(len(features_to_drop))
X = df_normalized.drop(features_to_drop, axis=1)
print(X.shape)
print(X.columns)
y = df_normalized["HeartDisease"]

X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X, y, test_size=0.2, random_state=21)
print(X_train_sel.shape)


LogisticRegressionClassifier(X_train_sel, y_train_sel, X_test_sel, y_test_sel, 0.1, 100, fold=1, sample=1)
print(Custom_LR_Training_Accuracies[1][1])
print(Custom_LR_Testing_Accuracies[1][1])
print(Sklearn_LR_Training_Accuracies[1][1])
print(Sklearn_LR_Testing_Accuracies[1][1])


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train_sel, y_train_sel)
dtc.score(X_test_sel, y_test_sel)


dt2 = DecisionTreeClassifier()
dt2.fit(X_train, y_train)
dt2.score(X_test, y_test)


NaiveBayesClassifier(X_train_sel, y_train_sel, X_test_sel, y_test_sel, 1, 1)


print(Custom_NB_Training_Accuracies[1][1])
print(Custom_NB_Testing_Accuracies[1][1])
print(Sklearn_NB_Training_Accuracies[1][1])
print(Sklearn_NB_Testing_Accuracies[1][1])


XGBoostClassifier(X_train_sel, y_train_sel, X_test_sel, y_test_sel, n_estimators=100, max_depth=6)


XGBoostClassifier(X_train, y_train, X_test, y_test, n_estimators=5, max_depth=3)



k1 = KNeighborsClassifier()
k1.fit(X_train_sel, y_train_sel)
k1.score(X_test_sel, y_test_sel)


k2 = KNeighborsClassifier()
k2.fit(X_train, y_train)
k2.score(X_test, y_test)


from sklearn.model_selection import RandomizedSearchCV

# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]}

np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)
# CV = 5, for 5 fold cross validation as shown in image above, can use any value
# n_iter = 20, for 20 different combinations np.logspace(-4, 4, '20')

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train, y_train)


rs_log_reg.best_params_


rs_log_reg.score(X_test, y_test)



from sklearn.ensemble import RandomForestClassifier
# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": [10, 20, 50, 100],
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}

rs_rf = RandomizedSearchCV(RandomForestClassifier(), 
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model for RandomForestClassifier()
rs_rf.fit(X_train, y_train)


rs_rf.best_params_


rs_rf.score(X_test, y_test)


plt.bar(["CustomLR", "<CustomLR>", "SklearnLR", "<SklearnLR>", "SklearnDTree", "<SklearnDTree>", "CustomNB",
"<CustomNB>", "SklearnNB", "<SklearnNB>", "CustomXGBoost", "<CustomXGBoost>", "SklearnXGBoost", "<SklearnXGBoost>", "SklearnKNN",
        "<SklearnKNN>"],
        [91.347, 91.34758, 91.4210, 91.407, 86.499, 91.02862, 91.4085, 91.39605, 83.6692, 85.210838, 
         91.33, 91.3444, 91.42, 91.438, 90.4438781, 90.76908], 
        color=["lightblue", "salmon"])
plt.title("Comparison of test accuracy of various algos before and after feature selection")
plt.ylabel("Accuracy in percentage")
plt.ylim([80,100])
plt.yticks([80,82.5,85,87.5, 90,92.5, 95,97.5,100])
plt.xticks(rotation=90)
plt.text(1, 70, "Note: <model> indicates performance after feature selection", fontsize=12)


# Data
custom_boosting_data = {
    (1, 2): 8.6524,
    (1, 3): 8.6524,
    (2, 2): 8.6524,
    (2, 3): 8.6524,
}

sklearn_boosting_data = {
    (1, 2): 8.6524,
    (1, 3): 8.7243,
    (2, 2): 8.8135,
    (2, 3): 8.7634,
}

# Create figure and axis
fig, ax = plt.subplots()

# Plot custom boosting data
x1 = [f"depth:{depth},bag size:{bag_size}" for depth, bag_size in custom_boosting_data.keys()]
y1 = list(custom_boosting_data.values())
ax.plot(x1, y1, label="Custom Boosting", color="lightblue", marker='o')

# Plot sklearn boosting data
x2 = [f"depth:{depth},bag size:{bag_size}" for depth, bag_size in sklearn_boosting_data.keys()]
y2 = list(sklearn_boosting_data.values())
ax.plot(x2, y2, label="sklearn Boosting", color="salmon", marker='o')

# Set x-axis labels and tick positions
ax.set_xticks(range(len(x1)))
ax.set_xticklabels(x1)

# Add legend and title
ax.legend()
ax.set_title("Error Rates of Custom and Sklearn Boosting")

# Show plot
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# example confusion matrix
cm = np.array([[0, 5534], [0, 58425]])

# create ConfusionMatrixDisplay object
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])

# display the plot
disp.plot(cmap="Blues")
plt.title("Confusion Matrix for Custom Boosting [depth=1, bag_size=2]")
plt.show()



# Using custom Bagging
# Depth =  5 
# Bag Size =  3
# Error rate = 8.5148%
# Confusion Matrix = 
# [225, 5309]
# [137, 58288]

# example confusion matrix
cm = np.array([[225, 5309], [137, 58288]])

# create ConfusionMatrixDisplay object
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])

# display the plot
disp.plot(cmap="Blues")
plt.title("Confusion Matrix for Custom Bagging [depth=5, bag_size=3]")
plt.show()




import matplotlib.pyplot as plt

# custom bagging data
custom_bagging_data = [
    {"depth": 3, "bag_size": 3, "error_rate": 8.5868},
    {"depth": 3, "bag_size": 4, "error_rate": 8.5742},
    {"depth": 5, "bag_size": 3, "error_rate": 8.5148},
    {"depth": 5, "bag_size": 4, "error_rate": 8.5383}
]

# sklearn bagging data
sklearn_bagging_data = [
    {"depth": 3, "bag_size": 3, "error_rate": 8.6524},
    {"depth": 3, "bag_size": 4, "error_rate": 8.6524},
    {"depth": 5, "bag_size": 3, "error_rate": 8.5570},
    {"depth": 5, "bag_size": 4, "error_rate": 8.5414}
]

# extract the x and y values for the two datasets
x = [(f"depth:{d}, bag size:{b}") for d, b in zip([d['depth'] for d in custom_bagging_data], [d['bag_size'] for d in custom_bagging_data])]
y_custom_bagging = [d['error_rate'] for d in custom_bagging_data]
y_sklearn_bagging = [d['error_rate'] for d in sklearn_bagging_data]

# create the plot with two lines with different colors
plt.plot(x, y_custom_bagging, color='lightblue', label='Custom Bagging', marker='o')
plt.plot(x, y_sklearn_bagging, color='salmon', label='Sklearn Bagging', marker='o')

# set the labels and title of the plot
plt.xlabel('Depth and Bag Size')
plt.ylabel('Error Rate')
plt.title('Custom vs Sklearn Bagging Error Rates')

# add a legend to the plot
plt.legend()

# display the plot
plt.show()




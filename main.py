from data_loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt

scoring = {'accuracy': make_scorer(accuracy_score),
           'prec': 'precision'}

from models import SVM

dl = DataLoader(data_set_nr=1,
                samples_amount=10000,
                shuffle=True,
                compressed=False)

X, y = dl.load()

n_samples = len(X)

from image_transforms import TransformHandler
from filters import fourier

tf = TransformHandler(transform_method=fourier)

X = tf.transform_image_array(X)

X = X.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, shuffle=True)


def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}


test_model = SVM()

test_model.model_train(X_train, y_train)

cv_results = cross_validate(test_model.model, X_test, y_test, cv=5,
                            scoring=confusion_matrix_scorer, verbose=True)
# Getting the test set true positive scores
print(cv_results['test_tp'])

# Getting the test set false negative scores
print(cv_results['test_fn'])

predictions = test_model.model.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=test_model.model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=test_model.model.classes_)
disp.plot()

plt.show()

print(X.shape, y.shape)

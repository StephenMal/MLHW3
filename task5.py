import numpy as np
from sklearn.svm import SVC

# Features and Labels
X, y = [[1, 1],[2, 2],[2, 0],[0, 0],[1, 0],[0, 1]], [1, 1, 1, -1, -1, -1]
# Create the model
clf = SVC(kernel='linear', C=1000)
# Train
clf.fit(X, y)

# Get details
print(f'Support Vectors\n{clf.support_vectors_}')
print(f'Coef\n{clf.coef_}')
print(f'Acc{clf.score(X,y)}')

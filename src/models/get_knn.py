from sklearn.neighbors import KNeighborsClassifier

def get_knn(n_neighbors=3):
    return KNeighborsClassifier(n_neighbors=n_neighbors)
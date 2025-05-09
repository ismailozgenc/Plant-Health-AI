from sklearn.svm import SVC

def get_svm(kernel='rbf', gamma='scale'):
    return SVC(kernel=kernel, gamma=gamma)
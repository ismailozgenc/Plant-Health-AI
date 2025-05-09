from sklearn.linear_model import LogisticRegression

def get_lr(max_iter=1000):
    return LogisticRegression(solver='saga', max_iter=1000)
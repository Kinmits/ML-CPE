from sklearn.svm import SVC

def train_svm_models(X_train_scaled, y_train, kernels=['linear', 'poly', 'rbf']):
    """สร้างและเทรนโมเดล SVM ตาม Kernels ที่กำหนด"""
    trained_models = {}
    for kernel in kernels:
        clf = SVC(kernel=kernel, random_state=42)
        clf.fit(X_train_scaled, y_train)
        trained_models[kernel] = clf
        print(f"[MODEL TRAIN] Trained SVM with '{kernel.upper()}' kernel")
    return trained_models
folder = "./data"

nyq = get_nyquist(folder)
X_train, X_test, y_train, y_test = load(folder, subjects=2)

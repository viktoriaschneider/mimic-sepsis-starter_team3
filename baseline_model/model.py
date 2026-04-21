import numpy as np
import cloudpickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
 
def get_model():
    def medical_feature_engineering(X):
        hr = X[:, [0]]
        temp = X[:,[2]]
        sbp = X[:, [3]]
        bun = X[:, [15]]
        creat = X[:, [19]]
        o2sat = X[:, [11]]
        fio2 = X[:, [10]]
        map = X[:, [4]]
        age = X[:, [34]]
        resp = X[:, [6]]
        lactate = X[:, [22]]
        o2sat_fio2_ratio = o2sat/(fio2 + 1e-6) 
        map_age_ratio = map/(age + 1e-6)
        hr_temp = hr*temp
        resp_lactate =resp*lactate
        shock_index = hr / (sbp + 1e-6)
        bun_creat_ratio = bun / (creat + 1e-6)
        return np.hstack([X, shock_index, bun_creat_ratio, o2sat_fio2_ratio, map_age_ratio, hr_temp, resp_lactate])
 
    model = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("engineering", FunctionTransformer(medical_feature_engineering)),
            ("scaler", StandardScaler()),
            # Random Forest helyett egy Neurális Háló (MLP), ami FL-kompatibilis
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32), # Két rejtett réteg (mint a RF fái)
                max_iter=1,                  # FL miatt körönként csak egyet lépünk
                warm_start=True,             # Megtartja az előző kör tudását
                random_state=42,
                learning_rate_init=0.005 ,     # Kicsit gyorsabb tanulás
                alpha=0.01,
                batch_size=64,
                solver="adam"
            )),
        ]
    )
 
    # Pre-fit dummy adattal, hogy a súlyok (coefs_) létrejöjjenek a 40 oszlophoz
    model.fit(np.zeros((5, 40)), np.array([0, 1, 0, 1, 0]))
    return model
 
def get_model_parameters(model):
    # MLP-nél több réteg van, így a coefs_ és intercepts_ listákat fűzzük össze
    clf = model.named_steps["clf"]
    return clf.coefs_ + clf.intercepts_
 
def set_model_parameters(model, parameters):
    clf = model.named_steps["clf"]
    # A kapott paraméterlista első fele a súlyok, második az eltolások
    n_layers = len(clf.coefs_)
    clf.coefs_ = parameters[:n_layers]
    clf.intercepts_ = parameters[n_layers:]
 
def save_model(model, path="final_model.pkl"):
    with open(path, "wb") as f:
        cloudpickle.dump(model, f)
 
def load_model(path="final_model.pkl"):
    with open(path, "rb") as f:
        return cloudpickle.load(f)
from pathlib import Path
import h5py, numpy as np, faiss

DATA_DIR = Path(__file__).resolve().parent / "data"
FILES = {"sift": "sift-128-euclidean.hdf5", "glove": "glove-100-angular.hdf5"}
N_TUNE = 5000

def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return np.ascontiguousarray(x / n, dtype=np.float32)

def load(name, n_train=None):
    """Returns train, test. GloVe is L2-normalised, so Euclidean ranking == angular ranking."""
    with h5py.File(DATA_DIR / FILES[name], "r") as f:
        train = f["train"][:n_train] if n_train else f["train"][:]
        test = f["test"][:]
    train = np.ascontiguousarray(train, dtype=np.float32)
    test = np.ascontiguousarray(test, dtype=np.float32)
    if name == "glove":
        train, test = normalize(train), normalize(test)
    return train, test

def split_queries(test):
    """First 5000 = tune, last 5000 = report."""
    return test[:N_TUNE], test[N_TUNE:]

def file_neighbors(name, k=10):
    """Stored ground truth. Only valid for the FULL training set."""
    with h5py.File(DATA_DIR / FILES[name], "r") as f:
        return f["neighbors"][:, :k]

def exact_neighbors(name, train, test, k=100):
    """Exact top-k ids with FAISS flat search, cached on disk (data/ is git-ignored)."""
    path = DATA_DIR / f"gt_{name}_{len(train)}.npz"
    if path.exists():
        return np.load(path)["I"]
    index = faiss.IndexFlatL2(train.shape[1])
    index.add(train)
    _, I = index.search(test, k)
    np.savez(path, I=I)
    return I
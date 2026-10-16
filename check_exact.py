import data, metrics
for name in ("sift", "glove"):
    train, test = data.load(name)
    I = data.exact_neighbors(name, train, test, k=100)
    kd = metrics.kth_dist2(test, train, I, 10)
    r = metrics.recall_at_k(data.file_neighbors(name, 10), test, train, kd, k=10)
    print(name, train.shape, "recall of file neighbours vs my exact:", r)
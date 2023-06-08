from data_loader import DataLoader

dl = DataLoader(data_set_nr=1,
                samples_amount=20,
                shuffle=True,
                compressed=False)

X, y = dl.load()
print(X.shape, y.shape)

import pandas as pd 
import numpy as np 


class KNN:
    def __init__(self, k):
        self.k = k
        
    def fit (self, X, y):
        self.x_train = X 
        self.y_train = y
        
    def predict(self, X):
        predictions = [self.predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        most_common = unique[np.argmax(counts)]
        return most_common
    
    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

class GestureEngine:
    def __init__(self):
        self.labels_map = {}
        self.data_path = pd.read_csv('data.csv')
        self.data_path = 'datasets/Datafull terakhir test.csv'
        self.df = pd.read_csv(self.data_path, sep=';')
        
        # preprocessing code
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        
        label_col = self.df.columns[-1]
        coord_cols = self.df.columns[:-1]
        
        if self.df[label_col].dtype != object:
            raise ValueError("Kolom label harus bertipe karakter")
        
        expected_cols = []
        for i in range(len(coord_cols) // 2):
            expected_cols.extend([f"{i}x", f"{i}y"])
            
        if list(coord_cols) != expected_cols:
            raise ValueError("Nama atau urutan kolom koordinat tidak sesuai")

        len(coord_cols) // 2    
        
        
        self.df[coord_cols] = self.df[coord_cols].apply(pd.to_numeric, errors='coerce')
        
        # cek data missing dan duplicated
        count_isna = self.df.isna().sum().sum()
        count_duplicated = self.df.duplicated().sum()
        
        missing_value = print(f"Jumlah missing value : {count_isna} ")         
            
    # def _euclidean_distance(self, x1, x2):
    #     return np.sqrt(np.sum((x1 - x2) ** 2))
    
    
    def split_train_test(X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        return X_train, X_test, y_train, y_test
    
    def define_conture(A, self):
        X_raw = self.df.drop(["char", "char_encoded"], axis=1).values
        y = self.df["char_encoded"].values
        
        X_processed = []
        for x in X_raw:
            x = x.reshape(1, -1)
            x_prep = GestureEngine.preprocess_single_hand(x)
            X_processed.append(x_prep[0])
            
        X = np.array(X_processed)
        X_train, X_test, y_train, y_test = GestureEngine.split_train_test(
            X, y, test_size=0.2, random_state=42
        )
        
        classify = KNN(k=3)
        classify.fit(X_train, y_train)
        
        A = GestureEngine.preprocess_single_hand(A)
        pred = classify.predict(A)
        pred_label = int(pred[0])
        
        result_char = self.df.loc[self.df["char_encoded"] == pred_label, "char"].iloc[0]
        output_pred = print("Prediksi: ", result_char)
        
    def preprocess_single_hand(A):
        A = A.copy()
        coords = A.reshape(-1, 2)
        
        coords -= coords[0]
        
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist != 0:
            coords /= max_dist
            
        ref = coords[9]
        angle = np.arctan2(ref[1], ref[0])
        
        rot = np.array(
            [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
        )
        coords = coords @ rot.T
        
        return coords.flatten().reshape(1, -1)
        
        
        
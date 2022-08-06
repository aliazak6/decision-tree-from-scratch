from typing import List
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.rightChild = right
        self.threshold = threshold
        self.info_gain = info_gain
        self.leftChild = left
        self.feature_index = feature_index
        self.value = value 

class DecisionTreeClassifier():
    def __init__(self, max_depth=5):
        # Constructor methodu
        
        # Ağacın kök nodeu başlatılır.
        self.root = None
        # durma koşulları ayarlanır.
        self.max_depth = max_depth
        #get_best_split methodu için;
        self.num_features = 1
    def build_tree(self, dataset, curr_depth=0):
        #Nodeların gelişmesini sağlayan method. Recursive çalışır. 
        
        num_samples = len(dataset)
        y = []
        for row in dataset:
            y.append(row[-1])

        if num_samples> 0 and curr_depth<=self.max_depth:
            best_split = self.get_best_split(dataset)
            if "info_gain" in best_split:
                if best_split["info_gain"]>0:
                    left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                    right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                    return Node(best_split["feature_index"], best_split["threshold"], 
                                left_subtree, right_subtree, best_split["info_gain"])
            '''
            Burada dönülen node decision için kullanılır.
            best_split'te bulunan thresholddan küçük değerlere sahip veriler left datasete
            büyük değerlere sahip veriler right datasete atılır ve bunları kullanarak yeni
            subtree'ler oluşturulur.
            '''
                                
        #decision node oluşturulamadığı zaman yaprak düğümü oluşturulur
        final_value = self.calculate_leaf_value(y)
        # return leaf node
        return Node(value=final_value)

    def get_best_split(self, dataset):
        ''' function to find best split '''
        
        best_split = {}
        max_info_gain = -float("inf")
        
        for feature_index in range(self.num_features):
            possible_thresholds = []
            y = []
            for row in dataset:
                y.append(row[-1]) # labellar
                feature = (row[feature_index])   
                if feature not in possible_thresholds:  #if unique
                    possible_thresholds.append(feature) 

            for threshold in possible_thresholds:
                # Her possible thresh için dataseti ayırıp gain hesaplayarak max gaini buluruz.
                dataset_left, y_left, dataset_right, y_right = self.split(dataset, feature_index, threshold)

                if len(dataset_right)>0 and len(dataset_left)>0:
                    # Üstte yapılan işlemin aynısıyla labellar çıkarılır.                
                    curr_info_gain = self.information_gain(y, y_left, y_right)
                    # Eğer parent node'un gaini çocuklardan fazlaysa daha fazla dallanmıyoruz.
                    if not curr_info_gain <= max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["info_gain"] = curr_info_gain   
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["threshold"] = threshold
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):

        dataset_left = []
        y_left = []
        dataset_right = []
        y_right = []
        for row in dataset:
            if row[feature_index]<=threshold:
                dataset_left.append(row)
                y_left.append(row[-1])
            else:
                dataset_right.append(row)
                y_right.append(row[-1])
        return dataset_left,y_left,dataset_right,y_right
    
    def information_gain(self, parent, l_child, r_child):

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        return gain
    
    
    def gini_index(self, y):
        class_labels = []
        for el in y:
            if el not in class_labels:
                class_labels.append(el)
        gini = 0
        for cls in class_labels:
            p_cls = y.count(cls) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X: List[List[float]], y: List[int]):
        ''' Ağacın eğitildiği fonksiyon '''
        self.num_features = len(X[0])
        dataset = []
        for i in range(len(X)):
            dataset.append(X[i]+[y[i]]) # np.concat işlemi
        self.root = self.build_tree(dataset)
    
    def predict(self, X: List[List[float]]):        
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x, tree):
        
        if tree.value!=None: return tree.value # if leaf node return value
        feature_val = x[tree.feature_index]    # Değilse feature değerini thresholdla karşılaştırıp sola veya
        if feature_val<=tree.threshold:        # sağa git.
            return self.make_prediction(x, tree.leftChild)
        else:
            return self.make_prediction(x, tree.rightChild)


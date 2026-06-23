import numpy as np
from collections import Counter
class ID3(object):
    def __init__(self,feature_names):
        self.tree=None
        self.feature_names=feature_names
    def entropy(self,y):
        """
        计算熵
        :param y:当前节点的标签
        :return: 返回熵
        """
        if len(y) == 0:
            return 0
        class_counts = Counter(y)
        probs = np.array([count / len(y) for count in class_counts.values()])
        return -np.sum(probs * np.log2(probs + 1e-10))
    def conditional_entropy(self, x_col,y):
        """
        计算每个特征列的条件熵
        :param y:当前节点的标签
        :param x_col: 一个特征列
        """
        cond_entropy=0
        unique_values = np.unique(x_col)
        for value in unique_values:
            y_sub=y[x_col == value]
            entropy_sub=self.entropy(y_sub)
            w=len(y_sub)/len(y)
            cond_entropy+=w*entropy_sub
        return cond_entropy
    def information_gain(self, x_col,y):
        """
        计算每个特征列的信息增益
        :param x_col: 一个特征列
        :y:当前根节点的标签
        """
        original_entropy = self.entropy(y)
        cond_entropy = self.conditional_entropy(x_col,y)
        return original_entropy - cond_entropy

    def is_same_on_features(self, x, feature_indices):
        """
        判断所有特征列取值是不是一样
        :param x: 当前节点下所有特征列
        :param feature_indices: 当前节点下的所有特征序列
        """
        for idx in feature_indices:
            # 检查该特征列是否只有一个唯一值
            if len(np.unique(x[:, idx])) > 1:
                return False  # 只要有一个特征有不同取值，就返回False
        return True  # 所有特征都只有唯一值
    def id3Tree(self, x, y,feature_indices):
        """
        生成id3
        :param x: 当前剩余的特征矩阵
        :param y: 与x对应的y
        :param feature_indices: 与x对应的特征序列
        :return: tree
        """
        x=np.array(x)
        y=np.array(y)
        #计算当前节点的多数类
        current_majority = Counter(y).most_common(1)[0][0]
        #如果所有样本属于同一类别，返回叶子节点
        if len(np.unique(y)) == 1:
            return y[0]
        #如果没有剩余特征，或者所有特征取值相同返回多数类别
        if len(feature_indices)==0 or self.is_same_on_features(x,feature_indices):
            return Counter(y).most_common(1)[0][0]
        
        best_gain = -1#最大信息增益
        best_feature_idx = None#最大信息增益索引
        for idx in feature_indices:
            x_col=x[:,idx]
            gain = self.information_gain(x_col, y)
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = idx
        #生成树
        tree = {}
        feature_name = self.feature_names[best_feature_idx]
        tree['feature'] = feature_name
        tree['feature_idx'] = best_feature_idx
        tree['majority'] = current_majority
        tree['children'] = {}

        x_col = x[:, best_feature_idx]
        unique_values = np.unique(x_col)
        for value in unique_values:
            # 获取在该特征上取值为value的样本子集
            mask = (x_col == value)
            x_sub = x[mask]
            y_sub = y[mask]
            new_feature_indices = [idx for idx in feature_indices if idx != best_feature_idx]
            #递归调用
            tree['children'][value] = self.id3Tree(x_sub, y_sub,new_feature_indices)
        return tree
    def fit(self,x_train,y_train):
        """
        训练id3
        """
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        feature_indices = list(range(x_train.shape[1]))
        self.tree=self.id3Tree(x_train,y_train,feature_indices)
    def predict_sample(self,tree,sample):
        if not isinstance(tree, dict):
            return tree
        idx = tree['feature_idx']
        value = sample[idx]
        if value in tree['children']:
            return self.predict_sample(tree['children'][value],sample)
        else:
            return tree['majority']

    def predict(self,x_test):
        x_test = np.array(x_test)
        predictions = []
        for sample in x_test:
            pre = self.predict_sample(self.tree,sample)
            predictions.append(pre)
        return np.array(predictions)
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    data = {
        '色泽': ['青绿', '乌黑', '乌黑', '青绿', '浅白', '青绿', '乌黑', '乌黑',
                 '乌黑', '青绿', '浅白', '浅白', '青绿', '浅白', '乌黑', '浅白', '青绿'],
        '根蒂': ['蜷缩', '蜷缩', '蜷缩', '蜷缩', '蜷缩', '稍蜷', '稍蜷', '稍蜷',
                 '稍蜷', '硬挺', '硬挺', '蜷缩', '稍蜷', '稍蜷', '稍蜷', '蜷缩', '蜷缩'],
        '敲声': ['浊响', '沉闷', '浊响', '沉闷', '浊响', '浊响', '浊响', '浊响',
                 '沉闷', '清脆', '清脆', '浊响', '浊响', '沉闷', '浊响', '浊响', '沉闷'],
        '纹理': ['清晰', '清晰', '清晰', '清晰', '清晰', '清晰', '稍糊', '清晰',
                 '稍糊', '清晰', '模糊', '模糊', '稍糊', '稍糊', '清晰', '模糊', '稍糊'],
        '脐部': ['凹陷', '凹陷', '凹陷', '凹陷', '凹陷', '稍凹', '稍凹', '稍凹',
                 '稍凹', '平坦', '平坦', '平坦', '凹陷', '凹陷', '稍凹', '平坦', '稍凹'],
        '触感': ['硬滑', '硬滑', '硬滑', '硬滑', '硬滑', '软粘', '软粘', '硬滑',
                 '硬滑', '软粘', '硬滑', '软粘', '硬滑', '硬滑', '软粘', '硬滑', '硬滑'],
        '好瓜': ['是', '是', '是', '是', '是', '是', '是', '是',
                 '否', '否', '否', '否', '否', '否', '否', '否', '否']
    }
    df = pd.DataFrame(data)
    feature_names = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    x = df[feature_names].values
    y = df['好瓜'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model=ID3(feature_names)
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    print(y_test)
    print(y_pre)
from id3 import ID3
import numpy as np
from collections import Counter
class C4_5(ID3):
    def information_gain_rate(self, x_col,y):
        return 0 if self.entropy(x_col)==0 else self.information_gain(x_col,y)/self.entropy(x_col)
    def c4_5Tree(self, x, y,feature_indices):
        """
                生成id3
                :param x: 当前剩余的特征矩阵
                :param y: 与x对应的y
                :param feature_indices: 与x对应的特征序列
                :return: tree
                """
        x = np.array(x)
        y = np.array(y)
        # 计算当前节点的多数类
        current_majority = Counter(y).most_common(1)[0][0]
        # 如果所有样本属于同一类别，返回叶子节点
        if len(np.unique(y)) == 1:
            return y[0]
        # 如果没有剩余特征，或者所有特征取值相同返回多数类别
        if len(feature_indices) == 0 or self.is_same_on_features(x, feature_indices):
            return Counter(y).most_common(1)[0][0]

        best_gain = -1  # 最大信息增益
        best_feature_idx = None  # 最大信息增益索引
        for idx in feature_indices:
            x_col = x[:, idx]
            gain = self.information_gain_rate(x_col, y)
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = idx
        # 生成树
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
            # 递归调用
            tree['children'][value] = self.id3Tree(x_sub, y_sub, new_feature_indices)
        return tree

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

    model =C4_5(feature_names)
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    print(y_test)
    print(y_pre)
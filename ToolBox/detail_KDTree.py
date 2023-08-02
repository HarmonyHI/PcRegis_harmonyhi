import numpy as np
import pandas as pd
class KdNode(object):
    def __init__(self,pInfo=None,pSplit=None,pLeft=None,pRight=None):

        self.leftChild = pLeft
        self.rightChild = pRight
        self.info = pInfo
        self.splitFeature = pSplit

def createKdTree(pDataset):
    '''
    :param pDataset:训练样本数据
    :return: KD-Tree
    :function 创建KNN二叉树，理论上应该将样本数据存放在叶子结点上
    这里将数据存储在所有节点上（包括叶子节点和非叶子节点）
    '''
    if len(pDataset) <= 0:
        return

    if len(pDataset) == 1:
        node = KdNode()
        node.info = pDataset.iloc[0]
        node.splitFeature = -1
        return node

    dataset = pd.DataFrame(pDataset)

    max_var = -9999999
    split_feature = -1
    for ix, col in dataset.iteritems():
        std = col.std()
        if(std > max_var):
            max_var = std
            split_feature = ix

    if split_feature >= 0:
        dataset.sort_values(by=split_feature,axis=0,inplace=True)
    else:
        return

    split_point = dataset.iloc[int(dataset.shape[0]/2)]
    node = KdNode(split_point,split_feature)
    node.leftChild = createKdTree(dataset.iloc[0:int(dataset.shape[0]/2)])
    node.rightChild = createKdTree(dataset.iloc[int(dataset.shape[0]/2+1):])

    return node

def printTree(pRoot):
    '''
    :param pRoot: KD-Tree根节点
    :function 打印KD-Tree
    '''
    if(pRoot):
        print(pRoot.info.values)
    else:
        return

    printTree(pRoot.leftChild)
    printTree(pRoot.rightChild)

def findSmlstDis(pRoot,pQuery):
    '''
    :param pRoot:KD-Tree根节点
    :param pQuery: 查找的节点
    :return: 和pQuery距离最近的节点和最近距离（欧氏距离）
    '''
    pQuery = pd.Series(pQuery)
    cur_node = pRoot
    nearest_node = cur_node
    min_distance = 999999999
    node_stack = []

    while cur_node:
        node_stack.append(cur_node)
        cur_distance = np.sum((pQuery - cur_node.info)**2)
        if cur_distance < min_distance:
            min_distance = cur_distance
            nearest_node = cur_node

        split_feature = cur_node.splitFeature
        if split_feature >= 0:
            if(pQuery[split_feature] <= cur_node.info[split_feature]):
                cur_node = cur_node.leftChild
            else:
                cur_node = cur_node.rightChild
        else:
            break

    while node_stack:
        back_point = node_stack.pop()
        split_feature = back_point.splitFeature

        cur_distance = np.sqrt(np.sum((pQuery - back_point.info) ** 2))
        if min_distance > cur_distance:
            min_distance = cur_distance
            nearest_node = back_point

        if(split_feature >= 0):
            temp_node = None
            if( np.abs(pQuery[split_feature] - back_point.info[split_feature]) < min_distance ):
                if pQuery[split_feature] <= back_point.info[split_feature]:
                    temp_node = back_point.rightChild
                else:
                    temp_node = back_point.leftChild
            if temp_node:
                node_stack.append(temp_node)
                cur_distance = np.sqrt(np.sum((pQuery-temp_node.info)**2))
                if min_distance > cur_distance:
                    min_distance = cur_distance
                    nearest_node = temp_node

    return nearest_node, min_distance

def DistanceChecker(px,py):
    x = np.sqrt(np.sum((px - py)**2,axis=1))
    x.sort(axis=0)
    return x

if __name__ == "__main__":
    y = [1.0,2.0,3.0]
    x = np.array(
        [
            [2.0, 2.1, 2.2],
            [2.2, 2.4, 2.1],
            [1.9, 2.0, 3.0],
            [1.0, 1.1, 1.5],
            [1.0, 2.0, 3.0],
            [1.0, 1.0, 1.8],
            [0.0, 0.0, 0.5],
            [0.1, 0.0, 0.3],
            [0.0, 0.1, 0.5],
            [0.4, 0.5, 0.6],
            [0.5, 0.4, 0.4],
            [1.0, 2.1, 2.2],
            [2.2, 2.4, 2.1],
            [3.9, 3.0, 3.0],
            [4.0, 3.1, 1.5],
            [5.0, 4.0, 1.8],
            [6.0, 5.0, 0.5],
            [7.1, 0.0, 0.3],
            [8.0, 0.1, 1.5],
            [9.4, 1.5, 0.6],
            [10.5, 1.4, 0.4],
        ]
    )
    root = createKdTree(x)
    d = DistanceChecker(x,y)
    print(d)
    a,b = findSmlstDis(root,y)
    print(a.info,b)
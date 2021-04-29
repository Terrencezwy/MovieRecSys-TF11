import numpy as np
from numpy import linalg as la

USER_NUM=943
ITEM_NUM=1683
workdir = 'ml-100k/'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'

user_pos_train=[None]*USER_NUM
for i in range(USER_NUM):
    user_pos_train[i]=[0]*(ITEM_NUM)
with open(workdir + 'movielens-100k-train.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        user_pos_train[uid][iid]=r


user_pos_test=[None]*USER_NUM
cnt = 0
for i in range(USER_NUM):
    user_pos_test[i]=[0]*(ITEM_NUM)

with open(workdir + 'movielens-100k-test.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        user_pos_test[uid][iid] = r
        cnt = cnt+1

def Rec_list_precision(u,all_score):

    print(all_scores)
    r=[]
    for i in range(10):
        if all_score[u][i][0] in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    print(r)
    p_3 = mean(r[:3])
    p_5 = mean(r[:5])
    p_10 = mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)
    return array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])

def compute_precision(dataMat):
    all_score=recommend(dataMat)
    accuracy = 0
    precision = 0
    TP = 0
    PY = 0
    AY = 0
    for i in range(USER_NUM):
      for j in range(ITEM_NUM):
        if(all_score[i][j] == user_pos_test[i][j]):
          accuracy = accuracy + 1
        if(all_score[i][j] > 3 and user_pos_test[i][j] > 3):
          TP = TP + 1
        if(all_score[i][j] > 3 && user_pos_test[i][j] != 0):
          PY = PY + 1
        if(user_pos_test[i][j] > 3):
          AY = AY + 1
    accuracy = accuracy / cnt
    precision = TP / PY
    recall = TP / AY


    print(accuracy)
    print(precision)
    print(recall)




def dcg_at_k(r, k):
    r = asfarray(r)[:k]
    return sum(r / log2(arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def loadExData():
    return np.mat((user_pos_train))

def ecludSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))

def pearsSim(inA,inB):
    if len(inA)<3: return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    num=float(inA.T*inB)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


def sigmaPct(sigma,percentage):
    sigma2=sigma**2
    sumsgm2=sum(sigma2)
    sumsgm3=0
    k=0
    for i in sigma:
        sumsgm3+=i**2
        k+=1
        if sumsgm3>=sumsgm2*percentage:
            return k


def svdEst(dataMat,percentage):
    # n=shape(dataMat)[1]
    # simTotal=0.0;ratSimTotal=0.0
    u,sigma,vt=la.svd(dataMat)
    k=sigmaPct(sigma,percentage)
    sigmaK=np.mat(np.eye(k)*sigma[:k])
    xformedItems=dataMat.T*u[:,:k]*sigmaK.I
    return xformedItems



def recommend(dataMat,N=10,simMeas=ecludSim,percentage=0.9):
    xformedItems = svdEst(dataMat, percentage)
    All_scores=[]
    All_scores_2 = np.zeros((USER_NUM, ITEM_NUM))
    n = np.shape(dataMat)[1]
    for user in range (np.shape(dataMat)[0]):
        unratedItems=np.nonzero(dataMat[user,:].A==0)[1]

        if len(unratedItems)==0:return 'you rated everything'
        itemScores=[]
        for item in range(ITEM_NUM):
            simTotal=0.0
            ratSimTotal=0.0
            for j in range(n):
                userRating = dataMat[user, j]
                if userRating == 0 or j == item: continue
                similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
                simTotal += similarity
                ratSimTotal += similarity * userRating
            if simTotal == 0: estimatescore=0
            else: estimatescore=ratSimTotal/simTotal
            All_scores_2[user][item]=round(estimatescore)
            # itemScores.append((item,estimatescore))
        # itemScores=sorted(itemScores,key=lambda x:x[1],reverse=True)
        # itemScores=itemScores[:N]
        print(user)
        # print(All_scores_2[user][-1])
        # All_scores.append(itemScores)
        #All_scores like [[(1, 3.0), (3, 2.5), (4, 2.3)],[(11,3.2),(23,2.8),(88,2.7)]]
    return All_scores_2


if __name__=='__main__':
    testdata=loadExData()

    compute_precision(testdata)






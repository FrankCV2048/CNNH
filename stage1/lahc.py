import numpy as np
import itertools
def learning_hash(s,q,T,error):
    """
    :param s: a pairwise similarity matrix 
    :param q: the number of bits 
    :param T: maximum iterations T
    :param error: the tolerance error 
    :return: coordinate descent algorithm for hash bit learning
    """
    n,m=np.shape(s)
    H=np.random.randint(-1,1,[n,q])
    a=q*s
    L=np.matmul(H,H.T)-q*s
    indexs_i=np.array(range(n))
    indexs_j=np.array(range(q))
    np.random.shuffle(indexs_i)
    np.random.shuffle(indexs_j)
    F=np.linalg.norm(L)
    for t in range(T):
        L_temp=L
        H_temp=H
        for item in itertools.product(indexs_i,indexs_j):
            i=item[0]
            j=item[1]
            H_1=4*np.dot(L_temp[i,:],H_temp[:,j].T)
            H_2=4*(np.dot(H_temp[:,j].T,H_temp[:,j])+np.power(H_temp[i][j],2))+L_temp[i][i]
            d=np.maximum(-1-H_temp[i][j],np.minimum((-1)*np.divide(H_1,H_2),1-H_temp[i][j]))
            H_temp[i][j]=H_temp[i][j]+d
            L_temp[i,:]=L_temp[i,:]+d*H_temp[:,j].T
            L_temp[:,i]=L_temp[:,i]+d*H_temp[:,j]
            L_temp[i][i]=L_temp[i][i]+np.power(d,2)
        F_temp=np.linalg.norm(L_temp)
        if F_temp<=F:
            H=H_temp
            L=L_temp
        else:
            continue
        if np.divide((F-F_temp),F)<error:
            break
    return H
    pass

# a=np.random.randint(-1,1,[5000,5000])
# h=learning_hash(a,20,10,0.3)
# print(h)

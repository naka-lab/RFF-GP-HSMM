# encoding: utf8
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import copy


class GP:
    def __init__(self, in_dim=1, M=20, sigma=0.1, alpha=1.0, beta=10.0 ):
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.D = in_dim

        if M>0:
            self.M = M
            self.K = np.random.randn(self.D,self.M)*self.sigma
            self.B = np.random.random(self.M)*np.pi
        else:
            self.M = 20
            self.K = np.load("K_20.npy")
            self.B = np.load("B_20.npy")

    def phi( self, xt ):
        return np.cos(np.dot(xt,self.K)+self.B) * math.sqrt(2/self.M)

    def learn(self, xt, yt, feat=None, S=None ):
        xt = np.array(xt)
        yt = np.array(yt)
        N = len(xt)

        # 基底関数で写像
        if feat is None:
            feat = self.phi( xt )

        # 線形回帰の重みを計算
        #self.W = np.linalg.solve( np.dot(feat.T, feat), np.dot(feat.T, self.yt) )

        # ベイズ線形回帰
        if S is None:
            self.S = np.linalg.inv(self.alpha * np.eye(self.M) + self.beta * np.dot(feat.T, feat))
        else:
            self.S = S

        self.mu = self.beta * np.dot(self.S, np.dot(feat.T, yt))

        return feat, self.S


    def predict( self, x ):
        feat = self.phi(x)
        #return np.dot(feat, self.W)
        return np.dot( feat, self.mu ), 1/self.beta + np.diag(np.dot(np.dot(feat,self.S), feat.T))

    def calc_lik( self, x, y ):
        mus, sigmas = self.predict(x)
        return -0.5*np.sum( np.log(sigmas) + (y-mus)**2/sigmas )

    def save_model(self, dir ):
        if not os.path.exists(dir):
            os.makedirs(dir)

        np.save( os.path.join(dir, "K.npy"), self.K )
        np.save( os.path.join(dir, "B.npy"), self.B )
        np.save( os.path.join(dir, "S.npy"), self.S )
        np.save( os.path.join(dir, "mu.npy"), self.mu )
        np.save( os.path.join(dir, "alpha_beta_sigma.npy"), [self.alpha, self.beta, self.sigma] )

    def load_model(self, dir ):
        self.K = np.load( os.path.join(dir, "K.npy") )
        self.B = np.load( os.path.join(dir, "B.npy") )
        self.S = np.load( os.path.join(dir, "S.npy") )
        self.mu = np.load( os.path.join(dir, "mu.npy") )
        self.alpha, self.beta, self.sigma = np.load( os.path.join(dir, "alpha_beta_sigma.npy") )

class GPMD:
    def __init__(self, dim, M, sigma, alpha, beta ):
        self.__dim = dim
        self.__M = M

        gp = GP(1, M, sigma, alpha, beta)
        self.__gp = [gp] + [ copy.deepcopy(gp) for i in range(dim-1)] # 同じパラメータのGPを複数作成

    def learn(self,x, y, same_cov=True ):
        y = np.array(y, dtype=float).reshape( (-1,self.__dim, 1) )
        x = np.array(x, dtype=float).reshape( (-1,1) )

        # featとSは各次元で共通なので，1回計算したら使い回す
        feat = None
        S = None

        for d in range(self.__dim):
            if len(y)!=0:
                feat, S = self.__gp[d].learn( x, y[:,d], feat, S )
            else:
                feat, S = self.__gp[d].learn( x, np.array([]), feat, S )

    def calc_lik(self, x, y ):
        lik = 0.0

        if self.__dim==1:
            y = np.asarray(y, dtype=float).reshape( (-1,self.__dim) )
        #x = np.asarray(x,dtype=np.float)
        for d in range(self.__dim):
            lik += self.__gp[d].calc_lik( x.reshape(-1,1) , y[:,d].reshape(-1,1) )

        return lik

    def plot(self, x ):
        for d in range(self.__dim):
            plt.subplot( self.__dim, 1, d+1 )

            mus, sigmas = self.__gp[d].predict(x.reshape(-1,1))
            y_min = mus.flatten() - sigmas
            y_max = mus.flatten() + sigmas

            plt.fill_between( x, y_min, y_max, facecolor="lavender" , alpha=0.9 , edgecolor="lavender"  )
            plt.plot(x, y_min, 'b--')
            plt.plot(x, mus, 'b-')
            plt.plot(x, y_max, 'b--')

    def predict(self, x ):
        params = []
        for d in range(self.__dim):
            mus, sigmas = self.__gp[d].predict(np.array(x, dtype=float).reshape(-1,1))
            params.append( (mus, sigmas) )
        return params

    def save_model(self, dir ):
        if not os.path.exists(dir):
            os.makedirs(dir)

        for d in range(self.__dim):
            self.__gp[d].save_model( os.path.join(dir, f"gp{d:03}") )
        
    def load_model(self, dir ):
        for d in range(self.__dim):
            self.__gp[d].load_model( os.path.join(dir, f"gp{d:03}") )


def main():
    g = GPMD(1, 5)
    g.learn([],[])
    g.plot( np.linspace(0,10,100) )
    plt.show()

if __name__ == '__main__':
    main()
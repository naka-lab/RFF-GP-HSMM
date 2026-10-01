import numpy as np
from GPSegmentation import GPSegmentation
import time
from sklearn.metrics.cluster import adjusted_rand_score


def test( modeldir ):
    gpsegm = GPSegmentation(2, 3, min_max_ave_len=(15,20,30))
    files = [ f"dataset/test/data{i:03}.txt" for i in range(3) ]
    gpsegm.load_data( files )
    gpsegm.load_model( modeldir )

    start = time.time()
    gpsegm.recog()
    gpsegm.save_model( "temp/" )

    labels = []
    pred_labels = []
    for i in range(3):
        labels.append( np.loadtxt( f"dataset/test/labels{i:03}.txt" )[:,0] )
        pred_labels.append( np.loadtxt( f"temp/segm{i:03}.txt" )[:,0] )

    labels = np.concatenate( labels )
    pred_labels = np.concatenate( pred_labels )
    print( "Adjusted Rand Index:", adjusted_rand_score( labels, pred_labels ) )

def learn_and_evaluate(ITR=2, incremental = True):
    for i in range(0,4):
        gpsegm = GPSegmentation(2, 3, min_max_ave_len=(15,20,30))

        if i==0:
            data_files = [ f"dataset/initial/data{i:03}.txt" for i in range(2) ]
        else:
            data_files = [ f"dataset/additional/data{i:03}.txt" for i in range((i-1)*2, i*2) ]
            if incremental:
                gpsegm.load_model( f"learn{i-1:03}/" )

        gpsegm.load_data( data_files )

        start = time.time()
        for it in range(ITR):
            gpsegm.learn()
        gpsegm.save_model( f"learn{i:03}/" )

        test( f"learn{i:03}/" )

    return gpsegm.calc_lik()


def main():
    print("---- non incremanetal learning  ----")
    learn_and_evaluate(1, incremental=False)

    print("---- incremental learning  ----")
    learn_and_evaluate(1, incremental=True)
    return

if __name__=="__main__":
    main()

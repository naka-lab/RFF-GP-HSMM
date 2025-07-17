# -*- coding: utf-8 -*-
from GPSegmentation import GPSegmentation
import time


def learn( savedir ):
    gpsegm = GPSegmentation(8, 11, min_max_ave_len=(15,20,30))
    files = [ f"data{i:03}.txt" for i in range(3) ]*5
    gpsegm.load_data( files )

    print("---- training ----")

    start = time.time()
    ITR = 10
    for it in range(ITR):
        gpsegm.learn()
        print(f"{it+1}/{ITR}\r", end="")

    lik = gpsegm.calc_lik()
    print()
    print( "lik =", lik )
    print( time.time()-start, "sec" )
    gpsegm.save_model( savedir )

    return gpsegm.calc_lik()


def recog( modeldir, savedir ):
    gpsegm = GPSegmentation(8, 11, min_max_ave_len=(15,20,30))
    files = [ f"data{i:03}.txt" for i in range(3) ]
    gpsegm.load_data( files )
    gpsegm.load_model( modeldir )

    print("---- recognition ----")
    start = time.time()
    gpsegm.recog()
    print( "lik =", gpsegm.calc_lik() )
    print( time.time()-start, "sec" )
    gpsegm.save_model( savedir )

def main():
    learn( "learn/" )
    recog( "learn/" , "recog/" )
    return

if __name__=="__main__":
    main()

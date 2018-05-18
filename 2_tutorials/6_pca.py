import scipy as sp
from scipy import linalg
from scipy import nan as NA
import pandas as pd
from scipy.stats import rankdata

def pca(dat, npca=None, verbose = False):
    if isinstance(dat, sp.ndarray):
        dat = pd.DataFrame(dat)
        names = []
        for i in range(dat.shape[1]):
            names.append("x"+str(i+1))
        dat.columns = names
    names = list(dat.columns)
    nr = dat.shape[0]
    nc = dat.shape[1]
    r = sp.corrcoef(dat, rowvar=False)
    heikin = dat.mean(axis=0)
    bunsan = dat.var(axis=0, ddof=1)
    sd = sp.sqrt(bunsan)
    eval, evec = linalg.eig(r)
    eval = sp.real(eval)
    rank = rankdata(eval, method="ordinal")
    rank = nc+1-rank
    eval2 = eval.copy()
    evec2 = evec.copy()
    for i in range(nc):
        j = sp.where(rank == i+1)[0][0]
        eval[i] = eval2[j]
        evec[:, i] = evec2[:, j]
    contr = eval/nc*100
    cum_contr = sp.cumsum(contr)
    fl = (sp.sqrt(eval)*evec)
    for i in range(nc):
        dat.ix[:, i] = (dat.ix[:, i]-heikin[i]) / sd[i]
    fs = sp.dot(dat, evec*sp.sqrt(nr/(nr-1)))
    if npca is None:
        npca = sp.sum(eval >= 1)
    eval = eval[0:npca]
    cont = eval/nc
    cumc = sp.cumsum(cont)
    fl = fl[:, 0:npca]
    rcum = sp.sum((fl ** 2), axis=1)
    if verbose:
        print("            ", end="")
        for j in range(npca):
            print("{0:>8s}".format("PC"+str(j+1)), end="")
        print("  Contribution")
        for i in range(nc):
            print("{0:>12s}".format(names[i]), end="")
            for j in range(npca):
                print(" {0:7.3f}".format(fl[i, j]), end="")
            print(" {0:7.3f}".format(rcum[i]))
        print("  Eigenvalue", end="")
        for j in range(npca):
            print(" {0:7.3f}".format(eval[j]), end="")
        print("\nContribution", end="")
        for j in range(npca):
            print(" {0:7.3f}".format(cont[j]), end="")
        print("\nCum.contrib.", end="")
        for j in range(npca):
            print(" {0:7.3f}".format(cumc[j]), end="")
        print()
    return {"r":r, "fl":fl, "eval":eval, "fs":fs[:, 0:npca]}

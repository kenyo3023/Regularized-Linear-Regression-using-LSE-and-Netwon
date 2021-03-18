from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from optimizer import LSE, Newton

def X_polybases(data, N, n=2):
    X = [np.ones(N)]
    for j in range(n-1):
        X.append([data[i*2]**(j+1) for i in range(N)])
    return np.vstack(X[::-1]).T

def plot(X, y, theta):
    x1 = X[:, -2]
    plt.figure(figsize=(10, 4))
    plt.scatter(x1, y)
    plt.plot(x1, np.dot(X, theta))
    #plt.show()

def main(args):
    n = args.n
    lambd = args.lambd
    optimizer = args.optimizer
    isplot = args.isplot

    data = [-5.0, 51.76405234596766,
    -4.795918367346939, 45.42306433039972,
    -4.591836734693878, 41.274448104888755,
    -3.979591836734694, 26.636216497466364,
    -3.571428571428571, 20.256806057008426,
    -2.9591836734693877, 11.618429243797276,
    -2.7551020408163263, 10.450525068812203,
    -1.7346938775510203, 1.8480982318414874,
    -1.3265306122448979, -1.0405349639051173,
    -0.9183673469387754, -4.614630798757861,
    -0.7142857142857144, -1.3871977310902517,
    -0.3061224489795915, -1.9916444039966117,
    0.1020408163265305, -0.912924608376358,
    0.7142857142857144, 6.63482003068499,
    1.1224489795918373, 9.546867459016372,
    1.7346938775510203, 15.72016146597016,
    1.9387755102040813, 20.62251683859554,
    2.5510204081632653, 33.48059725819715,
    2.959183673469388, 40.76391965675495,
    3.979591836734695, 66.8997605629381,
    4.387755102040817, 78.44316465660981,
    4.591836734693878, 86.99156782355371,
    5.0, 99.78725971978604]

    N = int(len(data)/2)
    X = X_polybases(data, N, n=n)
    y = [data[i*2+1] for i in range(N)]
    theta_box = {}

    print('\nCase: n = %d, lambda = %d'%(n, lambd))

    if (optimizer == 'lse') or (optimizer == 'both'):
        lse = LSE()
        lse.fit(X, y, lambd=lambd)
        lse.show_report(X, y)
        theta_box['lse'] = lse.theta

    if (optimizer == 'newton') or (optimizer == 'both'):
        newton = Newton()
        newton.fit(X, y)
        newton.show_report()
        theta_box['newton'] = newton.theta
    
    if isplot:
        x1 = X[:, -2]
        num = 2 if optimizer == 'both' else 1
        colsize = 4 * (num)
        f = plt.figure(figsize=(10, colsize))
        axs = [f.add_subplot(str(num)+'1'+str(i)) for i in range(1,num+1)]
        for i, (key, value) in enumerate(theta_box.items()):
            axs[i].scatter(x1, y)
            axs[i].plot(x1, np.dot(X, value))
            axs[i].set_xlabel(key.upper())
        plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--n", help="number of polynomial bases n.", default=2, type=int)
    parser.add_argument("--lambd", help="regularized weight.", default=0, type=int)
    parser.add_argument("--optimizer", help="fit the model using LSE, Newton or both", default='both')
    parser.add_argument("--isplot", help="whether to draw & save the result plots", action="store_true")
    args = parser.parse_args()
    main(args)

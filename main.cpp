#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <vector>
#include <queue>
#include <deque>
#include <cstring>
#include <string>
#include <cstdlib>
#include <time.h>
#include <fstream>
using namespace std;
#define fi "input.txt"
#define fo "output.txt"
#define fileopen freopen(fi,"r",stdin);freopen(fo,"w",stdout);
#define FOR(i,l,r) for(int i=(int)l;i<=(int)r;i++)
#define FORD(i,l,r) for(int i=(int)l;i>=(int)r;i--)
#define xy pair<int,int>
#define pb push_back
#define int64 long long
#define X first
#define Y second
#define init(a,v) memset(a,v,sizeof(a))
#define Sz(s) ((int)s.size())
#define forever while (true)
#define ran(l,r) (rand()%((int)(r)-(int)(l)+1)+(int)(l))
#define EL printf("\n")

const int OO = 2e9;
const int MOD = 1e9+7;
const double Pi = 3.141592653589793;
const int debug = 0;
const int extra = 0;

const double eps = 1;
const double EPS = 1e-5;
const double alpha = 1;
const double lambda = 1e-4;
const int times = 0;

typedef vector<double> vd;
typedef vector<vd> vdd;
typedef vector<vdd> vddd;

typedef vector<int> vi;
typedef vector<vi> vii;


double ranw() {
    return eps*2*ran(1,100)/100.0-eps;
}

/**** DEBUG ******/

void print(vd &a) {
    FOR(i,0,Sz(a)-1) printf("%.5lf ",a[i]);EL;
}

void print(vdd &a) {
    FOR(i,0,Sz(a)-1) print(a[i]);
}

void print(vddd &a) {
    FOR(i,0,Sz(a)-1) {
        print(a[i]);
        EL;
    }
}

/**MATRIX OPERATORS*/

void resize(vdd &a,int n,int m) {
    a.resize(n);
    FOR(i,0,n-1) a[i].resize(m);
}

void reset(vdd &a) {
    FOR(i,0,Sz(a)-1) FOR(j,0,Sz(a[i])-1)
        a[i][j]=0;
}

vdd T(vdd &a) {
    vdd res;
    resize(res,Sz(a),Sz(a[0]));
    FOR(i,0,Sz(a)-1) FOR(j,0,Sz(a[i])-1)
        res[j][i]=a[i][j];
    return res;
}

vdd operator *(vdd &a,vdd &b) {
    vdd res;int n=Sz(a),m=Sz(b[0]);
    resize(res,n,m);reset(res);
    FOR(i,0,n-1) FOR(k,0,Sz(a[0])-1) FOR(j,0,m-1)
        res[i][j]+=a[i][k]*b[k][j];
    return res;
}

vd operator *(vdd &a,vd &b) {
    vd res; res.resize(Sz(a));
    FOR(i,0,Sz(res)-1) {
        res[i]=0;
        FOR(j,0,Sz(a[i])-1)
            res[i]+=a[i][j]*b[j];
    }
    return res;
}

vd mul(vd &a,vd &b) {
    vd res;res.resize(Sz(a));
    FOR(i,0,Sz(a)-1) res[i]=a[i]*b[i];
    return res;
}

/********************************************************************************************/

int last=0;

struct neural_network{
    /**=================================Architect===============================================*/
    vdd z,a,e,b,db;
    int layer,ntrain;
    vi nlayer;
    vddd w,wT,dw;
    double lr,lmd,J;

    void build(vi &x) {
        J=0;
        lr=alpha;
        lmd=lambda;
        layer=Sz(x);
        nlayer.resize(layer);
        z.resize(layer);
        a.resize(layer);
        e.resize(layer);
        b.resize(layer-1);
        db.resize(layer-1);
        w.resize(layer-1);
        wT.resize(layer-1);
        dw.resize(layer-1);
        FOR(l,0,layer-1) {
            z[l].resize(x[l]);
            a[l].resize(x[l]);
            e[l].resize(x[l]);
            nlayer[l]=x[l];
            if (l<layer-1) {
                b[l].resize(x[l+1]);
                db[l].resize(x[l+1]);
                resize(w[l],x[l+1],x[l]);
                resize(wT[l],x[l],x[l+1]);
                resize(dw[l],x[l+1],x[l]);
            }
        }
        J=1e9;
    }

    void random() {
        FOR(l,0,layer-2) {
            FOR(i,0,Sz(w[l])-1)
                FOR(j,0,Sz(w[l][i])-1)
                    w[l][i][j]=wT[l][j][i]=ranw();
            FOR(i,0,Sz(b[l])-1) b[l][i]=ranw();
        }
    }

    /**=================================FUNCTIONS===============================================*/
    /**Activate function and its derivative*/
    double f(double x) {
        //return tanh(x);
        return 1/(1+exp(-x));
    }
    double df(double x) {
        //double fx=f(x);
        //return 1-fx*fx;
        double ex=exp(x);
        return ex/(ex+1)/(ex+1);
    }

    void f(vd &a) {
        FOR(i,0,Sz(a)-1)
            a[i]=f(a[i]);
    }
    void df(vd &a) {
        FOR(i,0,Sz(a)-1)
            a[i]=df(a[i]);
    }
    /**Cost function and its derivative*/
    double L(double x,double y) {
        return (-y*log(x)-(1-y)*log(1-x))/ntrain;
    }
    double dL(double x,double y) {
        return (x-y)/(x-x*x)/ntrain;
    }
    void dL(vd &a,vd &y) {
        FOR(i,0,Sz(a)-1)
            a[i]=dL(a[i],y[i]);
    }
    double calJ(vd &y) {
        double res=0;
        FOR(i,0,nlayer[layer-1]-1)
            res+=L(a[layer-1][i],y[i]);
        return res;
    }
    /**==========================FEED FORWARD & BACK PROPAGATION==========================*/
    void feedforw(vd &x) {
        a[0]=x;
        FOR(l,1,layer-1) {
            z[l]=w[l-1]*a[l-1];
            FOR(i,0,Sz(z[l])-1) z[l][i]+=b[l-1][i];
            a[l]=z[l];f(a[l]);
        }
    }
    void backprop(vd &y) {
        e[layer-1]=a[layer-1];
        dL(e[layer-1],y);
        df(z[layer-1]);
        e[layer-1]=mul(e[layer-1],z[layer-1]);
        FORD(l,layer-2,1) {
            e[l]=wT[l]*e[l+1];
            df(z[l]);
            e[l]=mul(e[l],z[l]);
        }
        FOR(l,0,layer-2) {
            FOR(i,0,Sz(w[l])-1) FOR(j,0,Sz(w[l][i])-1)
                dw[l][i][j]+=e[l+1][i]*a[l][j];
            FOR(i,0,Sz(b[l])-1) db[l][i]+=e[l+1][i];
        }
    }
    void update() {
        FOR(l,0,layer-2) {
            FOR(i,0,Sz(w[l])-1) FOR(j,0,Sz(w[l][i])-1) {
                w[l][i][j]-=lr*(dw[l][i][j]+lmd*w[l][i][j]);
                wT[l][j][i]=w[l][i][j];
            }
            FOR(i,0,Sz(b[l])-1) b[l][i]-=lr*(db[l][i]+lmd*b[l][i]);
        }
    }
    void train(vddd &train) {
        ntrain=Sz(train);
        double preJ=J;J=0;
        FOR(l,0,layer-2) reset(dw[l]);
        reset(db);
        FOR(i,0,ntrain-1) {
            feedforw(train[i][0]);
            J+=calJ(train[i][1]);
            backprop(train[i][1]);
        }
        if (debug) printf("====================J = %.15lf=====================\n",J);
        if (preJ+EPS<J)
            cout<<"LAERNING RATE IS TOO LARGE !"<<endl;
        update();
    }
    /**===================================OUTPUT & PRINTING==========================================*/
    vd output(vd &x) {
        feedforw(x);
        return a[layer-1];
    }
    void p() {
        FOR(i,0,layer-1) {
            cout<<"Layer "<<i<<endl;
            if (i<layer-1) {
                cout<<"Weight:"<<endl;
                print(w[i]);
            }
            if (i>0) {
                cout<<"Bias:"<<endl;
                print(b[i-1]);
            }
            EL;
        }
    }
    /**=========================================DEBUG=================================================*/
    double check(int l,int i,int j,vddd &train) {
        double J1=0,J2=0,tmp=w[l][i][j];
        w[l][i][j]=tmp-EPS;
        FOR(t,0,Sz(train)-1) {
            feedforw(train[t][0]);
            J1+=calJ(train[t][1]);
        }
        w[l][i][j]=tmp+EPS;
        FOR(t,0,Sz(train)-1) {
            feedforw(train[t][0]);
            J2+=calJ(train[t][1]);
        }
        w[l][i][j]=tmp;
        return (J2-J1)/2.0/EPS;
    }
    /**========================================END===================================================*/
} nn;

int F[2],n,ntest,layer,cnst,error;
vddd trainset,testset;
vi sz;

/**========================== INPUT PROCEDURES ==========================*/
void readST() {
    cin>>layer;
    sz.resize(layer);
    FOR(i,0,layer-1) cin>>sz[i];
    F[0]=sz[0];
    F[1]=sz[layer-1];
    nn.build(sz);
}

void readTS() {
    cin>>n;
    trainset.resize(n);
    FOR(i,0,n-1) {
        trainset[i].resize(2);
        FOR(t,0,1) {
            trainset[i][t].resize(F[t]);
            FOR(j,0,F[t]-1) scanf("%lf",&trainset[i][t][j]);
        }
    }
}

void readW() {
    cin>>cnst;
    if (cnst==0) {nn.random();return;}
    FOR(l,0,layer-2) {
        FOR(i,0,Sz(nn.w[l])-1) FOR(j,0,Sz(nn.w[l][i])-1) {
            scanf("%lf",&nn.w[l][i][j]);
            nn.wT[l][j][i]=nn.w[l][i][j];
        }
        FOR(i,0,Sz(nn.b[l])-1) scanf("%lf",&nn.b[l][i]);
    }
}

void readTEST() {
    ifstream f1("D:/CodeBlocks/share/CodeBlocks/My Codes/test/output.txt");
    cin>>ntest;
    testset.resize(ntest);
    FOR(i,0,ntest-1) {
        testset[i].resize(2);
        FOR(t,0,1) {
            testset[i][t].resize(F[t]);
            FOR(j,0,F[t]-1) f1>>testset[i][t][j];
        }
    }
    f1.close();
}

void READ() {
    fileopen;
    string header;
    cin>>header;
        readST();
    //cin>>header;
    //    readTS();
    cin>>header;
        readW();
    cin>>header;
        readTEST();
}
/**===========================================TEST=================================================*/
/**int TEST(neural_network &nn,vddd &test) {
    vd outp;vi oi;oi.resize(Sz(test[0][1]));
    bool ok; int res=0;
    FOR(i,0,Sz(test)-1) {
        outp=nn.output(test[i][0]);
        ok=true;
        FOR(j,0,Sz(outp)-1) {
            oi[j]=round(outp[j]);
            ok&=(oi[j]==test[i][1][j]);
        }
        if (!ok) res++;
        if (1) {
            cout<<"TEST CASE "<<i<<endl;
            cout<<"INPUT :"<<endl;
            FOR(j,0,Sz(test[i][0])-1) cout<<test[i][0][j]<<" ";EL;
            cout<<"OUTPUT :"<<endl;
            FOR(j,0,Sz(oi)-1) printf("%i ",oi[j]);EL;
            FOR(j,0,Sz(outp)-1) printf("%.5lf ",outp[j]);EL;
            cout<<"ANSWER :"<<endl;
            FOR(j,0,Sz(test[i][1])-1) printf("%.5lf ",test[i][1][j]);EL;
            cout<<"VERDICT : ";
            if (ok) cout<<"OK"; else cout<<"WRONG";
            EL;EL;
        }
    }
    return res;
}*/
int TEST(neural_network &nn,vddd &test) {
    if (Sz(test)==0) return 0;
    vd outp;
    int res=0,p1,p2;
    FOR(i,0,Sz(test)-1) {
        outp=nn.output(test[i][0]);
        p1=p2=0;
        FOR(j,1,Sz(outp)-1) {
            if (outp[j]>outp[p1]) p1=j;
            if (test[i][1][j]>test[i][1][p2]) p2=j;
        }
        if (p1!=p2) res++;
        //cout<<"ANSWER : "<<p2<<endl;
        cout<<"OUTPUT : "<<p1<<endl;
        //cout<<"ANSWER :";
        //FOR(j,0,Sz(test[i][1])-1) printf("%.5lf ",test[i][1][j]);EL;
        cout<<"OUTPUT :";
        FOR(j,0,Sz(outp)-1) printf("%.5lf ",outp[j]);EL;
    }
    //printf("ERROR : %i/%i ~ %.2lf %%\n",res,Sz(test),100.0*res/Sz(test));
    return res;
}

/**==========================================MAIN==================================================*/
int main() {
    //srand(time(NULL));
    READ();
    /** TRAIN AND TEST */
    cnst=1;
    int cnt=0;
    if (cnst) {
        FOR(i,1,times) nn.train(trainset);
    } else {
        do {
            cnt++;
            nn.random();
            FOR(i,1,times) nn.train(trainset);
            printf("J = %.8lf\n",nn.J);
        } while (TEST(nn,testset)>ntest/200);
    }
    if (!cnst) cout<<"RESET TIMES : "<<cnt<<endl;
    /**PRINT STUCTURE */
    //cout<<"=======================STRUCTURE======================"<<endl;
    //nn.p();
    printf("*****************J = %.10lf**************************\n\n",nn.J);
    /** TEST */
    error = TEST(nn,testset);
    //printf("ERROR : %i/%i ~ %.2lf %%\n",error,ntest,100.0*error/ntest);
}
/**=========================INPUT==FORMAT============================================
****HEADER...************************************************
    L : Number of layers
    S[L] : Array of layers' size
****HEADER...************************************************
    N : Number of traning set
    N pair of lines :
        |Input vector of i-th training set
        |Output vector of i-th training set
****HEADER...************************************************
    X : 1/0 integer
    if (X==0), nothing remain in this sector.
    Otherwise :

    L-1 real-number matrixes come.
    The i-th matrix is weight matrix of layer i and layer i+1.
    Size of i-th matrix is S[i+1]xS[i].

    L-1 real-number vectors come.
    The i-th vector is bias vector of layer i+1.
    Size of i-th vector is S[i+1].
****HEADER...************************************************
    Ntest : Number of test.
    Ntest pairs of lines :
        |Input vector of i-th test.
        |Output vector of i-th test
============================OUTPUT==FORMAT===========================================
    ERROR : #failed test/Ntest ~ failed percentage.
    J = Value of cost function
    L-1 matrix and L-1 vector come with the same above format.
=================CODE=WRITTEN=BY=PHAM=QUANG=HUY===================================*/

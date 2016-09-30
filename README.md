# cpuneunet
C++ CPU Neural Network
-----------------------------------------------
Language : C++ 11
IDE : Codeblock
-----------------------------------------------
Third project about neural network.
Implemented from STD library.

============================INPUT==FORMAT===========================================
--HEADER--(Dummy line for easy reading)
    L : Number of layers
    S[L] : Array of layers' size
--HEADER--(Dummy line for easy reading)
    N : Number of traning set
    N pair of lines :
        |Input vector of i-th training set
        |Output vector of i-th training set
--HEADER--(Dummy line for easy reading)
    X : 0 or 1 integer
    if (X==0), nothing remain in this section.
    Otherwise :
        L-1 real-number matrixes come.
        The i-th matrix is weight matrix of layer i and layer i+1.
        Size of i-th matrix is S[i+1]xS[i].

        L-1 real-number vectors come.
        The i-th vector is bias vector of layer i+1.
        Size of i-th vector is S[i+1].
--HEADER--(Dummy line for easy reading)
    Ntest : Number of test.
    Ntest pairs of lines :
        |Input vector of i-th test.
        |Output vector of i-th test
============================OUTPUT==FORMAT===========================================
    Error_perc : #failed test/Ntest ~ failed percentage.
    J = Value of cost function
    L-1 matrix and L-1 vector come with the same above format.
=================CODE=WRITTEN=BY=PHAM=QUANG=HUY===================================*/

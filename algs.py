import math
import random
import copy
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sympy
import string

class PriorityQueue:
    def __init__(self):

        ## List for the queue
        self.Q = []

        ## decrease key needs an easy way to know
        ##  which index the vertex is in
        self.lookup = {}

    def update_index(self, v, i):
        assert (v in self.lookup)

        #print("Setting %c to %d" %(v, i))
        self.lookup[v] = i

    def set_index(self, v, i):
        assert (v not in self.lookup)

        self.lookup[v] = i

    def vertex_from_index(self, i):
        for v in self.lookup:
            if self.lookup[v] == i:
                return v

        ## Invalid index
        assert (1==0)

    def insert(self, v, c):
        """ The values should be unique or
        decreasekey may select the wrong value to decrease"""

        ## Add the new cost to the queue and
        ##  make sure that it is being tracked.
        self.Q.append(c)
        self.set_index(v, len(self.Q) - 1)

        ch = len(self.Q) - 1

        ## 0 is the min
        while ch > 0:
            ## We want to check if the child is smaller than the parent.
            ## The idea is that it floats up the tree to the point where it
            ## is the minimum in that sub-tree
            par = ((ch + 1) // 2) - 1

            ## If the parent is larger then swap the two values
            if self.Q[par] > self.Q[ch]:
                t = self.Q[ch]
                self.Q[ch] = self.Q[par]
                self.Q[par] = t
                vc = self.vertex_from_index(ch)
                vp = self.vertex_from_index(par)
                self.update_index(vc, par)
                self.update_index(vp, ch)

            ## Float On - Modest Mouse
            ch = par

    def extract_min(self):
        if len(self.Q) == 1:
            c = self.Q.pop()
            v = self.vertex_from_index(0)
            return (c,v)

        c = self.Q[0]
        v = self.vertex_from_index(0)

        ## Remove it from the lookup
        self.lookup.pop(v)

        ## Get the high value vertex
        high_v = self.vertex_from_index(len(self.Q) - 1)

        ## update it to be the 0th index
        self.update_index(high_v, 0)

        ## pop it from the queue
        high = self.Q.pop()

        ## Set it to the root
        self.Q[0] = high

        ## Time to float it down
        par = 0
        
        ## i / 2 will be the bottom level
        while par < int( math.floor(len(self.Q) / 2)):
            ch = (2*par) + 1

            ## choose the smaller child
            if ch < len(self.Q) - 1 and self.Q[ch + 1] < self.Q[ch]:
                ch = ch + 1

            ## If the parent is greater than the child then swap them
            if self.Q[par] > self.Q[ch]:
                t = self.Q[ch]
                self.Q[ch] = self.Q[par]
                self.Q[par] = t

                #print("Updating %c to index %d Value: %d" %(self.vertex_from_index(ch), par, self.Q[par]))
                #print("Updating %c to index %d Value: %d" %(self.vertex_from_index(par), ch, self.Q[ch]))

                vc = self.vertex_from_index(ch)
                vp = self.vertex_from_index(par)
                self.update_index(vc, par)
                self.update_index(vp, ch)
            par = ch
        return (c, v)

    def decrease_key(self, k, value):
        """ Decreases the value with the associated key and floats it up
            if necessary
        """
        if k not in self.lookup:
            return

        ## Get the index
        i = self.lookup[k]

        # Do nothing if it isn't a decrease
        if value >= self.Q[i]:
            return

        self.Q[i] = value

        ch = i

        ## Same code as with insert but just starting at
        ## The decreased key instead of the furthest leaf
        while ch > 0:
            ## We want to check if the child is smaller than the parent.
            ## The idea is that it floats up the tree to the point where it
            ## is the minimum in that sub-tree
            par = ((ch + 1) // 2) - 1

            ## If the parent is larger then swap the two values
            if self.Q[par] > self.Q[ch]:
                t = self.Q[ch]
                self.Q[ch] = self.Q[par]
                self.Q[par] = t
                #print("Updating %c to index %d Value: %d" %(self.vertex_from_index(ch), par, self.Q[par]))
                #print("Updating %c to index %d Value: %d" %(self.vertex_from_index(par), ch, self.Q[ch]))
                vc = self.vertex_from_index(ch)
                vp = self.vertex_from_index(par)
                self.update_index(vc, par)
                self.update_index(vp, ch)
                #print("%c is now %d" %(self.vertex_from_index(ch), self.lookup[self.vertex_from_index(ch)]))

            ## Float On - Modest Mouse
            ch = par

class MaxHeap:
    def __init__(self):

        ## List for the queue
        self.Q = []

        ## decrease key needs an easy way to know
        ##  which index the vertex is in
        self.lookup = {}

    def empty(self):
        return len(self.Q) == 0

    def update_index(self, v, i):
        assert (v in self.lookup)

        #print("Setting %c to %d" %(v, i))
        self.lookup[v] = i

    def set_index(self, v, i):
        assert (v not in self.lookup)

        self.lookup[v] = i

    def vertex_from_index(self, i):
        for v in self.lookup:
            if self.lookup[v] == i:
                return v

        ## Invalid index
        assert (1==0)

    def insert(self, v, c):
        """ The values should be unique or
        decreasekey may select the wrong value to decrease"""

        ## Add the new cost to the queue and
        ##  make sure that it is being tracked.
        self.Q.append(c)
        self.set_index(v, len(self.Q) - 1)

        ch = len(self.Q) - 1

        ## 0 is the min
        while ch > 0:
            par = ((ch + 1) // 2) - 1

            ## If the parent is larger then swap the two values
            if self.Q[par] < self.Q[ch]:
                t = self.Q[ch]
                self.Q[ch] = self.Q[par]
                self.Q[par] = t
                vc = self.vertex_from_index(ch)
                vp = self.vertex_from_index(par)
                self.update_index(vc, par)
                self.update_index(vp, ch)

            ## Float On - Modest Mouse
            ch = par

    def extract_max(self):
        if len(self.Q) == 1:
            c = self.Q.pop()
            v = self.vertex_from_index(0)
            return (c,v)

        c = self.Q[0]
        v = self.vertex_from_index(0)

        ## Remove it from the lookup
        self.lookup.pop(v)

        ## Get the high value vertex
        high_v = self.vertex_from_index(len(self.Q) - 1)

        ## update it to be the 0th index
        self.update_index(high_v, 0)

        ## pop it from the queue
        high = self.Q.pop()

        ## Set it to the root
        self.Q[0] = high

        ## Time to float it down
        par = 0
        
        ## i / 2 will be the bottom level
        while par < int( math.floor(len(self.Q) / 2)):
            ch = (2*par) + 1

            ## choose the larger child
            if ch < len(self.Q) - 1 and self.Q[ch + 1] > self.Q[ch]:
                ch = ch + 1

            if self.Q[par] < self.Q[ch]:
                t = self.Q[ch]
                self.Q[ch] = self.Q[par]
                self.Q[par] = t

                #print("Updating %c to index %d Value: %d" %(self.vertex_from_index(ch), par, self.Q[par]))
                #print("Updating %c to index %d Value: %d" %(self.vertex_from_index(par), ch, self.Q[ch]))

                vc = self.vertex_from_index(ch)
                vp = self.vertex_from_index(par)
                self.update_index(vc, par)
                self.update_index(vp, ch)
            par = ch
        return (c, v)

    def increase_key(self, k, value):
        """ Decreases the value with the associated key and floats it up
            if necessary
        """
        if k not in self.lookup:
            return

        ## Get the index
        i = self.lookup[k]

        # Do nothing if it isn't an increase
        if value <= self.Q[i]:
            return

        self.Q[i] = value

        ch = i

        ## Same code as with insert but just starting at
        ## The decreased key instead of the furthest leaf
        while ch > 0:
            par = ((ch + 1) // 2) - 1

            if self.Q[par] < self.Q[ch]:
                t = self.Q[ch]
                self.Q[ch] = self.Q[par]
                self.Q[par] = t
                #print("Updating %c to index %d Value: %d" %(self.vertex_from_index(ch), par, self.Q[par]))
                #print("Updating %c to index %d Value: %d" %(self.vertex_from_index(par), ch, self.Q[ch]))
                vc = self.vertex_from_index(ch)
                vp = self.vertex_from_index(par)
                self.update_index(vc, par)
                self.update_index(vp, ch)

            ## Float On - Modest Mouse
            ch = par

class COSC031:
    def __init__(self, seed=random.randint(0, 2**64 - 1) ):
        self.seed = seed

        print('[INFO] SEED: {0}'.format(seed))

        random.seed(seed)

    def SetSeed(self, seed ):
        self.seed = seed

        print('[INFO] SEED: {0}'.format(seed))

        random.seed(seed)

    # O(n^2)
    def InsertionSort(self,  A ):
        Ac = copy.deepcopy(A)

        for j in range(1, len(Ac)):
            i = j - 1
            key = Ac[j]

            while i >= 0 and Ac[i] > key:
                Ac[i+1] = Ac[i]
                i -= 1
                
            Ac[i+1] = key 

        return Ac

    def RandomArray(self, n = 10, a = 0, b = 100):
        A = []

        for i in range(n):
            c = random.randint(a, b)
            A.append(c)

        return A

    def RandomArrayDistinct(self, n = 10, a = 0, b = 100):
        A = []

        ## If the range is smaller than the number available throw an error
        if b - a < n:
            print("[ERROR] Unable to generate %d distinct elements" %(n))
            sys.exit(1)

        while len(A) < n:
            c = random.randint(a, b)

            if c in A:
                continue

            A.append(c)

        return A

    def RandomDistinctPoints(self, n = 10, maxx = 20, maxy = 20):

        pts = []

        cnt = 0

        while cnt < n:
            x = random.randint(0, maxx)
            y = random.randint(0, maxy)

            if ( (x, y) not in pts):
                pts.append((x,y))
                cnt += 1

        return pts

    def RandomPoints(self, n = 10, maxx = 20, maxy = 20):

        pts = []

        cnt = 0

        while cnt < n:
            x = random.randint(0, maxx)
            y = random.randint(0, maxy)

            pts.append((x,y))
            cnt += 1

        return pts

    def ArrayToNByN( self, A ):
        n = int(math.sqrt(len(A)))

        B = []

        while len(A):
            B.append(A[:n])
            A = A[n:]

        return B

    def PrintGrid(self, A):
        s = ''

        for x in A:
            for y in x:
                s += '%d\t' %(y)
            s += '\n'

        print(s)


    def Merge(self, B, C):
        A = []

        i = 0
        j = 0

        while i < len(B) and j < len(C):
            if B[i] < C[j]:
                A.append(B[i])
                i += 1
            else:
                A.append(C[j])
                j += 1

        if i == len(B):
            A += C[j:]
        else:
            A += B[i:]

        return A

    def CountInvNaive(self, A):
        count = 0

        for i in range(len(A) - 1):
            for j in range(1, len(A)):
                if A[i] > A[j]:
                    count += 1

        return count

    def MergeSortIns(self, A, p):
        if len(A) == 1:
            return A

        if len(A) <= p:
            return self.InsertionSort(A)

        n = int(math.floor(len(A)/2))
        B = self.MergeSortIns(A[0:n], p)
        C = self.MergeSortIns(A[n:], p)

        return self.Merge(B, C)

    def BuildMaxHeap(self, A):
        ## The first element is already "sorted" so start at 1
        for h in range(1, len(A)):
            ch = h

            comps = 0
            ## 0 is the max
            while ch > 0:

                ## We want to check if the child is greater than the parent.
                ## The idea is that it floats up the tree to the point where it
                ## is the maximum in that sub-tree
                par = int( math.floor(ch/2) )

                ## If the parent is smaller then swap the two values
                comps += 1
                if A[par] < A[ch]:
                    t = A[ch]
                    A[ch] = A[par]
                    A[par] = t

                ## Float On - Modest Mouse
                ch = par

        return A

    ## SEED: 13296911453475430226
    ## [88, 86, 75, 69, 64, 27, 52, 65, 0, 56, 0, 11, 33, 39, 23]
    ##                      88
    ##             86               75
    ##         69      64       27      52
    ##      65    0  56   0   11  33  39  23
    ##
    def HeapSort(self, A):
        Amax = self.BuildMaxHeap(A)

        print(Amax)
        B = []

        for h in range(len(Amax) - 1, 0, -1):
            ## Repeatedly extracts the maximum value
            B.append(Amax[0])

            ## Now, rebalance the tree by pushing down the "last" node
            Amax[0] = Amax[h]

            ## The last element is no longer relevant so use h - 1
            i = h - 1
            par = 0

            comps = 0
            ## i / 2 will be the bottom level
            while par <= int( math.floor(i / 2)):
                ch = (2*par) + 1

                comps += 1
                ## choose the larger child
                if ch < i and Amax[ch + 1] > Amax[ch]:
                    ch = ch + 1

                comps += 1
                ## If the parent is less than the child then swap them
                if Amax[par] < Amax[ch]:
                    t = Amax[ch]
                    Amax[ch] = Amax[par]
                    Amax[par] = t

                par = ch

        B.append(Amax[0])

        B.reverse()
        return B


    ## Array must have distinct elements
    def FindLocalMinimum(self, A):
        if len(A) == 1:
            return A[0]

        ## Find the midpoint
        mid = int(math.floor(len(A)/2))

        ## Check if it is a local minimum
        if A[mid] < A[mid - 1]:
            return self.FindLocalMinimum(A[mid:])
        else:
            return self.FindLocalMinimum(A[:mid])
    
    def GetMajorityMember(self, A):
        ## Find just a single majority member.
        ##    Each level performs n/2 introductions
        ##    The worst case scenario is if the size of the new array is (n/2) + 1
        ##    Runs in O(logn)

        if len(A) == 1:
            return A[0]

        B = []

        for i in range(0, len(A)-1, 2):
            if A[i] == A[i+1]:
                B.append(A[i])

        if len(A) % 2:
            B.append(A[-1])

        return self.GetMajorityMember(B)

    def FindMajorityParty(self, A):
        maj = self.GetMajorityMember(A)

        B = []

        for i in range(len(A)):
            if A[i] == maj:
                B.append(A[i])

        return B

    def PrintCols(self, A, B):
        for x in range(len(A)-1, 0, -1):
            print("%s\t\t%s" %(str(A[x]), str(B[x])))

    ## Input is an Array sorted by x then by y
    def MaximalPointsRec(self, A ):

        if len(A) <= 1:
            return A

        mid = math.ceil(len(A)/2)

        L = A[:mid]
        R = A[mid:]

        Lm = self.MaximalPointsRec(L)
        Rm = self.MaximalPointsRec(R)

        print('*'*5)
        print(Lm)
        print(Rm)
        print('*'*5)

        print("Lm", Lm)
        print("Rm", Rm)

        i = 0
        B = []
        while i < len(Lm):
            if Lm[i][1] > Rm[0][1]:
                B.append(Lm[i])
            i += 1

        B += Rm

        print("B", B)

        return B

    def MaximalPoints(self, A):
        A.sort()

        print(A)

        return self.MaximalPointsRec(A)

    def SubsetSum(self, L, t):
        A = np.zeros((len(L )+ 1, t + 1))

        for i in range(len(L) + 1):
            A[i, 0] = 1

        print(L)

        for i in range(1, len(L) + 1):
            for u in range(1, t+1):
                if L[i-1] > u:
                    A[i, u] = A[i-1, u]
                else:
                    A[i, u] = A[i-1, u] or A[i - 1, u - L[i-1]]

    def SubsetSum2(self, L, t):
        A = np.zeros((len(L )+ 1, t + 1))

        for i in range(len(L) + 1):
            A[i, 0] = 1

        print(L)

        for i in range(1, len(L) + 1):
            for u in range(1, t+1):
                k = 0
                while (k*L[i-1] <= u):
                    A[i,u] += A[i-1, u - k*L[i-1]]
                    k += 1

        print(A)

    # v - number of vertices
    # e - number of edges
    # minw - mininum weight
    # maxw - maximum weight
    ## Returns an Adjacency Matrix with associated costs
    def GenDirectedWeightedGraph(self, v, e, minw, maxw):
        V = np.zeros((v,v))

        C = np.full((v,v), np.inf)

        count = 0

        while count < e:
            ## choose two vertices
            v_1 = random.randint(0, v-1)

            v_2 = v_1

            while v_2 == v_1:
                v_2 = random.randint(0, v-1)

            ## There is already an edge
            if V[v_1, v_2] == 1:
                continue

            V[v_1, v_2] = 1
            C[v_1, v_2] = random.randint(minw, maxw)
            count += 1

        return (V, C)

    def edgesToWeightedAdjList( self, V, E):
        G = {}

        for i in range(V):
            G[i+1] = []

        for e in E:
            u, v, c = e
            if u not in G:
                G[u] = []
            G[u].append( (v, c))

        return G

    ## The adjacency map should also have
    ##   the cost for each edge too.
    def Dijkstra_SSSP(self, Adj, s):
        d = {}
        par = {}

        assert s in Adj

        for V in Adj:
            d[V] = np.inf
            par[V] = None

        d[s] = 0

        Q = PriorityQueue()

        for v in Adj:
            Q.insert(v, d[v])

        while len(Q.Q):
            c, u = Q.extract_min()

            for e in Adj[u]:
                v, c = e
                if d[v] > d[u] + c:
                    d[v] = d[u] + c
                    par[v] = u

                Q.decrease_key(v, d[u] + c)

        return d, par

    def genReverseWeightedG(self, Adj):
        RAdj = {}

        for v in Adj:
            for e in Adj[v]:
                if e[0] not in RAdj:
                    RAdj[e[0]] = []

                RAdj[e[0]].append(v)

        return RAdj 

    def cost(self, Adj, u, v):
        if u not in Adj:
            return np.inf

        for x in Adj[u]:
            e, c = x

            if e == v:
                return c

        return np.inf

    ## It assumes that the vertices are 1 indexed
    def Bellman_Ford_SSSP(self, Adj, s):
        RAdj = self.genReverseWeightedG(Adj)

        OPT = np.full( (len(Adj), len(Adj) + 1), np.inf)
        parent = np.full( (len(Adj) + 1), None)

        OPT[0, s] = 0

        ## The outer loop goes over the possible lengths of a path
        ## since a 0 length path only reaches s it is set to 0 all others
        ## will be infinity. Also, since a path must contain unique vertices
        ##  the longest that it can possibly be is n - 1
        for k in range(1, len(Adj) ):
            ## loop over each vertex
            for v in range(1, len(Adj) + 1):
                # Since all costs are > 0 then the cheapest cost will be
                ## what we already found. So if there is a path from s to t of length l
                ## then adding an additional allowable hop will not make it shorter.
                OPmailT[k, v] = OPT[k - 1, v]

                ## if v has no incoming edges then skip it
                if v not in RAdj:
                    continue

                ## Loop over each vertex
                for u in range(1, len(Adj) + 1):
                    ## Make sure that there is a path from u to v
                    if u not in RAdj[v]:
                        continue

                    ## If the path to u plus the cost of the path from u to v is currently
                    ### less than the already found path from s to v then update it.
                    if OPT[k-1, u] + self.cost(Adj, u, v) < OPT[k-1, v]:
                        OPT[k, v] = OPT[k-1, u] + self.cost(Adj, u, v)
                        parent[v] = u

        return (OPT, parent)


    def bestHops(self, Adj, s, t):
        d = {}
        par = {}

        assert s in Adj

        for V in Adj:
            d[V] = np.NINF
            par[V] = None

        d[s] = np.inf

        Q = MaxHeap()

        pi = []

        ## Everything starts at negative infinity initially except s
        for v in Adj:
            Q.insert(v, d[v])

        while len(Q.Q):
            ## Take out the best next vertex
            ## Initally this will just be s
            c_u, u = Q.extract_max()

            print('Extracted ', (u,c_u))
            ## For each edge from u lets examine the cost
            for e in Adj[u]:
                v, c_v = e

                print('Checking from {0} to {1} {2}'.format(s,v,c_v))

                ## The cost to this point is d[u]
                ## The cost to dv is the minimum of d[u] and c
                ## Only update if this is better than the current one
                tc = min(c_u, c_v)

                ## Set the cost 
                if tc > d[v]:
                    print("SETTING V", v)
                    d[v] = tc
                    par[v] = u

                print("increasing by ",v,  tc)
                Q.increase_key(v, tc)

        tp = par[t]

        pi.append(t)

        while tp != None:
            pi.insert(0, tp)
            tp = par[tp]

        print(pi)
        return d, par

    # O(n^2)
    def adjacencyMatrixToList(self, AdjM):
        AdjL = {}

        for row in range(len(AdjM)):
            AdjL[row+1] = []
            for col in range(len(AdjM[row])):
                if AdjM[row][col] != 0:
                    ## The col + 1 makes it 1 indexed instead of 0
                    AdjL[row+1].append( (col + 1, AdjM[row][col]))

        return AdjL

    ## Calculates the profit per edge and returns an adjacency matrix representing the result
    def calcProfit(self, AdjP, AdjT, desiredProfit):
        AdjTP = []

        for row in range(len(AdjP)):
            AdjTP.append([])

            for col in range(len(AdjP[row])):
                if AdjT[row][col] == 0:
                    pft = 0
                else:
                    pft = (AdjP[row][col] / AdjT[row][col]) - desiredProfit
                AdjTP[row].append(pft)

        return AdjTP

    ## Takes an adjacency matrix p of profit
    ## Takes an adjacency matrix t of time
    ## an expected dollars per hour
    ## a starting node (s)
    def ferrymanProblemBad(self, AdjP, AdjT, dph, s):
        AdjTP = self.calcProfit(AdjP, AdjT, dph)
        AdjList = self.adjacencyMatrixToList(AdjTP)

        RAdj = self.genReverseWeightedG(AdjList)

        OPT = np.full( (2*len(AdjList), len(AdjList) + 1), np.NINF)
        parent = np.full( (2*len(AdjList), len(AdjList) + 1), None)

        OPT[0, s] = 0

        ## The outer loop goes over the possible lengths of a path
        ## since a 0 length path only reaches s it is set to 0 all others
        ## will be infinity. Also, since a path must contain unique vertices
        ##  the longest that it can possibly be is n - 1
        for k in range(1, 2*(len(AdjList))):
            ## loop over each vertex
            for v in range(1, len(AdjList) + 1):
                # Since all costs are > 0 then the cheapest cost will be
                ## what we already found. So if there is a path from s to t of length l
                ## then adding an additional allowable hop will not make it shorter.
                OPT[k, v] = OPT[k - 1, v]

                ## if v has no incoming edges then skip it
                if v not in RAdj:
                    continue

                ## Loop over each vertex
                for u in range(1, len(AdjList) + 1):
                    ## Make sure that there is a path from u to v
                    if u not in RAdj[v]:
                        continue

                    ## If the path to u plus the cost of the path from u to v is currently
                    ### less than the already found path from s to v then update it.
                    if OPT[k-1, u] + self.cost(AdjList, u, v) > OPT[k-1, v]:
                        OPT[k, v] = OPT[k-1, u] + self.cost(AdjList, u, v)
                        parent[k,v] = u
            if OPT[k, s] > 0:
                print('Found path at length %d' %(k))
                break

        if parent[k,s] == None:
            return OPT, None

        path = [s]
        c = parent[k,s]

        while c != None:
            path.insert(0,c)

            k = k - 1

            c = parent[k, c]

        print(path)
        return OPT, path

    def ferrymanProblem(self, AdjP, AdjT, dph, s):
        AdjPL = self.adjacencyMatrixToList(AdjP)

        RAdj = self.genReverseWeightedG(AdjPL)

        OPT = []

        for i in range(2*len(AdjPL)):
            a = []
            for j in range(len(AdjPL) + 1):
                a.append((0, 0))
            OPT.append(a)

        parent = np.full( (2*len(AdjPL), len(AdjPL) + 1), None)

        #OPT[0][s] = (0, 0)

        ## The outer loop goes over the possible lengths of a path
        ## since a 0 length path only reaches s it is set to 0 all others
        ## will be infinity. Also, since a path must contain unique vertices
        ##  the longest that it can possibly be is n - 1
        for k in range(1, 2*(len(AdjPL))):
            ## loop over each vertex
            for v in range(1, len(AdjPL) + 1):
                # Since all costs are > 0 then the cheapest cost will be
                ## what we already found. So if there is a path from s to t of length l
                ## then adding an additional allowable hop will not make it shorter.
                OPT[k][v] = OPT[k - 1][v]

                ## if v has no incoming edges then skip it
                if v not in RAdj:
                    continue

                ## Loop over each vertex
                for u in range(1, len(AdjPL) + 1):
                    ## Make sure that there is a path from u to v
                    if u not in RAdj[v]:
                        continue

                    print(u,v)

                    numo, deno = OPT[k-1][u]

                    if deno == 0:
                        rez = 0
                    else:
                        rez = numo / deno

                    new_profit = (numo + AdjP[u-1][v-1]) / (deno + AdjT[u-1][v-1]) - dph

                    print(new_profit)
                    if new_profit > rez:
                        OPT[k][v] = (OPT[k-1][u][0] + AdjP[u-1][v-1], OPT[k-1][u][1] + AdjT[u-1][v-1])
                        parent[k,v] = u

            numo, deno = OPT[k][s]

            if deno == 0:
                rez = 0
            else:
                rez = numo / deno

            if rez > 0:
                print('Found path at length %d' %(k))
                break

        if parent[k,s] == None:
            return OPT, None

        path = [s]
        c = parent[k,s]

        while c != None:
            path.insert(0,c)

            k = k - 1

            c = parent[k, c]

        print(path)
        return OPT, path
def rp72():
    my = COSC031()

    V = 6
    E = []

    ## u,v,c
    E.append( (1,2,3) )
    E.append( (2,3,7) )
    E.append( (3,6,5) )
    E.append( (6,5,3) )
    E.append( (1,4,2) )
    E.append( (4,5,2) )

    G = my.edgesToWeightedAdjList(V, E)

    #print(G)

    d, par = my.bestHops(G, 1, 5)

    print(d)

    print(par)

my = COSC031()



## Hours adjacency matrix
AdjT = [ [ 0, 6, 1, 0, 0, 0, 0, 0, 0, 0 ],
         [ 8, 0, 0, 0, 0, 0, 3, 0, 0, 0 ],
         [ 0, 0, 0, 7, 0, 0, 0, 0, 0, 0 ],
         [ 0, 0, 0, 0, 0, 0, 0, 4, 4, 0 ],
         [ 0, 0, 0, 5, 0, 0, 0, 0, 0, 0 ],
         [ 5, 0, 9, 0, 0, 0, 0, 0, 0, 0 ],
         [ 0, 0, 0, 5, 0, 3, 0, 0, 0, 0 ],
         [ 0, 0, 0, 0, 0, 0, 8 ,0, 0, 0 ],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 9 ],
         [ 0, 0, 0, 0, 8, 0, 0, 4, 0, 0 ]]

AdjP = [ [ 00, 18,  5, 00, 00, 00, 00, 00, 00, 00 ],
         [ 32, 00, 00, 00, 00, 00, 15, 00, 00, 00 ],
         [ 00, 00, 00, 35, 00, 00, 00, 00, 00, 00 ],
         [ 00, 00, 00, 00, 00, 00, 00, 24, 20, 00 ],
         [ 00, 00, 00, 25, 00, 00, 00, 00, 00, 00 ],
         [ 15, 00, 27, 00, 00, 00, 00, 00, 00, 00 ],
         [ 00, 00, 00, 40, 00, 33, 00, 00, 00, 00 ],
         [ 00, 00, 00, 00, 00, 00, 32, 00, 00, 00 ],
         [ 00, 00, 00, 00, 00, 00, 00, 00, 00, 27 ],
         [ 00, 00, 00, 00, 24, 00, 00, 20, 00, 00 ]]

"""
AdjT = [ [ 0,  1,  0,  0],
         [ 0,  0,  1,  0],
         [ 1,  1,  0,  1],
         [ 0,  0,  1,  0],
]

AdjP = [ [ 0, 10,  0,  0],
         [10,  0, 10,  0],
         [ 0, 10,  0, 10],
         [ 0,  0,  10, 0],
]
"""

AdjT = [ [ 0,  1,  0,  0,  0,  0],
         [ 0,  0,  1,  0,  0,  0],
         [ 0,  0,  0,  1,  0,  0],
         [ 0,  0,  0,  0,  1,  0],
         [ 1,  0,  0,  0,  0,  1],
         [ 0,  0,  0,  0,  1,  0]
]

AdjP = [ [ 0, 10,  0,  0,  0,  0],
         [ 0,  0, 10,  0,  0,  0],
         [ 0,  0,  0, 10,  0,  0],
         [ 0,  0,  0,  0, 10,  0],
         [ 10, 0,  0,  0,  0, 10],
         [ 0,  0,  0,  0, 10,  0]
]


## I picked this after seeing what kind of numbers I got from AdjP/AdjT
desiredProfit = 40
AdjTP = my.calcProfit(AdjP, AdjT, desiredProfit)



#print(my.adjacencyMatrixToList(AdjTP))

s = 3

opt, path = my.ferrymanProblem(AdjP, AdjT, desiredProfit, s)
for l in opt:
    print(l)

print(path)

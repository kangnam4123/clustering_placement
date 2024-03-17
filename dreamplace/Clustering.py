##
# @file   Clustering.py
# @author Hyeonwoo Park
# @date   DEC 2023
# @brief  Make Clustering Result using igraph
#

import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
import numpy as np
import logging
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import dreamplace.configure as configure
import Params
import PlaceDB
import Timer
import NonLinearPlace
import pdb
#내가 추가한 거
import igraph as ig
#from igraph import Graph
#from igraph import community_leiden

def name2id_map2str(m):
    id2name_map = [None]*len(m)
    for k in m.keys():
        id2name_map[m[k]] = k
    content = ""
    for i in range(len(m)):
        if i:
            content += ", "
        content += "%s : %d" % (id2name_map[i], i)
    return "{%s}" % (content)

def array2str(a):
    content = ""
    for v in a:
        if content:
            content += ", "
        content += "%s" % (v)
    return "[%s]" % (content)

def clustering(params):
    """
    @brief parameter update and parsing
    @param params parameters
    """

    assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
            "CANNOT enable GPU without CUDA compiled"

    np.random.seed(params.random_seed)
    # read database
    tt = time.time()
    placedb = PlaceDB.PlaceDB()
    placedb(params)
   
    content = ""
    content += "num_physical_nodes = %s\n" % (placedb.num_physical_nodes)
    content += "num_movable_nodes = %s\n" % (placedb.num_movable_nodes)
    content += "num_fixed_macros = %s\n" % (placedb.num_terminals)
    #content += "node_name2id_map = %s\n" % (name2id_map2str(placedb.node_name2id_map))
    #content += "node_names = %s\n" % (array2str(placedb.node_names))
    #content += "node_x = %s\n" % (placedb.node_x)
    #content += "node_y = %s\n" % (placedb.node_y)
    #content += "node_orient = %s\n" % (array2str(placedb.node_orient))
    #content += "node_size_y = %s\n" % (placedb.node_size_y)
    #content += "pin_direct = %s\n" % (array2str(placedb.pin_direct))
    #content += "pin_offset_x = %s\n" % (placedb.pin_offset_x)
    #content += "pin_offset_y = %s\n" % (placedb.pin_offset_y)
    #content += "net_name2id_map = %s\n" % (name2id_map2str(placedb.net_name2id_map))
    #content += "net_weights = %s\n" % (placedb.net_weights)
    #content += "net_names = %s\n" % (array2str(placedb.net_names))
    #content += "net2pin_map = %s\n" % (placedb.net2pin_map)
    #content += "flat_net2pin_map = %s\n" % (placedb.flat_net2pin_map)
    #content += "flat_net2pin_start_map = %s\n" % (placedb.flat_net2pin_start_map)
    #content += "node2pin_map = %s\n" % (placedb.node2pin_map)
    #content += "flat_node2pin_map = %s\n" % (placedb.flat_node2pin_map)
    #content += "flat_node2pin_start_map = %s\n" % (placedb.flat_node2pin_start_map)
    #content += "pin2node_map = %s\n" % (placedb.pin2node_map)
    #content += "pin_name2id_map = %s\n" % (name2id_map2str(placedb.pin_name2id_map))
    #content += "pin2net_map = %s\n" % (placedb.pin2net_map)
    #content += "rows = %s\n" % (placedb.rows)
    content += "xl = %s\n" % (placedb.xl)
    content += "yl = %s\n" % (placedb.yl)
    content += "xh = %s\n" % (placedb.xh)
    content += "yh = %s\n" % (placedb.yh)
    #content += "row_height = %s\n" % (placedb.row_height)
    #content += "site_width = %s\n" % (placedb.site_width)
    #content += "num_movable_pins = %s\n" % (placedb.num_movable_pins)
    content += "total cell area = %s\n" %(placedb.total_movable_node_area)

    #Macro cell을 인지해서(height가 몇배 이상이면 macro cell) cell list를 만들어서 macro cell이면 1로 만들기
    #db에 있는 data 쪼개서 graph넣어줄 수 있는 형태로 바꾸기
    isMacro = [0] * placedb.num_movable_nodes
    stdheight = min(placedb.node_size_y[0:placedb.num_movable_nodes])
    macronum = 0
    for i in range(placedb.num_movable_nodes):
        if(placedb.node_size_y[i] > stdheight):
            isMacro[i] = 1
            macronum +=1
    #Macro cell과 연결되어 있는 cell들은 1로 표시하기(Macro cell 중복돼도 괜찮)
    isMacro1 = [0] * placedb.num_movable_nodes
    for index, value in enumerate(isMacro):
        if value ==1:
            pins = placedb.node2pin_map[index]
            for pin in pins:
                pins2 = placedb.net2pin_map[placedb.pin2net_map[pin]]
                for pin2 in pins2:
                    isMacro1[placedb.pin2node_map[pin2]] = 1
    edge = []
    edge_weight = []
    Node_weight = Node_weight = [[] for _ in range(placedb.num_movable_nodes)]
    macro_weight=4
    for j in range(len(placedb.net_names)):
        if(len(placedb.net2pin_map[j])<2 or len(placedb.net2pin_map[j])> 50):
            continue
        else:
            weight = placedb.net_weights[j] / (len(placedb.net2pin_map[j])-1)
            for k in range(len(placedb.net2pin_map[j])-1):
                num1 = placedb.net2pin_map[j][k]
                Node1 = placedb.pin2node_map[num1]
                for l in range(k+1,len(placedb.net2pin_map[j])):
                    num2 = placedb.net2pin_map[j][l]
                    Node2 = placedb.pin2node_map[num2]
                    if((Node1 == Node2)):
                        continue
                    else:
                        #macro와 연결된 node일 경우 edge weight를 4배 늘린다.
                        #Macro와 두다리 연결되어 있는 node의 경우 edge weight를 2배 늘린다.
                        if((Node2 < placedb.num_movable_nodes)):
                            if((Node1 < placedb.num_movable_nodes)):
                                edge.append([Node1,Node2])
                                if(isMacro[Node2]==1 or isMacro[Node1]==1):
                                    macro_weight = 4
                                else:
                                    if(isMacro1[Node2]==1 or isMacro1[Node1]==1):
                                        macro_weight = 2
                                    else:
                                        macro_weight = 1
                                edge_weight.append(weight*macro_weight)
                                Node_weight[Node1].append([Node2,weight*macro_weight])
                                Node_weight[Node2].append([Node1,weight*macro_weight])
                            else:
                                if(isMacro[Node2]==1):
                                    macro_weight = 4
                                else:
                                    macro_weight = 2 if isMacro1[Node2]==1 else 1
                                    Node_weight[Node2].append([Node1,weight*macro_weight])
                        else:
                            if((Node1 < placedb.num_movable_nodes)):
                                if(isMacro[Node1]==1):
                                     macro_weight = 4
                                else:
                                    macro_weight = 2 if isMacro1[Node1]==1 else 1
                                    Node_weight[Node1].append([Node2,weight*macro_weight])

    #content += "edge = %s\n" % (edge)
    #content += "edge_weight = %s\n" % (edge_weight)
    content += "macronum = %s\n" % (macronum)
    print(content)

    tt1 = time.time()
    logging.info("reading database takes %.2f seconds" % (tt1 - tt))

    #graph 만들기
    g = ig.Graph(
        n= placedb.num_movable_nodes,
        edges = edge,
        edge_attrs={'weight' : edge_weight}
    )
    #print(g)
    #clustering 진행하기
    Clusteringresults= ig.Graph.community_leiden(g, objective_function="modularity", weights=edge_weight, resolution=1.0, beta=0.01, initial_membership=None, n_iterations=2, node_weights=None)
    tt2 = time.time()
    #print(Clusteringresults)
    logging.info("Clustering takes %.2f seconds" % (tt2 - tt1))
    
    #결과 출력하기
    """
    for i, community in enumerate(Clusteringresults):
        print(f"Community {i}:")
        for v in community:
            print(f"\t{v}")
    """
    #Cluster 수 중에서 갯수가 적은 Cluster는 없애기 - Cluster에 마지막 하나 추가해서 unplace된 애들 label 붙이기
    Clusters = []
    Clusterarea = []
    Clusternum=0
    totalclusterarea = 0
    Nodetocluster = [0] * placedb.num_movable_nodes
    rest = np.array([],dtype=np.int32)
    for i, community in enumerate(Clusteringresults):
        arr = np.array([],dtype=np.int32)
        area = 0
        if(len(community))<10:
            for j in community:
                rest = np.append(rest,j)
            continue
        else:
            for v in community:
                arr = np.append(arr,v)
                Nodetocluster[v]= Clusternum
                area += placedb.node_size_x[v] * placedb.node_size_y[v]
            Clusters.append(arr)
            Clusterarea.append(area)
            totalclusterarea += area
            Clusternum +=1

    for k in rest:
        Nodetocluster[k] = Clusternum

    #Cluster 결과 확인하기
    """
    for idx, array in enumerate(Clusters):
        print(f"Cluster{idx}: {array}")
        print(Clusterarea[idx])
    print(rest)
    print(Nodetocluster)
    print(Clusternum)
    """

    #Cluster 결과로 행렬 만들기 (A,Bx,By)
    A = np.zeros((len(Clusters), len(Clusters)),dtype=placedb.dtype)
    Bx = np.zeros((len(Clusters),1),dtype=placedb.dtype)
    By = np.zeros((len(Clusters),1),dtype=placedb.dtype)
    for i in range(len(Clusters)):
        for j in range(len(Clusters[i])):
            for k in range(len(Node_weight[Clusters[i][j]])):
                if(Node_weight[Clusters[i][j]][k][0] >=placedb.num_movable_nodes):
                    Bx[i] += (Node_weight[Clusters[i][j]][k][1] * placedb.node_x[Node_weight[Clusters[i][j]][k][0]])
                    By[i] += (Node_weight[Clusters[i][j]][k][1] * placedb.node_y[Node_weight[Clusters[i][j]][k][0]])
                    A[i,i] += Node_weight[Clusters[i][j]][k][1]
                else:
                    if(Nodetocluster[Node_weight[Clusters[i][j]][k][0]]<Clusternum and Nodetocluster[Node_weight[Clusters[i][j]][k][0]]!=i):
                        A[i,Nodetocluster[Node_weight[Clusters[i][j]][k][0]]] -=  Node_weight[Clusters[i][j]][k][1]
                        A[i,i] += Node_weight[Clusters[i][j]][k][1]
                    else:
                        if(Nodetocluster[Node_weight[Clusters[i][j]][k][0]]==Clusternum and Nodetocluster[Node_weight[Clusters[i][j]][k][0]]!=i):
                            Bx[i] += (Node_weight[Clusters[i][j]][k][1] * (placedb.xl * 1.0 + placedb.xh * 1.0) / 2)
                            By[i] += (Node_weight[Clusters[i][j]][k][1] * (placedb.yl * 1.0 + placedb.yh * 1.0) / 2)
                            A[i,i] += Node_weight[Clusters[i][j]][k][1]

    #Cluster 중에 외부와 연결이 안되어있는 경우 singular matrix가 되기 때문에 이를 방지하기 위해 빼주기 - Clusters에서도 빼줘야 한다.

    rows_with_zeros = np.all(A == 0, axis=1)
    A = np.delete(A, np.where(rows_with_zeros), axis=0)
    A = np.delete(A, np.where(rows_with_zeros), axis=1)
    Bx = np.delete(Bx, np.where(rows_with_zeros), axis=0)
    By = np.delete(By, np.where(rows_with_zeros), axis=0)
    indices = np.where(rows_with_zeros)[0]
  
    for i in reversed(indices):
        print(i)
        del Clusters[i]
        totalclusterarea -= Clusterarea[i]
        del Clusterarea[i]
        Clusternum -=1
    
    print(Clusternum)

    for row in A:
        print(row)
    print(Bx)
    print(By)

    AverageClusterarea = totalclusterarea / Clusternum
    print(AverageClusterarea)
    #Cluster size에 따라서 weight scaling 하기
    #1번째 함수
    """
    for i in range(len(Clusters)-1):
        for j in range(i+1,len(Clusters)):
            if (Clusterarea[i] < AverageClusterarea * 0.1 and Clusterarea[j] < AverageClusterarea * 0.1):
                A[i,j] = A[i,j] * 
                A[j,i] = A[j,i] * 
            else:
                if ((Clusterarea[i] < AverageClusterarea * 0.1 and Clusterarea[j] < totalclusterarea * 0.1) or (Clusterarea[j] < AverageClusterarea * 0.1 and Clusterarea[i] < totalclusterarea * 0.1)):
                    A[i,j] = A[i,j] * 
                    A[j,i] = A[j,i] * 
            if(Clusterarea[i] > totalclusterarea * 0.1 and Clusterarea[j]>totalclusterarea * 0.1):
                A[i,j] = A[i,j] * 
                A[j,i] = A[j,i] * 
            else :
                if((Clusterarea[i] > totalclusterarea * 0.1 and Clusterarea[j]>AverageClusterarea * 0.1) or (Clusterarea[j] > totalclusterarea * 0.1 and Clusterarea[i]>AverageClusterarea * 0.1)):
                    A[i,j] = A[i,j] * 
                    A[j,i] = A[j,i] * 
    """
    print(f"Cluster weight update")
    #2번째 함수
    for i in range(len(Clusters)-1):
        for j in range(i+1,len(Clusters)):
            value = AverageClusterarea * ((1/(Clusterarea[i]*Clusterarea[j]))**0.5)
            weightupdate = round(value,3)
            A[i,j] = A[i,j] * weightupdate
            A[j,i] = A[j,i] * weightupdate
            print(weightupdate)

    for row in A:
        print(row)
    print(Bx)
    print(By)        
            
    #행렬 계산하기
    xx = np.linalg.solve(A,Bx)
    xy = np.linalg.solve(A,By)
    
    
    #계산 결과 바탕으로 cell location 정해주기
    initiallocation = np.zeros(placedb.num_movable_nodes * 2, dtype=placedb.dtype)
    initiallocation[0:placedb.num_movable_nodes] = np.random.normal(
                loc=(placedb.xl * 1.0 + placedb.xh * 1.0) / 2,
                scale=(placedb.xh - placedb.xl) * 0.001,
                size=placedb.num_movable_nodes)
    initiallocation[placedb.num_movable_nodes:placedb.num_movable_nodes*2]  =  np.random.normal(
                loc=(placedb.yl * 1.0 + placedb.yh * 1.0) / 2,
                scale=(placedb.yh - placedb.yl) * 0.001,
                size=placedb.num_movable_nodes)
    for i in range(len(Clusters)):
        for j in range(len(Clusters[i])):
            initiallocation[Clusters[i][j]] = xx[i]
            initiallocation[placedb.num_movable_nodes+Clusters[i][j]] = xy[i]
    """
    print(initiallocation)
    """
    tt3 = time.time()
    logging.info("Initial location takes %.2f seconds" % (tt3 - tt2)) 
    return  initiallocation

if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow.
    """
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    params = Params.Params()
    params.printWelcome()
    if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        params.printHelp()
        exit()
    elif len(sys.argv) != 2:
        logging.error("One input parameters in json format in required")
        params.printHelp()
        exit()

    # load parameters
    params.load(sys.argv[1])
    logging.info("parameters = %s" % (params))
    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # run placement
    tt = time.time()
    results = clustering(params)

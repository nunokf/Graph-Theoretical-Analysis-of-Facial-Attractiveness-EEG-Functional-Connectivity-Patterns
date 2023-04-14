import numpy as np
from pymatreader import read_mat
import mne
import mne_connectivity
import pandas as pd
from mne_connectivity import spectral_connectivity_epochs
from scipy import linalg
from itertools import combinations
#import matplotlib as plt
import matplotlib.pyplot as plt

def connectivity(wave, min_epoch, max_epoch, method):
    #store data
    results = {"attractive":{},"unattractive":{}}
    #multiple participants
    for participant in range(1,10):
        if participant!=11:
            data = read_mat(r"C:\Users\nunok\Documents\2021\Attractiveness\EEG_predict\eeg_data\timelock{}.mat".format(participant))
            #subset by attractiveness
            attractive = [i for i in range(len(data["timelock"]["trialinfo"][:,5])) if data["timelock"]["trialinfo"][i,5] == 2]
            #unattractive
            unattractive = [i for i in range(len(data["timelock"]["trialinfo"][:,5])) if data["timelock"]["trialinfo"][i,5] == 1]
            array = data["timelock"]["trial"]
            #604(evaluations)x58(electrodes)x425(epochs)
            #access the desired data as a Numpy array and convert that into a Raw object using mne.io.RawArray
            #requires in this format: (n_epochs, n_channels, n_samples)
            array=np.swapaxes(array,0,2)

            sampling_freq = 250

            info = mne.create_info(ch_names=data["timelock"]["label"],
                               ch_types=['eeg'] * len(data["timelock"]["label"]),
                               sfreq=sampling_freq)

            epochs = mne.EpochsArray(array, info)

            #remove "M1" and "M2" -> out of head
            indx = []
            for i in range(len(epochs.ch_names)):
                if epochs.ch_names[i] == "M1" or epochs.ch_names[i] =="M2":
                    indx.append(i)
            array = np.delete(array, indx, axis=1)

            #attractiveness
            attractive = np.delete(array, attractive, axis = 2)
            unattractive = np.delete(array, unattractive, axis = 2)

            ch_names = [i for i in data["timelock"]["label"] if i not in ["M1","M2"]]
            #again, since removed chanels
            info = mne.create_info(ch_names=ch_names,
                                   ch_types=['eeg'] * len(ch_names),
                                   sfreq=sampling_freq)
            attractive
            epochs = mne.EpochsArray(attractive, info)


            tmin = 0
            # Resting state Functional Connectivity analysis at the sensor level - Davide Aloi
            ### Global Variables ###

            waves={
            "delta" : (1,4),
            "alpha" : (8,13),
            "theta" : (4,8),
            "beta" : (13,30),
            "gamma" : (30,100)
            }
            fmin, fmax = waves[wave]
            min_epochs = min_epoch #Start from epoch n.
            max_epochs = max_epoch #End at epoch n.
            # Get the strongest connections
            #n_con = 124*123 # show up to n_con connections THIS SHOULD BE CHECKED.
            n_con = 124*123
            min_dist = 0  # exclude sensors that are less than 4cm apart THIS SHOULD BE CHECKED
            method = method # Method used to calculate the connectivity matrix


            #Connectivity
            sfreq = epochs.info['sfreq']  # the sampling frequency

            #attractive
            con = mne_connectivity.spectral_connectivity_epochs(epochs[min_epochs:max_epochs], method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
                faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

            baseline = mne_connectivity.spectral_connectivity_epochs(epochs[0:86], method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
                faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

            matrix = con.get_data(output='dense')[:, :, 0]
            baseline = baseline.get_data(output='dense')[:, :, 0]

            matrix = matrix/baseline


            def create_dataframe(matrix):
                df = pd.DataFrame(matrix)
                #df.index.name = 'Index_Name'
                df["Index"] = epochs.ch_names
                df = pd.DataFrame(df).set_index('Index')
                df.columns = epochs.ch_names
                #store results
                return df
            results["attractive"]["p"+str(participant)] = create_dataframe(matrix)

            #unattractive
            epochs = mne.EpochsArray(unattractive, info)
            con = mne_connectivity.spectral_connectivity_epochs(epochs[min_epochs:max_epochs], method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
                faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

            baseline = mne_connectivity.spectral_connectivity_epochs(epochs[0:86], method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
                faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

            matrix = con.get_data(output='dense')[:, :, 0]

            baseline = baseline.get_data(output='dense')[:, :, 0]

            matrix = matrix/baseline
            #store results
            results["unattractive"]["p"+str(participant)] = create_dataframe(matrix)

    return results

#set condition to either: "attractive" or "unattractive"
def overall(results,condition):
    df_new = (
        # combine dataframes into a single dataframe
        pd.concat(results[condition].values())
        # replace 0 values with nan to exclude them from mean calculation
        #.replace(0, np.nan)
        #.reset_index()
        # group by the row within the original dataframe
        .groupby("Index")
        # calculate the mean
        .mean()
    )
    df = df_new.reindex(df_new.columns)

    def symmetrize(data):
        data = data.fillna(-99)
        for row in data.index:
            for col in data.columns:
                if data.loc[row, col] >= 0:
                    data.loc[col, row] = data.loc[row, col]
        data = data.replace(-99, np.nan)
        # data = data.where(np.triu(np.ones(data.shape)).astype(np.bool), other=np.nan)

        # keep only left diagonal
        for row in range(len(data)):
            for col in range(len(data.iloc[row])):
                if col > row:
                    data.iloc[row, col] = -99
        data = data.replace(-99, np.nan)

        return data

    return symmetrize(df)

#CHECK WHICH PARTICIPANT HAS HIGHER NUMBER OF ELECTRODES

# Ugly hack due to acquisition problem when specifying the channel types
#layout = mne.layouts.read_layout('Vectorview-mag.lout')
layout = mne.channels.make_standard_montage('biosemi64')
layout_ = mne.channels.read_layout('biosemi')

coord = {}

for i in range(len(layout_.names)):
    chanel = layout_.names[i]
    coord[chanel] = layout_.pos[i][0:2]

#change coord names accordingly to data
coord["PO6"] = coord["P6"]
coord["PO5"] = coord["P5"]

def plot_matrix(con,method):
    import numpy as np
    import matplotlib.pyplot as plt
    r_con = con + con.T - np.diag(np.diag(con)) #I reflect the matrix
    plt.imshow(r_con);
    clb = plt.colorbar()
    clb.ax.set_title(method)
    plt.xlabel('Channels')
    plt.show()
    return

class Graph:
  def __init__(self, data):
    self.data = data
    self.columns = data.columns

  def con_nodes(self):
    lista = []
    for i in range(len(self.columns)):
        temp = []
        for j in range(i+1,len(self.columns)):
            temp.append([self.columns[i],self.columns[j]])
        lista.append(temp)
    import itertools
    con_nodes = list(itertools.chain(*lista))
    return con_nodes

  def weights(self):
    weights = []
    for node in self.con_nodes():
        weights.append(self.data.loc[node[1], node[0]])
    return weights

  def graph(self):
    import networkx as nx

    G = nx.Graph()

    for x in self.columns:
        G.add_node(x,pos=(coord[x][0],coord[x][1]))

    pos=nx.get_node_attributes(G,'pos')
    con_nodes_new = np.array(self.con_nodes())


    for x in range(0,len(self.con_nodes())):
        G.add_edge(self.con_nodes()[x][0],self.con_nodes()[x][1], weight=self.weights()[x],alpha=self.weights()[x])

    labels = {}
    for x in range (0,len(self.columns)):
        labels[x] =  self.columns[x]  #Needed to show the correct electrode label

    #Label positions needs to be changed in order to avoid the overlap with electrodes
    label_pos = {k:v for k,v in pos.items()}
    for i in pos:
        pos_x = pos[i][0]
        pos_y = pos[i][1]
        upd = {i:[pos_x+0.03,pos_y+0.01]}
        label_pos.update(upd)

    self.G = G
    self.pos = pos
    self.label_pos = label_pos

#small-world organization
def draw_smallworld(G,weights):
    sm_w = G
    pos_c=nx.circular_layout(sm_w)

    #Label positions needs to be changed in order to avoid the overlap with electrodes
    label_pos_c = {k:v for k,v in pos_c.items()}
    for i in pos_c:
        pos_x = pos_c[i][0]
        pos_y = pos_c[i][1]
        upd = {i:[pos_x+0.03,pos_y+0.01]}
        label_pos_c.update(upd)


    nx.draw(sm_w,pos_c,node_size=32,node_color='black',edge_color=weights,edge_cmap=plt.cm.Reds) #check how to add edge_vmin properly
    nx.draw_networkx_labels(sm_w,label_pos_c,font_size=7,font_color='grey')
    plt.show()
    return

#Plot the adjacency matrix
def plot_adjacency_matrix(A):
    import numpy as np
    import matplotlib.pyplot as plt
    A = A.todense()
    plt.imshow(A);
    plt.xlabel('Channels')
    plt.show()
    return

#####################
def plot_degree_distribution (G):
    import collections
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')
    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    # draw graph in inset
    plt.axes([0.4, 0.4, 0.5, 0.5])
    #Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
    Gcc = sorted(G.subgraph(c) for c in nx.connected_components(G))
    pos = nx.spring_layout(G)
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.show()
    return


def MST(results, condition):
    import networkx as nx
    dici = {}
    for key, data in results[condition].items():
        p1 = Graph(data)
        p1.graph()
        G = p1.G
        pos = p1.pos
        label_pos = p1.label_pos
        from networkx.algorithms import tree
        # We create new weights for the minimum spanning tree (1/weights)

        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

        new_weights = []

        for i in range(0, len(weights)):
            new_weights.append(1 / weights[i])  # Stronger connection will have a shortest distance

        new_weights = tuple(new_weights)  # convert the list to tuple
        # This part does not show the real MST, I have no idea why. I solved it using the maximum spanning tree
        # T=nx.minimum_spanning_tree(G,new_weights) #Minimum spanning tree
        T = nx.maximum_spanning_tree(G)  # This should do extactly the same thing as using w = 1/w as it maximise the distance

        from networkx import adjacency_matrix
        A = nx.adjacency_matrix(T)
        links = len(T.edges)

        edges = list(T.edges)

        # Metrics list
        # Degree, leaf number, betweenness centrality (BC), eccentricity,
        # diameter, hierarchy (Th), and degree correlation (R).
        from networkx import diameter, eccentricity, betweenness_centrality

        # is the ratio of the number of leaf nodes (nodes with degree 1) to the total number of nodes in the tree.
        new = {}
        for i in list(T.edges):
            for j in i:
                if j in new.keys():
                    new[j] += 1
                else:
                    new[j] = 1
        n_leafs = len([i for i in new.values() if i == 1])
        leaf_fraction = n_leafs / len(new.keys())

        # Max degree in the MST
        max_degree = 0
        degree = list(T.degree)
        for i in range(len(T.edges)):
            val = degree[i][1]
            if val > max_degree:
                max_degree = val

        # Diameter and eccentricity
        nx_diameter = diameter(T, e=None)
        nx_eccentricity = eccentricity(T, v=None, sp=None)
        nx_eccentricity_max = max(nx_eccentricity.values())

        # Betweenness centrality (BC) and BCmax
        nx_btw_centrality = betweenness_centrality(T, k=None, normalized=True, weight=None, endpoints=False, seed=None)
        # Applying round to the betweenness centrality to show only the first 3 values
        nx_btw_max = max(nx_btw_centrality.values())

        #nx_btw_max = 0
        #for i in range(len(T.edges)):
        #    val = nx_btw_centrality[i]
        #    if val > nx_btw_max:
        #        nx_btw_max = val

        closeness = nx.closeness_centrality(T)
        nx_closeness_max = max(closeness.values())

        # Tree hierarchy (Th=L/(2mBCmax))
        nx_th = n_leafs / (2 * links * nx_btw_max)  # TO BE CHECKED

        # Degree correlation
        from networkx import degree_pearson_correlation_coefficient
        nx_d = degree_pearson_correlation_coefficient(T, weight=None, nodes=None)

        dici[key] = {
            "n_nodes": len(T.nodes),
            "n_edges": len(T.edges),
            "n_leaf_nodes": n_leafs,
            "leaf_fraction": leaf_fraction,
            "max_degree": max_degree,
            "diameter": nx_diameter,
            "eccentricity": nx_eccentricity_max,
            "betweenness_centrality": nx_btw_max,
            "closeness_centrality": nx_closeness_max,
            "tree_hierarchy": nx_th,
            "degree_correlation": nx_d
        }

    return dici


def connectivity_plot(condition):

    central = [i for i in overall(condition).index if (ord(i[-1]) < ord("0") or ord(i[-1]) > ord("9"))]
    attractive_plot = overall(condition)
    attractive_plot = attractive_plot.drop(central, axis=0)
    attractive_plot = attractive_plot.drop(central, axis=1)

    # Connectivity Plot
    from mne_connectivity.viz import plot_connectivity_circle
    #left hemisphere (impar)
    left = [attractive_plot.index[i] for i in range(len(attractive_plot.index)) if int(attractive_plot.index[i][-1])%2 != 0]
    #right
    right = [attractive_plot.index[i] for i in range(len(attractive_plot.index)) if int(attractive_plot.index[i][-1])%2 == 0]
    #label_colors = [label.color for label in ]
    # First, we reorder the labels based on their location in the left hemi
    left_coord = {key:value for key,value in coord.items() if key in left}

    #order based on higher y
    def order(dici):
        keys = list(dici.keys())
        values = list(dici.values())
        ordered = []
        for i in range(len(keys)):
            for j in range(i+1,len(keys)):
                if values[j][1] > values[i][1]:
                    temp = keys[i]
                    keys[i] = keys[j]
                    keys[j] = temp

                    temp = values[i]
                    values[i] = values[j]
                    values[j] = temp
        return keys


    # For the right hemi
    right_coord = {key:value for key,value in coord.items() if key in right}

    # Save the plot order and create a circular layout
    node_order = list()
    #node_order.extend(order(left_coord)[::-1])  # reverse the order
    node_order.extend(order(left_coord))
    node_order.extend(order(right_coord)[::-1])

    from mne.viz import circular_layout
    node_angles = circular_layout(attractive_plot.index, node_order, start_pos=90,
                                  group_boundaries=[0, (len(attractive_plot.index) / 2)]) #check par ou impar

    import matplotlib.pyplot as plt
    # Plot the graph using node colors from the FreeSurfer parcellation. We only
    # show the 300 strongest connections.
    fig, ax1 = plt.subplots(figsize=(8, 8), facecolor='black',
                           subplot_kw=dict(polar=True))

    if condition=="attractive":
        title = "Attractive"
    else:
        title = "Unattractive"

    #attractive
    plot_connectivity_circle(np.array(attractive_plot), attractive_plot.index, n_lines=50,
                             node_angles=node_angles,
                             title=title+' Condition (PLI)\n (gamma)' ,ax=ax1)

    fig.tight_layout()
from matplotlib import pyplot as plt
import networkx as nx
from dendrogram_handler import *
from scipy.cluster.hierarchy import dendrogram
from collections import defaultdict
from matplotlib.lines import Line2D
import numpy as np

class graphHandler:
    def __init__(self, G, colormap, use_pydot = True):
        self.G = nx.convert_node_labels_to_integers(G)
        self.color_template = self.color_template = ['y', 'b', 'm', 'c', 'k'] 
        self.color_map = colormap
        if use_pydot: self.pos = nx.nx_pydot.graphviz_layout(self.G,prog='neato')
        else: self.pos = nx.nx_agraph.graphviz_layout(self.G,prog='neato')
    ####################
    ## Public methods ##
    ####################
    def getGraph(self): return self.G
    def showGraph(self,agent_colors = None,title = None, with_labels = False, legend = None):
        if agent_colors == None: agent_colors = self.color_map
        if title == None: title = "Network with " + str(len(self.G.nodes)) + ' agents'
        ax = plt.gca()
        ax.set_title(title)
        if with_labels:
            nx.draw(self.G,self.pos,node_color = agent_colors, with_labels = with_labels, node_size = 200, alpha=0.8)
        else:
            nx.draw(self.G,self.pos,node_color = agent_colors, node_size = 70, alpha=0.8)
        if legend is not None:
            plt.legend(handles = legend, loc = 'best')
    def showLouvainCommunities(self, title = None):
        agent_colors = self._getAgentColors_from_LouvainCommunities()
        if title == None: title = "Louvain communities for network with " + str(len(self.G.nodes)) + ' agents'
        ax = plt.gca()
        ax.set_title(title)
        nx.draw(self.G,self.pos,node_color = agent_colors, node_size = 70, alpha=0.8)
    def showDendrogram(self):
        myHandler = DendrogramHandler(self.G)
        Z = myHandler.getLinkMatrix()
        ZLabels = myHandler.getLinkMatrixLabels()
        #plt.figure(figureNumber);plt.clf()
        dendrogram(Z, labels=ZLabels)
        del myHandler
    def show_kCores(self):
        """ Visualize by k-cores. 
        Thanks to [Corralien's response on stackoverflow]
        (https://stackoverflow.com/questions/70297329/visualization-of-k-cores-using-networkx).
        """
        # build a dictionary of k-level with the list of nodes
        kcores = defaultdict(list)
        for n, k in nx.core_number(self.G).items():
            kcores[k].append(n)

        # compute position of each node with shell layout
        nlist = []
        for k in sorted(kcores.keys(),reverse=True):
            nlist.append(kcores[k])
        pos = nx.layout.shell_layout(self.G, nlist = nlist)
        colors = ['black','lightblue','yellow','magenta','olive', 'cyan', 'red']
        legend_elements = []

        # draw nodes, edges and labels
        for kcore in sorted(list(kcores.keys()),reverse = True):
            nodes = kcores[kcore]
            color = colors[kcore%len(colors)]
            nx.draw_networkx_nodes(self.G, pos, nodelist=nodes, node_color=color)
            label = f"kcore = {kcore}"
            legend_elements.append(Line2D([0], [0], marker='o', color=color, label=label,markerfacecolor=color, markersize=15))
        nx.draw_networkx_edges(self.G, pos, width=0.2)
        nx.draw_networkx_labels(self.G, pos)
        plt.title("K-core layout of network")
        plt.legend(handles = legend_elements, loc = 'best')

    def show_kCores_by_partition(self, colors, title = "K-core of Network"):
        """ Visualize by k-cores. 
        Thanks to [Corralien's response on stackoverflow]
        (https://stackoverflow.com/questions/70297329/visualization-of-k-cores-using-networkx).
        """
        # build a dictionary of k-level with the list of nodes
        kcores = defaultdict(list)
        for n, k in nx.core_number(self.G).items():
            kcores[k].append(n)

        # Shapes
        shapes = ["o", "v", "s", "*", "+", "d"]

        # compute position of each node with shell layout
        nlist = []
        for k in sorted(kcores.keys(),reverse=True):
            nlist.append(kcores[k])
        pos = nx.layout.shell_layout(self.G, nlist = nlist)
        legend_elements = []

        # draw nodes, edges and labels
        for kcore in sorted(kcores.keys(),reverse=True):
            nodes = kcores[kcore]
            shape = shapes[kcore%len(shapes)]
            
            #nx.draw_networkx_nodes(self.G, pos, nodelist=nodes, node_color=colors[nodes[0]], node_shape=shape, alpha = 0.5, node_size=90)
            for node in nodes:
                nx.draw_networkx_nodes(self.G, pos, nodelist=[node], node_color=colors[node], node_shape=shape, alpha = 0.5, node_size=90)
            label = f"kcore = {kcore}"
            legend_elements.append(Line2D([0], [0], marker=shape, color='k', markerfacecolor = 'w', label=label, markersize=10))
        
        nx.draw_networkx_edges(self.G, pos, width=0.1)
        #nx.draw_networkx_labels(self.G, pos)
        plt.title(title)
        plt.legend(handles = legend_elements, loc = 'best')

    def show_partitions(self, partition_list, title = "Network colored by partitions"):
        plt.figure()
        plt.axis('off')
        for i in range(len(partition_list)):
            nx.draw_networkx_nodes(partition_list[i],self.pos,node_color=self.color_template[i%len(self.color_template)], alpha = 0.8)
        for edge in self.G.edges:
            self._draw_edge_by_type(edge, partition_list)
        nx.draw_networkx_labels(self.G,self.pos)
        if len(partition_list) == 0:
            mod = 0
        else:
            mod = nx.algorithms.community.quality.modularity(self.G,partition_list)
        title = title + ": Modularity = " + str(np.round(mod,2))
        plt.title(title)
    
    #####################
    ## Private methods ##
    #####################
    def _getAgentColors_from_LouvainCommunities(self):
        """ Use the Louvain partition method to break the graph into communities """
        # Louvain method pip install python-louvain
        # see https://arxiv.org/pdf/0803.0476.pdf
        # see https://github.com/taynaud/python-louvain
        color_map = self.color_map
        set_of_partitions = nx.community.louvain_communities(self.G)
        print(f"The Louvain algorithm found {len(set_of_partitions)} partitions.")
        partition_number = 0
        for partition in set_of_partitions:
            print(f"Partition {partition_number} is {partition}")
            for node in partition:
                color_map[node] = self.color_template[partition_number%len(self.color_template)]
            partition_number += 1
        return color_map
    def _draw_edge_by_type(self, edge, partition):
        edge_style = 'dashed'
        for part in partition:
            if edge[0] in part and edge[1] in part:
                edge_style = 'solid'
                break
        nx.draw_networkx_edges(self.G, self.pos, edgelist=[edge], style = edge_style)
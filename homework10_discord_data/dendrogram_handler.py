""" Dendrogram

Mike Goodrich
Brigham Young University
February 2022

Code adapted from
https://stackoverflow.com/questions/59821151/plot-the-dendrogram-of-communities-found-by-networkx-girvan-newman-algorithm 
Thank you Giora Simchoni
"""
import networkx as nx
from itertools import chain, combinations

class DendrogramHandler:
    def __init__(self,G):
        self.G = G
        self.communities = self._getGirvanNewmanCommunities()
        self.node_id_to_children, self.node_labels = self._getNode_id_to_children_dict()
        self.subset_rank_dict = self._getSubset_rank_dict()
        self.Z, self.leaves = self._getLinkMatrix()

    """ Public methods """
    def getLinkMatrix(self): return self.Z
    #def getAll(self): return self.Z, self.leaves, self.node_labels
    def getLinkMatrixLabels(self): return [self.node_labels[node_id] for node_id in self.leaves]
    #def getLeaves(self): return self.leaves
    #def getNodeLabels(self): return self.node_labels
    

    """ Private methods """
    def _getGirvanNewmanCommunities(self):
        # Get communities using edge betweenness algorithm from Girvan and Newman
        communities = list(nx.algorithms.community.centrality.girvan_newman(self.G))
        #tuple(sorted(c) for c in communities) # uncomment if you want to print out set of ocmmunities
        return communities
    def _getNode_id_to_children_dict(self):
        node_id = 0
        init_node2community_dict = {node_id: self.communities[0][0].union(self.communities[0][1])}
        for comm in self.communities:
            for subset in list(comm):
                if subset not in init_node2community_dict.values():
                    node_id += 1
                    init_node2community_dict[node_id] = subset

        # turning this dictionary to the desired format in @mdml's answer
        node_id_to_children = {e: [] for e in init_node2community_dict.keys()}
        for node_id1, node_id2 in combinations(init_node2community_dict.keys(), 2):
            for node_id_parent, group in init_node2community_dict.items():
                if len(init_node2community_dict[node_id1].intersection(init_node2community_dict[node_id2])) == 0 and group == init_node2community_dict[node_id1].union(init_node2community_dict[node_id2]):
                    node_id_to_children[node_id_parent].append(node_id1)
                    node_id_to_children[node_id_parent].append(node_id2)
        
        # also recording node_labels dict for the correct label for dendrogram leaves
        node_labels = dict()
        for node_id, group in init_node2community_dict.items():
            if len(group) == 1:
                node_labels[node_id] = list(group)[0]
            else:
                node_labels[node_id] = ''
        return node_id_to_children, node_labels

    def _getSubset_rank_dict(self):
        # also needing a subset to rank dict to later know within all k-length merges which came first
        subset_rank_dict = dict()
        rank = 0
        for e in self.communities[::-1]:
            for p in list(e):
                if tuple(p) not in subset_rank_dict:
                    subset_rank_dict[tuple(sorted(p))] = rank
                    rank += 1
        subset_rank_dict[tuple(sorted(chain.from_iterable(self.communities[-1])))] = rank
        return subset_rank_dict 
    def _getLinkMatrix(self):
        # finally using @mdml's magic, slightly modified:
        G           = nx.DiGraph(self.node_id_to_children)
        nodes       = G.nodes()
        leaves      = set( n for n in nodes if G.out_degree(n) == 0 )
        inner_nodes = [ n for n in nodes if G.out_degree(n) > 0 ]

        # Compute the size of each subtree
        subtree = dict( (n, [n]) for n in leaves )
        for u in inner_nodes:
            children = set()
            node_list = list(self.node_id_to_children[u])
            while len(node_list) > 0:
                v = node_list.pop(0)
                children.add( v )
                node_list += self.node_id_to_children[v]
            subtree[u] = sorted(children & leaves)

        inner_nodes.sort(key=lambda n: len(subtree[n])) # <-- order inner nodes ascending by subtree size, root is last

        # Construct the linkage matrix
        leaves = sorted(leaves)
        index  = dict( (tuple([n]), i) for i, n in enumerate(leaves) )
        Z = []
        k = len(leaves)
        for i, n in enumerate(inner_nodes):
            children = self.node_id_to_children[n]
            x = children[0]
            for y in children[1:]:
                z = tuple(sorted(subtree[x] + subtree[y]))
                i, j = index[tuple(sorted(subtree[x]))], index[tuple(sorted(subtree[y]))]
                Z.append([i, j, self._get_merge_height(subtree[n]), len(z)]) # <-- float is required by the dendrogram function
                index[z] = k
                subtree[z] = list(z)
                x = z
                k += 1
        return Z, leaves
    def _get_merge_height(self,sub):
        # Giora Simchoni's function to get a merge height so that it is unique (probably not that efficient)
        sub_tuple = tuple(sorted([self.node_labels[i] for i in sub]))
        n = len(sub_tuple)
        other_same_len_merges = {k: v for k, v in self.subset_rank_dict.items() if len(k) == n}
        min_rank, max_rank = min(other_same_len_merges.values()), max(other_same_len_merges.values())
        range = (max_rank-min_rank) if max_rank > min_rank else 1
        return float(len(sub)) + 0.8 * (self.subset_rank_dict[sub_tuple] - min_rank) / range

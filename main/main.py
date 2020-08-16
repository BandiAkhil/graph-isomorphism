from itertools import combinations
from utils.graph import *
from utils.graph_io import load_graph


class PartitionSet:

    def __init__(self, partitions: List[List[Vertex]]=None, with_colors=False):
        self._partitions: List[List[Vertex]] = partitions if partitions is not None else list()
        if with_colors:
            self.assign_vertex_colors()

    @staticmethod
    def single_partition(vertexes: List[Vertex], assign_colors=False) -> "PartitionSet":
        return PartitionSet([vertexes], assign_colors)

    def balanced(self):
        return all([len(partition) >= 2 for partition in self._partitions])

    def bijective(self):
        return all([len(partition) == 2 for partition in self._partitions])

    def clone(self) -> "PartitionSet":
        clone = [[*partition] for partition in self._partitions]
        return PartitionSet(clone)

    def assign_vertex_colors(self):
        index = 0
        for p in self._partitions:
            for v in p:
                v.color_num = index
            index += 1

    def partitions(self):
        return self._partitions

    def pop(self, index=None) -> List[Vertex]:
        return self._partitions.pop(index if index else len(self._partitions) - 1)

    # Finds a big enough partition, removes and returns it
    def pop_first(self) -> (int, List[Vertex]):
        for i in range(len(self._partitions)):
            if len(self._partitions[i]) > 2:
                return i, self._partitions.pop(i)

    def __len__(self):
        return len(self._partitions)

    def __getitem__(self, item):
        return self._partitions.__getitem__(item)

    def add_partition(self, newPart: List[Vertex]):
        self._partitions.append(newPart)

    def set_partition(self, index: int, part: List[Vertex]):
        self._partitions.insert(index, part)


def count(p: PartitionSet, aut):
    if p.bijective():
        return 1
    elif not p.balanced():
        return 0
    else:
        # Calculate the color classes
        partition_n, color_class = p.pop_first()
        x = color_class[0]
        num = 0
        y_index = -1
        for y in color_class:
            y_index += 1
            if y.original_graph == x.original_graph:
                continue  # Origin from same graph: not a wise pick...
            else:
                previous_xys = []
                for i in range(len(color_class)):
                    if i != y_index and i != 0:
                        previous_xys.append(color_class[i])
                p.add_partition(previous_xys)
                p.add_partition([x, y])
                p.assign_vertex_colors()
                if num == 0 or aut:
                    num = num + count(partition_refinement(p.clone()), aut)
                del p.partitions()[len(p.partitions())-2:]

    p.set_partition(partition_n, color_class)
    return num


def partition_refinement(p: PartitionSet):

    # Assign colors to vertices in partitions
    # p.assign_vertex_colors()

    partitions_length = len(p)
    available = partitions_length
    # We make a queue of colors that
    # is initially filled with all colors we need to consider
    # also, we have inQueue which indicates if it is in the queue

    in_queue = list()
    refinement_queue = []
    for i in range(partitions_length):
        in_queue.append(True)
        refinement_queue.append(i)

    # we terminate when the queue is empty (stable partitions are obtained)
    while len(refinement_queue) > 0:

        # We get the next element in the queue we can use
        current_color = refinement_queue.pop()
        in_queue[current_color] = False

        # loop through each state in p

        for i in range(len(p)):
            l = dict()  # will contain all colors i such that Ci contains q' in neighbors
            a = dict()  # will contain the amount of such states in Ci

            # we loop through all v in our colorclass
            # and also through all v' in the neighboring vertices
            # we count the number of states in the class where:
            # class contains a q' with the color of the current partition
            for v in p[i]:
                a[v] = 0
                for n in v.neighbours:
                    if n.color_num == current_color:
                        a[v] = a[v] + 1
                if a[v] not in l.keys():
                    l[a[v]] = list()
                l[a[v]].append(v)

            # Now, we loop through the list we composed above
            # For each color in the list, split up if counted<size of part
            # if we split up; choose new color and update queue
            first = True
            for k in l.keys():
                if first:  # we need to keep the first partition so it won't break
                    first = False
                    continue
                if k < len(p[i]) or in_queue[i]:
                    new = available
                    available += 1
                    # Now, we need to make sure that the new color is added
                    #  and that the Queue is updated
                    # We remove from old partition and update the new one
                    my_list = []
                    for n in l[k]:
                        n.color = new
                        my_list.append(n.n)
                    p.add_partition(l[k])
                    duplicate = p[i].copy()
                    p[i].clear()
                    for v in duplicate:
                        if v.n not in my_list:
                            p[i].append(v)
                    # and add the new color now
                    in_queue.append(True)  # it isn't in the queue now
                    refinement_queue.append(new)
                    # Now, we need to make sure that the new color is added
                    #  and that the Queue is updated
    return p


def count_isomorphisms(g1: Graph, g2: Graph, aut: bool):
    g = g1 + g2
    partition = PartitionSet.single_partition(g.vertices, True)
    return count(partition, aut)


#makes another graph the same as the original for automorphism
def count_automorphisms(list_of_graphs):
    for graph in list_of_graphs:
        new_g = Graph(False)
        for v in graph.vertices:
            v1 = Vertex(v.graph, v.n, v.label, v.color_num)
            new_g.add_vertex(v1)
        for edge in graph.edges:
            tail = edge.tail
            head = edge.head
            new_t = None
            new_h = None
            for v1 in new_g.vertices:
                if tail.label == v1.label:
                    new_t = v1
                elif head.label == v1.label:
                    new_h = v1
            e = Edge(new_t, new_h)
            new_g.add_edge(e)
        auto = count_isomorphisms(graph, new_g, True)
        print(graph.name, ":", "Automorphisims:", auto)


#Check for isomorphisms from a list of graphs
def check_isomorphisms(list_of_graphs):
    comb = combinations(list_of_graphs, 2)
    for graph1, graph2 in comb:
        if graph1 == graph2:
            continue
        if count_isomorphisms(graph1, graph2, False) > 0:
            print("Isomporphic:", (graph1.name, graph2.name))
            global found
            found = False


def main(problem):
    with open('../basic/basicGI2.grl', 'r') as f:
        list_of_graphs = load_graph(f, Graph, True)

        if problem == "auto":
            count_automorphisms(list_of_graphs[0])
        elif problem == "iso":
            check_isomorphisms(list_of_graphs[0])
        elif problem == "iso auto":
            check_isomorphisms(list_of_graphs[0])
            count_automorphisms(list_of_graphs[0])
        else:
            print("Invalid")


# Use this to specify if it's a GI problem("iso") or #Aut problem("auto") or both("iso auto")
main("iso")
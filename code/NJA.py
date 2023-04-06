# import skimage as img
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
import skimage.io as io
from skimage.morphology import skeletonize, medial_axis
# from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.measure import regionprops
# from skimage.feature import corner_harris, corner_peaks
# from scipy.spatial import KDTree
# from skimage.segmentation import active_contour
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import multiprocessing
# from os import path

# TODO: Add docstrings for all methods and funcs


# TODO: Document global dicts
dirs = np.array(["NW", "N", "NE", "W", None, "E", "SW", "S", "SE"])
"""numpy.array: Lookup for possible directions based upon index in a flattened 3x3 matrix relative to (1,1) 
"""

dir_deltas = np.array([(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)])
revdirs = list(reversed(dirs))
dir_lookup = {x: [i] for i, x in enumerate(dirs) if x is not None}
for x in dir_lookup:
    # print(x)
    dir_lookup[x] += [revdirs[dir_lookup[x][0]], 8 - dir_lookup[x][0], dir_deltas[dir_lookup[x][0]]]


def get_3x3(image, y, x, flatten=False):
    """Get surrounding pixels of a location from a binarised image.
    
    Args:
        image (numpy.array): the image to analyse.
        y (int): y coordinate of the pixel.
        x (int): x coordinate of the pixel.
        flatten : Return the submatrix in a flat (1D) form rather than its usual 3x3. Defaults to False.
    
    Returns:
        numpy.array: 3x3 submatrix of the surrounding pixels.
    """
    edgedict = {
        "top": [[0, 2, 1, 2], [1, 0]],  # Get a 3x2 matrix, then overlay that over point [r=1, c=0] on a 3x3 zeros mat
        "bottom": [[1, 1, 1, 2], [0, 0]],
        "left": [[1, 2, 0, 2], [0, 1]],
        "right": [[1, 2, 1, 1], [0, 0]],
        "topleft": [[0, 2, 0, 2], [1, 1]],
        "topright": [[0, 2, 1, 1], [1, 0]],
        "bottomleft": [[1, 1, 0, 2], [0, 1]],
        "bottomright": [[1, 1, 1, 1], [0, 0]]
    }

    # Detect if we are on an edge and construct a string that describes it
    yedge = ""
    xedge = ""
    if y == 0:
        yedge = "top"
    elif y == image.shape[0] - 1:
        yedge = "bottom"

    if x == 0:
        xedge = "left"
    elif x == image.shape[1] - 1:
        xedge = "right"

    edgepos = yedge + xedge

    if edgepos:
        # Look up ylo, yhi, xlo, xhi, and coords in dict of lists
        deltas, coords = edgedict[edgepos]
        # Deepcopy submat to make sure when we do temporary editing later, we don't modify the original image
        # This could be a bit slow so might be worth finding another way to do it
        submat = deepcopy(image[y - deltas[0]:y + deltas[1], x - deltas[2]:x + deltas[3]])
        finalmat = np.zeros((3, 3), dtype=bool)
        finalmat[coords[0]:coords[0] + submat.shape[0], coords[1]:coords[1] + submat.shape[1]] = submat
    else:
        finalmat = np.copy(image[y - 1:y + 2, x - 1:x + 2])

    if flatten:
        return finalmat.flatten()
    else:
        return finalmat


def detect_junc(image, y: int, x: int, flatten=False) -> tuple:
    """Detect number of junctions from a pixel based on the surrounding pixels in an image.
    
    Args:
        image (numpy.array): the image to analyse.
        y (int): y coordinate of the pixel.
        x (int): x coordinate of the pixel.
        flatten : Return the submatrix in a flat (1D) form rather than its usual 3x3. Defaults to False.
    
    Returns:
        tuple: (Submatrix, Num of junctions).
    """
    submat = get_3x3(image, y, x, flatten)
    submat[1][1] = False
    return submat, np.count_nonzero(submat.flatten())


def fmt_sm(submat, off="⬛", on="🟧", here="🟥"):
    """Format a 3x3 boolean submatrix using UTF-8 characters.
    
    For example:\n
    ⬛⬛⬛\n
    🟧🟥⬛\n
    ⬛🟧🟧
    
    Args:
        submat (ndarray): the image to analyse.
        off (str): Character to represent False pixels.
        on (str): Character to represent False pixels.
        here (str) : Character to represent the current location.
    
    Returns:
        str: Formatted string representation.
    """
    formatted = np.repeat([off], 9).reshape([3, 3])
    formatted[submat] = on
    formatted[1, 1] = here
    formatted = "\n".join(["".join(x) for x in formatted])
    return formatted


def trace_path(startpoint, direction, skel, print_journey=False, return_journey=False):
    """Trace the path between two junctions over a skeletonised image.

    Traces the path from a starting point on a skeletonised image along white pixels starting in a given direction until
    another junction is reached.

    Args:
        startpoint (list): A list specifying the startpoint and its context [position, surround, juncs, None]
            (probably needs a refactor at some point).
        direction (str): The direction to leave the starting pixel (captialised cardinal or ordinal directions,
            i.e. "N" or "SW").
        skel (numpy.array): Skeletonised image to traverse.
        print_journey (bool): Whether to print the journey taken to the console. Defaults to False.
        return_journey (bool): Whether to return the journey taken. Defaults to False.

    Returns:
        tuple: (endpoint, path length, [optionally, an array of the locations traversed during the trace]).
        """
    if print_journey:
        print(fmt_sm(startpoint[1]))
    # Init tracking set of places we've been
    curr_loc, curr_mat, curr_juncs, _ = deepcopy(startpoint)
    previously_visited = set()
    previously_visited.add(str(curr_loc))
    path_px = [deepcopy(curr_loc)]
    pathlength = 0
    while True:
        #         print(previously_visited)
        # Traverse to next loc
        dirdetails = dir_lookup[direction]
        curr_loc += dirdetails[3]

        # If previously visited, break and return None
        if str(curr_loc) in previously_visited:
            raise ValueError(f"Path looped at {curr_loc} going {direction}.")

        # We've moved successfully so now note it down
        if direction in ["N", "S", "E", "W"]:
            pathlength += 1
        else:
            # If we go diagonally, this distance is obviously longer
            pathlength += np.sqrt(2)
        previously_visited.add(str(curr_loc))
        path_px.append(deepcopy(curr_loc))
        # Test to see if junc
        curr_mat, curr_juncs = detect_junc(skel, *curr_loc)
        if print_journey:
            print(f"\n{direction} to {curr_loc} with {curr_juncs} juncs.\n")
            print(fmt_sm(curr_mat))
        if curr_juncs != 2:
            # If so, return location and length of path
            break
        # If not
        # Find next loc (by removing inverse of direction from submat and finding the only True location)
        invdirdetails = dir_lookup[dirdetails[1]]
        # We have do renormalise the offsets from dirdetails to (1,1), giving us the position in the submat to address
        # Remove previous position
        curr_mat[invdirdetails[3][0] + 1, invdirdetails[3][1] + 1] = False

        #         print(fmt_sm(curr_mat))

        # Detect only True in the submat (hopefully)
        poss_exits = dirs[np.flatnonzero(curr_mat)]
        if len(poss_exits) > 1:
            raise ValueError(f"Multiple exit points for {curr_loc}: ({poss_exits}) but not detected as a junc!")

        direction = poss_exits[0]

    # Finally
    if print_journey:
        print("Done!\n\n Final Path")
        print(np.asarray(path_px))
    if return_journey:
        return curr_loc, pathlength, np.asarray(path_px)
    else:
        return curr_loc, pathlength


def breadth_first(uid, candidates, net, threshold=2, timeout=100):
    """Find all nodes connected to the candidate by journeys over edges of a length <= threshold.
    
    Perform an efficient breadth-first search to find all nodes in a :class:`NJA.NJANet` network connected to a
    candidate :class:`NJA.NJANode` by taking a path consisting of only edges shorter or equal to the threshold distance.
    
    Args:
        uid (tuple): The uid of the candidate node.
        candidates (set): A set of candidate node uids (usually ones that have an edge of length <= threshold
            connected to them).
        net (:class:`NJA.NJANet`): The full network object.
        threshold (int): The threshold distance for each jump of the traversal. Defaults to 2.
        timeout (int) : The number of jumps to take before abandoning a search. Defaults to 100.
    
    Returns:
        set: The set of all node uids connected with the candidate node.
    """
    finalset = {uid}
    prevset = {uid}
    counter = 0
    while True:
        counter += 1
        if counter > timeout:
            break
        newset = set()
        # For each node in previous set
        for nodeuid in prevset:
            # Get node
            workingnode = net.nodes[nodeuid]
            # For each edge on this node
            for edge in workingnode.connected_edges.values():
                # Add all connected nodes to newset if the edge pixel length is at or below set threshold
                if edge.pixel_length <= threshold:
                    newset = newset.union(edge.connected_node_uids)
        # Filter against candidate set (eliminate non-candidates)
        newset = candidates.intersection(newset)

        # Filter against previous set (eliminate any that are in previous)
        newset = newset.difference(prevset)
        # Filter against final set (eliminate any that are in final)
        newset = newset.difference(finalset)

        # Shift all the sets up one level
        # print("preshift", counter, len(newset), len(prevset), len(finalset))
        finalset = finalset.union(prevset)
        prevset = newset
        # print("postshift", counter, len(newset), len(prevset), len(finalset))

        if len(newset) < 1:
            # If newset has nothing in it, we have nothing left to do
            break

    return finalset


class NJANode:
    """A component node of an NJANet.

    Stores all information about a node in a nice, easy to handle manner. Internally within other structures, NJANodes
    are often stored in dictionaries in the form:

    `{uid: NJANode, uid2: NJANode...}`

    Note:
        This allows one to quickly find nodes in a dict by indexing using their uids as the key.

    Attributes:
        position: A tuple(?) of ints indicating the position of the node.
        surround: A 3x3 numpy array indicating the pixel context of the node.
        juncs: An integer count of the number of junctions the node has
        dirs: A list of strings indicating the directions that junctions of the node emit
        uid: A tuple of the position, acting as a unique identifier
        connected_edges: A dictionary of all connected edges (filled using :meth:`NJA.NJAEdge.link_nodes_to_edges`)
    """
    def __init__(self, pos, surround=None, juncs=None, dirs=None, uid=None):
        self.position = pos
        self.surround = surround
        self.juncs = juncs
        self.dirs = dirs
        if uid is not None:
            self.uid = uid
        else:
            self.uid = tuple(self.position)
        self.connected_edges = {}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.position}, {self.uid}, {self.surround}, {self.juncs}, {self.dirs})"

    def __str__(self):
        return f"UID:{self.uid}\n{self.__class__.__name__} @ {self.position}\nSurround:\n{self.format_surround(self.surround)}\nJunctions:{self.juncs}\nDirections:{self.dirs})"

    @property
    def flipped_position(self):
        """The position of the node in y,x coordinates (for plotting with matplotlib).
        """
        # for plotting
        return [self.position[1], self.position[0]]

    @property
    def connected_edge_uids(self):
        """The uids of all connected edges as a tuple.
        """
        return tuple(self.connected_edges.keys())

    def find_directions(self, dirdict=None):
        """Find the directions of exits from the node based upon the surround

        Args:
            dirdict: An array of directions in the form of :data:`NJA.dirs`
        """
        # Requires dirs to be initialised
        if dirdict is None:
            dirdict = dirs
        self.dirs = list(dirdict[np.flatnonzero(self.surround)])

    def reset_connected(self):
        """Reinitialise :attr:`NJA.NJANode.connected_edges` to a blank dictionary
        """
        self.connected_edges = {}

    def format_surround(self, off="⬛", on="🟧", here="🟥"):
        """A convenience wrapper around :func:`NJA.fmt_sm`.
        """
        return fmt_sm(self.surround, off, on, here)


class NJAEdge:
    """A component node of an NJANet.

        Stores all information about an edg. Internally within other structures, NJAEdges are often stored in
        dictionaries in the form:

        `{(startuid enduid): NJAEdge, (startuid2 enduid2): NJAEdge,...}`

        These uids are thus formatted `(x1 y1 x2 y2)`

        Note:
            This allows one to quickly find edges using their node ids.

        Attributes:
            start:
            end:
            uid:
            pixel_length:
            direct_length:
            length_difference:
            path:
    """
    def __init__(self, start, end, uid=None, pixel_length=None, direct_length=None, path=None):
        self.start = start
        self.end = end
        if uid is not None:
            self.uid = uid
        else:
            self.uid = start.uid + end.uid
        self.pixel_length = pixel_length
        if direct_length is not None:
            self.direct_length = direct_length
        else:
            self.calc_direct_length()
        self.length_difference = None
        self.path = path
        try:
            self.calc_length_difference()
        except TypeError:
            pass

    def __repr__(self):
        return f"{__class__.__name__}({repr(self.start)}, {repr(self.end)},{self.uid}, {self.pixel_length}, {self.direct_length}, {self.path})"

    def __str__(self):
        return f"UID:{self.uid}\n{self.__class__.__name__}\n{repr(self.start.uid)}->{repr(self.end.uid)}\nPixel Length: {self.pixel_length}\nDirect Length: {self.direct_length}\nLength Difference: {self.length_difference}"

    @property
    def plotting_repr(self):
        return [self.start.flipped_position, self.end.flipped_position]

    @property
    def connected_node_uids(self):
        return self.start.uid, self.end.uid

    def calc_direct_length(self):
        # self.direct_length = np.sqrt((self.start.position[0] - self.end.position[0])**2 + (self.start.position[1] - self.end.position[1])**2)
        # Same as above as the L2 norm (ord=2) is the same as the euclidian distance.
        self.direct_length = np.linalg.norm(self.start.position - self.end.position, ord=2, axis=0)
        return self

    def calc_length_difference(self):
        self.calc_direct_length()
        self.length_difference = abs(self.pixel_length - self.direct_length)
        return self

    def regenerate_uid(self, returnuid=False):
        self.uid = self.start.uid + self.end.uid
        if returnuid:
            return self.uid

    def format_journey(self):
        if self.path is None:
            print(f"No path trace detected in edge {self.uid}")
            return None
        path = self.path
        ymin = path[:, 0].min()
        ymax = path[:, 0].max()
        xmin = path[:, 1].min()
        xmax = path[:, 1].max()
        dims = [(ymax - ymin) + 1, (xmax - xmin) + 1]
        newpath = deepcopy(path) - np.array([ymin, xmin])
        output = np.repeat("⬛", dims[0] * dims[1]).reshape(dims)
        for x in newpath:
            output[x[0], x[1]] = "🟧"
        output[newpath[0][0], newpath[0][1]] = "🟩"
        output[newpath[-1][0], newpath[-1][1]] = "🟥"
        formatted = "\n".join(["".join(x) for x in output])
        return formatted

    def print_journey(self):
        print(self.format_journey())


class NJANet:
    def __init__(self, image):
        self.image = image
        self.blurred = None
        self.skel = None
        self.centroid = None
        self.contour_centroids = None
        self.basenode = None
        self.nodes = {}
        self.edges = {}

    def __str__(self):
        return f"{self.__class__.__name__}\nNodes: {len(self.nodes)}\nEdges: {len(self.edges)}"

    def skeletonize(self):
        self.skel = skeletonize(self.image)
        return self
    
    def generate_blurred(self, sigma=30):
        self.blurred = gaussian(self.image, sigma, preserve_range=True)
        return self
    
    def find_centroid(self):
        region = regionprops(self.image.astype(np.uint8))[0]
        self.centroid = region.centroid
        return self
        
    @staticmethod
    def find_region_asymmetry(blurred, threshold):
        thresholded = blurred > threshold
        region = regionprops(thresholded.astype(np.uint8))[0]
        return region.centroid

    def calculate_contour_centroids(self, contours=128):
        # Fulfil requirements
        if self.blurred is None:
            self.generate_blurred()
        if self.centroid is None:
            self.find_centroid()
            
        threshes = np.linspace(0.0, np.amax(self.blurred), contours+1)[:-1]
        self.contour_centroids = np.array([self.find_region_asymmetry(self.blurred, x) for x in threshes])
        return self

    def find_nodes(self):
        if self.skel is None:
            self.skeletonize()
        whitepx = np.transpose(self.skel.nonzero())

        for px in tqdm(whitepx, bar_format="Finding Nodes: {l_bar}{bar}{r_bar}"):
            x = detect_junc(self.skel, *px, flatten=False)
            if x[1] > 0 and x[1] != 2:
                node = NJANode(px, *x)
                self.nodes[node.uid] = node
        return self

    def find_directions(self):
        for x in tqdm(self.nodes.values(), bar_format=" Finding Dirs: {l_bar}{bar}{r_bar}"):
            x.find_directions()
        return self
    
    @property
    def onenodes(self):
        return len([x for x in self.nodes.values() if x.juncs == 1])
    
    def find_basenode(self, target=None):
        if self.contour_centroids is None:
            self.calculate_contour_centroids()
        if target is None:
            target = self.contour_centroids[-1]

        tnodes = np.array(list(self.nodes.keys()))
        self.basenode = self.nodes[tuple(tnodes[np.argmin((np.linalg.norm(tnodes - target, ord=2, axis=1)))])]
        return self
    
    def basenode_to_roottip_distances(self):
        return np.linalg.norm(np.array([x.position for x in self.nodes.values() if x.juncs == 1]) - self.basenode.position, ord=2, axis=1)

    def trace_paths(self):
        # This is slightly less consistent than doing it all in one LC, but way easier to debug
        # outlist = []
        traceout = None
        for x in tqdm(self.nodes.values(), bar_format="Tracing Paths: {l_bar}{bar}{r_bar}"):
            # Loop through output directions
            for y in x.dirs:
                try:
                    traceout = trace_path([x.position, x.surround, x.juncs, None], y, self.skel, return_journey=True)
                    yuid = tuple(traceout[0])
                    self.edges[x.uid + yuid] = NJAEdge(x, self.nodes[yuid], uid=x.uid + yuid, pixel_length=traceout[1],
                                                       direct_length=None, path=traceout[2])
                except ValueError as e:
                    # Honestly cycles don't matter
                    #                     print(str(e))
                    pass
                except KeyError:
                    print(traceout)
                    print(x)
                    print(y)
                    raise
        return self

    @staticmethod
    def _multi_trace_path(*args):
        # print(args[0])
        try:
            result = trace_path(*args[1:], print_journey=False, return_journey=True)
            return args[0], result
        except ValueError:
            return None

    def trace_paths_multicore(self):
        # This is slightly less consistent than doing it all in one LC, but way easier to debug
        cpus = multiprocessing.cpu_count() - 1

        var_list = [[x.uid, [x.position, x.surround, x.juncs, None], y, self.skel] for x in self.nodes.values() for y in x.dirs]
        chunksize = int(np.ceil(len(var_list) / cpus))
        # print(chunksize)
        chunksize = None

        outlist = []
        with multiprocessing.Pool(cpus) as p:
            try:
                for result in p.starmap(self._multi_trace_path, var_list, chunksize=chunksize):
                    # print("Completed %s", result[0])
                    outlist.append(result)
            except KeyboardInterrupt:
                # logger.warning("Attempting to exit multicore run...")
                p.terminate()

        for x in outlist:
            xuid, traceout = x
            yuid = tuple(traceout[0])
            self.edges[xuid + yuid] = NJAEdge(self.nodes[xuid], self.nodes[yuid], uid=xuid + yuid,
                                              pixel_length=traceout[1], direct_length=None, path=traceout[2])
        return self

    def clean_edges(self, regenuids=False, purge=False):
        cleandict = dict()
        for x in tqdm(self.edges, bar_format="Cleaning Edgelist: {l_bar}{bar}{r_bar}"):
            oldx = x
            if regenuids:
                # Regenerate uids for edges in edgelist and force Node.connected_edges to repopulate
                x = self.edges[oldx].regenerate_uid(returnuid=True)
                purge = True
            if tuple([x[2], x[3], x[0], x[1]]) in cleandict:
                # Much more common
                pass
            elif x in cleandict:
                pass
            else:
                cleandict[x] = self.edges[oldx]
        self.edges = cleandict
        self.link_nodes_to_edges(purge=purge)
        return self

    def link_nodes_to_edges(self, purge=False):
        if purge:
            for x in self.nodes.values():
                x.reset_connected()
        for edge in self.edges.values():
            edge.start.connected_edges[edge.uid] = edge
            edge.end.connected_edges[edge.uid] = edge
        return self

    def remove_edges_by_uid(self, uids):
        if len(uids) == 4:
            if all([isinstance(x, int) for x in uids]):
                uids = [uids]
        for uid in uids:
            del self.edges[uid]
        return self

    def remove_nodes_by_uid(self, uids):
        if len(uids) == 2:
            if all([isinstance(x, int) for x in uids]):
                uids = [uids]
        for uid in uids:
            del self.nodes[uid]
        return self

    def cluster_close_nodes(self, threshold=2, timeout=100):
        # Can only be run after link_nodes...
        thresholded_edges = [x for x in self.edges.values() if x.pixel_length <= threshold]
        candidateset = {x.start.uid for x in thresholded_edges}.union({x.end.uid for x in thresholded_edges})

        # Find
        grouped_candidates = []
        try:
            while True:
                # Get a new uid
                uid = candidateset.pop()
                # Run breadth_first analysis to find all points reachable in jumps of 2 px or less
                cluster = breadth_first(uid, candidateset, self, threshold, timeout)
                # Remove all found points from candidate set as no matter where we start it should be the same cluster
                candidateset = candidateset.difference(cluster)
                grouped_candidates.append(cluster)
        except KeyError:
            #             print(f"Done in {len(grouped_candidates)} runs")
            pass

        # TODO: Maybe check here to see if linked nodes has actually been run.

        final_clusters = {}
        for x in grouped_candidates:
            array = np.array(list(x))
            centroid = np.mean(array, axis=0)
            # Find the euclidian distance of each point from the centroid
            final = np.linalg.norm(array - centroid, ord=2, axis=1)
            # Get the point that is the closest
            closest = tuple(array[np.argmin(final), :])
            final_clusters[closest] = x.difference({closest})
        # Final clusters are in the form (uid_closest_to_centroid):{uids_to_be_clustered}

        edges_to_purge = set()
        nodes_to_purge = set()
        # For each cluster
        for keyuid, topurge in final_clusters.items():
            keynode = self.nodes[keyuid]
            # For each node uid to purge
            nodes_to_purge = nodes_to_purge.union(topurge)
            for purgeuid in topurge:
                # For each connected edge
                for euid, e in self.nodes[purgeuid].connected_edges.items():
                    # Rebind start and endpoint
                    if e.start.uid in topurge:
                        e.start = keynode
                    if e.end.uid in topurge:
                        e.end = keynode
                    if e.start.uid == e.end.uid:
                        edges_to_purge.add(euid)
        self.remove_edges_by_uid(edges_to_purge)
        self.remove_nodes_by_uid(nodes_to_purge)
        # TODO: Check whether edge uids in self.edges need to be reassigned
        self.clean_edges(regenuids=True)

        return self

    def _check_integrity(self):
        detectederror = False
        # Check number of nodes
        if len(self.nodes) < 1 or len(self.edges) < 1:
            detectederror = True
        # Check number of edges
        # Check node uids match uids in nodes
        # Check edge uids match uids of edges
            # Check edge uids match linked nodeuids (maybe)

        node_dict_errors = [str(key) for key, node in
                            tqdm(self.nodes.items(), bar_format="Checking Node Dict UIDs: {l_bar}{bar}{r_bar}") if
                            key != tuple(node.position)]
        edge_dict_errors = [str(key) for key, edge in
                            tqdm(self.edges.items(), bar_format="Checking Edge Dict UIDs: {l_bar}{bar}{r_bar}") if
                            key != edge.start.uid + edge.end.uid]
        if len(node_dict_errors) > 0 or len(edge_dict_errors) > 0:
            detectederror = True
        if detectederror:
            print("\033[31mERROR DETECTED!")
        else:
            print(f"\033[32mNo Errors Found!")
        print(f"Nodes: {len(self.nodes)}\n"
              f"Edges: {len(self.edges)}")
        if node_dict_errors:
            print("Node Dict Errors:\n" + "\n".join(node_dict_errors))
        else:
            print("Node Dict Errors: 0")

        if edge_dict_errors:
            print("Edge Dict Errors:\n" + "\n".join(edge_dict_errors))
        else:
            print("Edge Dict Errors: 0")

        print("\033[0m")   # Clear colour

    @staticmethod
    def fromimage(image):
        # Load from path if needed
        if isinstance(image, str):
            image = io.imread(image)[:, :, :3]
            image = (rgb2gray(image) > 0)

        net = NJANet(image)
        net = net.skeletonize().find_nodes()
        net = net.find_directions()
        net = net.trace_paths().clean_edges()
        # net = net.link_nodes_to_edges()   # Probably don't need to do this as clean_edges should already link.
        #         net = net.cluster_close_nodes()
        return net

    def plot(self, plotoriginal=False):
        fig, ax = plt.subplots()
        if plotoriginal:
            ax.imshow(self.image, cmap=plt.cm.gray)
        else:
            ax.imshow(self.skel, cmap=plt.cm.gray)

        if self.nodes:
            nodes = np.asarray([x.position for x in self.nodes.values()])
            ax.plot(nodes[:, 1], nodes[:, 0], color='yellow', marker='o',
                    linestyle='None', markersize=3)
        if self.edges:
            edges = [x.plotting_repr for x in self.edges.values()]
            line_segments = LineCollection(edges)
            ax.add_collection(line_segments)
        plt.show()

    def plot_with_pixeldensity(self):
        gridsize = (1, 7)
        fig = plt.figure(figsize=(18, 10))
        ax0 = plt.subplot2grid(gridsize, (0, 0), colspan=6, rowspan=1)
        ax1 = plt.subplot2grid(gridsize, (0, 6))

        ax0.imshow(self.image, cmap=plt.cm.gray)
        amt = np.mean(self.image, axis=1)
        depth = np.arange(0, self.image.shape[0])
        ax1.plot(amt, depth, color="grey")
        ax1.invert_yaxis()
        ax1.axes.xaxis.set_ticks([])
        ax1.axes.yaxis.set_ticks([])
        ax1.set_frame_on(False)
        ax1.fill_betweenx(depth, amt, where=depth >= 0, color='lightgrey')
        plt.show()

    def plot_with_nodedensity(self):
        gridsize = (1, 7)
        fig = plt.figure(figsize=(18, 10))
        ax0 = plt.subplot2grid(gridsize, (0, 0), colspan=6, rowspan=1)
        ax1 = plt.subplot2grid(gridsize, (0, 6))
        nodedepth = np.asarray([x.position[0] for x in self.nodes.values() if x.juncs == 1])
        ax0.imshow(self.image, cmap=plt.cm.gray)
        if self.nodes:
            nodes = np.asarray([x.position for x in self.nodes.values()])
            ax0.plot(nodes[:, 1], nodes[:, 0], color='yellow', marker='o',
                     linestyle='None', markersize=3)
        #         amt = np.mean(self.image, axis=1)
        #         depth = np.arange(0, self.image.shape[0])
        sns.histplot(y=nodedepth, bins=100, kde=False, color="#aaaaaa", ax=ax1)
        #         ax1.plot(amt, depth, color ="grey")
        ax1.set_ylim((0, self.image.shape[0]))
        ax1.invert_yaxis()
        ax1.axes.xaxis.set_ticks([])
        ax1.axes.yaxis.set_ticks([])
        ax1.set_frame_on(False)
        #         plt.fill_betweenx(depth, amt,where=depth>=0, color='lightgrey')
        plt.show()

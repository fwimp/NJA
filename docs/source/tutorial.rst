Basic workflow example
======================
This tutorial is designed to provide a quick vignette to demonstrate the usage of NJA under typical conditions.
There is a significant amount of depth and configurability in the package, and for details of that it is worth both
experimenting and also consulting the linked API documentation.

The basic workflow of NJA is as follows:

1. Load image as a :class:`NJA.NJANet` class
2. Skeletonise
3. Find nodes
4. Find directions out of nodes
5. Trace paths
6. Clean edges
7. Cluster nodes

Alternatively we can list this as the functions to use (for quick reference)

1. :class:`NJA.NJANet`
2. :meth:`NJA.NJANet.skeletonize`
3. :meth:`NJA.NJANet.find_nodes`
4. :meth:`NJA.NJANet.find_directions`
5. :meth:`NJA.NJANet.trace_paths` or :meth:`NJA.NJANet.trace_paths_multicore`
6. :meth:`NJA.NJANet.clean_edges`
7. :meth:`NJA.NJANet.cluster_close_nodes`



Load image
-----------------
Practically all interaction with the NJA package is done through the lens of an :class:`NJA.NJANet` class. This
provides easy access to most of the standard actions you would wish to take on an image, alongside storing important
elements of data and traits of the network that could be leveraged for further analysis.

To start with, we therefore should load our image into a :class:`NJA.NJANet` instance, as so:

.. code-block::

    from NJA import NJANet
    import skimage.io as io
    from skimage.color import rgb2gray

    # Load image, ignoring any alpha channels
    image = io.imread("path/to/image.png")[:,:,:3]

    # Convert image into grayscale, and then to a binary image.
    image = (rgb2gray(image) > 0)

    # Load as NJANet instance
    net = NJANet(image)

.. note::
    Images for use with NJA should be converted to a grayscale binary image, with 0 (or False) symbolising background
    pixels and 1 (or True) symbolising foreground pixels. This is performed automatically when using the
    :meth:`NJA.NJANet.fromimage` method of loading images.

Skeletonise
-----------
Once the image is loaded as an :class:`NJA.NJANet` instance, we must perform a morphological transform called a
skeletonization (`see here for an excellent primer <https://homepages.inf.ed.ac.uk/rbf/HIPR2/skeleton.htm>`_).

In essence this reduces down a binary image to a minimal morphological representation that preserves both the extent and
connectivity of the image. This is the magic sauce underlying the later processes of node joining and so is absolutely
required before anything else can be completed. This is performed with :meth:`NJA.NJANet.skeletonize`.

.. code-block::

    # Skeletonise image
    net.skeletonize()

Find nodes
----------
Next we must find the nodes of the skeleton network by detecting the number of white pixels that each
white pixel borders. You will see throughout this documentation mentions of n-nodes such as 1-nodes or 4-nodes. This
terminology refers to the number of pixels bordering a given node.

For example the following pixel grid represents a 3-node as there are 3 orange squares (representing "True" pixels): to
the West, South, and Southeast of the target pixel (in red).

| ⬛⬛⬛
| 🟧🟥⬛
| ⬛🟧🟧

We must ignore any 2-nodes (as these are certainly just a pixel on a continuous line) and mark any other n-nodes as
being nodes of interest. We do this using :meth:`NJA.NJANet.find_nodes`.

.. code-block::

    # Find n-nodes where n != 2
    net.find_nodes()

    print(net.nodes)
    # > {
    # > (82, 659): NJANode([ 82 659] ...),
    # > (82, 975): NJANode([ 82 975] ...),
    # > ...
    # > }

Every node in an :class:`NJA.NJANet` is stored in :class:`NJA.NJANode` instances inside a dictionary called
:attr:`NJANet.nodes`. This allows for easy access and manipulation later if necessary.

If we print the :class:`NJA.NJANet` object we can see that 49 nodes have been found here, but there are no edges yet.

.. code-block::

    print(net)
    # > NJANet
    # > Nodes: 49
    # > Edges: 0

Find directions out of nodes
----------------------------
Once we have a set of nodes, in order to connect them with edges we must first work out which directions a line leaves
the node in the underlying image using :meth:`NJA.NJANet.find_directions`. Whilst this is information already essentially held in the :class:`NJA.NJANode`
object, in situations where you just want locations of the nodes, it is slightly better to defer this process until you
actually require it.

.. code-block::

    # Find directions out of all nodes
    net.find_directions()

    print(net.nodes[(82, 659)].surround)
    # > [[False False False]
    # >  [False False False]
    # >  [True  False False]]

    print(net.nodes[(82, 659)].dirs)
    # > ['SW']

Trace paths
-----------
Next comes the most important step: :meth:`NJA.NJANet.trace_paths`. Here, NJA traces along the lines connecting different nodes in the underlying
skeleton, pixel-by-pixel, until it reaches another node. Those two nodes can then be considered joined and an
:class:`NJA.NJAEdge` object is created. This is performed for every node along every path leading away from that node.

There are possibilities to ignore small gaps in paths, however that is out of scope for this basic tutorial.

.. code-block::

    # Trace all paths
    net.trace_paths()

    print(net)
    # > NJANet
    # > Nodes: 49
    # > Edges: 134

Now this looks fairly good, however if we look closely at a few edges from :attr:`NJANet.edges` we can see a small
problem:

.. code-block::

    print(net.edges[(576, 1227, 879, 1383)])

    # > UID:(576, 1227, 879, 1383)
    # > NJAEdge
    # > (576, 1227)->(879, 1383)
    # > Pixel Length: 367.61731573020387
    # > Direct Length: 340.8005281686048
    # > Length Difference: 26.816787561599085

    print(net.edges[(879, 1383, 576, 1227)])

    # > UID:(879, 1383, 576, 1227)
    # > NJAEdge
    # > (879, 1383)->(576, 1227)
    # > Pixel Length: 367.61731573020387
    # > Direct Length: 340.8005281686048
    # > Length Difference: 26.816787561599085

We have two edges that connect the same nodes (let's call them **A** and **B**) together, i.e there is both
**A** -> **B** and **B** -> **A** in the edge list! This leads to an inflated edge count and could definitely make
analysis trickier. As such we should probably make sure there's one edge for one line in the underlying image.

Clean edges
-----------
Luckily there is a convenient method to do this, called :meth:`NJA.NJANet.clean_edges`. This cleans up the duplicate
edges and makes sure all the IDs correctly line up internally.

.. code-block::

    # Clean edges of duplicates
    net.clean_edges()

    print(net)
    # > NJANet
    # > Nodes: 49
    # > Edges: 67

As expected, we have exactly halved the number of edges, which is brilliant!

There is one final issue however. If we look very closely at the nodes list we can see that there are a lot of nodes
that are basically in the same place!

.. code-block::

    print(list(net.nodes))

    # > [...
    # > (575, 1227),
    # > (576, 1226),
    # > (576, 1227),
    # > (645, 783),
    # > (646, 781),
    # > (646, 782),
    # > (646, 783),
    # > (646, 784),
    # > (647, 782),
    # > (675, 307),
    # > ...]

This significant inflation of node counts is actually an unavoidable consequence of the node detection methodology.
As such, fixing it is a top priority.

Cluster nodes
-------------
We can fix this using :meth:`NJA.NJANet.cluster_close_nodes`. This method uses a breadth-first search from candidate
nodes to group any nodes that are less than a certain threshold distant into one single representative node. This
threshold functions as a control on the amount of simplification you wish to apply to your network. In practice you
always want to cluster at a distance of at least 1.5 (to catch nodes that are pixel adjacent diagonally as well as
cardinally).

.. code-block::

    net.cluster_close_nodes(10)

    print(net)
    # > NJANet
    # > Nodes: 20
    # > Edges: 19

    print(list(net.nodes))
    # > [...
    # > (521, 791),
    # > (564, 421),
    # > (576, 1227),
    # > (646, 783),
    # > (675, 307),
    # > (765, 773),
    # > ...]

We can see here that now all of those similar nodes have been turned into one representative node, and the tiny
interconnecting edges have been removed too.

Once this has been completed, you can start doing any analysis you so choose. There are pieces of analysis built in to
NJA such as :meth:`NJA.NJANet.find_basenode` and :meth:`NJA.NJANet.basenode_to_roottip_distances`, or you can create
your own from the network. In the future we hope to provide interfaces for packages such as networkx too!


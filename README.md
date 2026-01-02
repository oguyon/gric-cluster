# stream-cluster
fast image clustering

Image Clustering Algorirth


Programmed in C language, optimized for speed
Compiler: gcc
Build system: cmake
Use CFITSIO library to read and write FITS files.

The goal of this algorithm is to rapidly group frames (images) into clusters.

Input to be provided as command line arguments:
rlim: distance limit for a frame to belong to a cluster
deltaprob (option, -dprob, default 0.01) incremental probability reward. High value indicates that the next image is most likely to belong to the same cluster as the last image.
maxnbclust (option, -maxcl, default value 1000): maximum number of clusters. stop program if/when number of clusters reaches this limit
maxnbfr (option, -maxim, default value 100000) : maximum number of frames to be clustered. Stop program if/when this limit is reached.
FITS file name containing frames to be clustered. This is a 2D or 3D image. The last axis is the frame number. For example, to cluster 500 2D frames of size 100x200, the frames are stored in a 100x200x500 3D FITS cube. The data format can be integers of floats. Stop program when the end of the FITS cube is reached.
Output:
ASCII file allocating frames to clusters: column 1 is frame index, column 2 is cluster index
FITS cube showing the anchor point for each cluster
One FITS cube per cluster. Each cube has all of the frames belonging to the cluster.
Text output on stdout: number of clusters, time to process, number of calls to framedist() function.

The algorithm shall group frames in clusters, based on distance. Each frame is allocated to a single cluster. A cluster is defined by its anchor point (which is a frame). All frames allocated to a cluster are within the distance limit to the anchor point. 

The distance between images is the euclidean distance.

Code structure:

Translation unit “framedistance.c”
A single function “framedist()” computes the distance between two frames, taking as argument the pointers to the frames. The same function can also compute frame-to-cluster and cluster-to-cluster distances.

Translation unit “frameread.c”
Handles reading input frames.
Contains a function “getframe()” which loads up the next frame, returning a pointer to it.

Translation unit “cluster.c”
This is where the main logic goes. The main function includes a loop calling getframe() to process the next frame. 

Notations:
dcc(ci,cj) is the cluster-to-cluster distance between clusters ci and cj. The distance is the euclidean distance between their anchor points
dfc(fi,cj) is the frame-to-cluster distance between frame fi and cluster cj.
dff(fi,fj) is the frame-to-frame distance between frames fi and fj.
Ncl is the current number of clusters (initially 0)
prob(ci) is the probability that a new frame will belong to cluster ci. Initially set to 0.

Initialization:
Set maxNcl as the max number of clusters
Create a dccarray, a maxNcl x maxNcl array, double format. This is the cluster-to-cluster distance matrix. Initialized it to -1.0 (not measured).

The steps for allocating this image to a cluster are as follows. The frame index is fi, to be incremented from 0 until one of the stop conditions is met.
[step 0] If no cluster exists, create a new one (index c0=0), with its anchor point equal to the frame, set its probability prob(c0)=1.0, set Ncl to 1, and terminate the loop iteration. Otherwise proceed to the next step.
[step 1] Normalize probabilities such that the sum of prob(cj) over all Ncl current clusters cj is unity
[step 2] Update the probability-sorted indexing array of clusters sorted by decreasing prob() using quicksort. The index is noted probsortedclindex[]
[step 3] Set k=0. Initialize the integer Ncl-array of possible cluster membership clmembflag[] to 1, indicating that at this point all clusters are valid candidates.
[step 4] Compute distance dfc(fi,cj) between the frame fi and the cluster cj=probsortedclindex[k]. If distance is less than rlim, allocate image to this cluster, increment prob(cj) by deltaprob, and terminate the loop iteration. Otherwise proceed to the next step.
[step 5] Use the previously computed dfc(fi,cj) value to update the Ncl-array of possible cluster membership using the triangle inequalities as follows, for each cluster index cl starting at 0:
If clmembflag[cl]=0, leave unchanged 
Otherwise, if dccarray[cj,cl]-dfc(fi,cj) > rlim, then set clmembflag[cl]=0 as cluster cl is not a possible candidate, otherwise go to step c
If dfc(fi,cj)-dccarray[cj,cl] > rlim, then set clmembflag[cl]=0
[step 6] Increment k to the next value where clmembflag[cl]=1, and go back to step 5. If reaching the end of the list of clusters, go to the next step.
[step 7] Increment Ncl and create a new cluster with probability prob()=1.0, using frame fi as its anchor point. Compute all distances between this new cluster and the existing ones, updating the corresponding values in dccarray.


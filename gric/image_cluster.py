import subprocess
import os
import tempfile
import shutil
import re
import numpy as np
import sys

class ImageCluster:
    """
    A wrapper for the gric-cluster binary.

    This class provides a Python interface to the main clustering executable,
    allowing for configuration, execution, and parsing of results.
    """
    def __init__(self, rlim, binary_path="gric-cluster", **kwargs):
        """
        Initializes the ImageCluster tool.

        Args:
            rlim (float): The radius limit, a fundamental parameter for the
                clustering algorithm.
            binary_path (str, optional): The path to the `gric-cluster`
                executable. Defaults to "gric-cluster", assuming it is in
                the system's PATH.
            **kwargs: Additional command-line options to pass to the binary.
                These are converted to command-line flags. For example,
                `ImageCluster(..., stream=True, maxcl=500)` will be converted
                to the flags `-stream -maxcl 500`. To enable anchor output,
                pass `anchors=True`.
        """
        self.rlim = rlim
        self.binary_path = binary_path
        self.options = kwargs

    def run(self, input_file, output_dir=None):
        """
        Run clustering on a given input source.

        This method executes the `gric-cluster` binary with the specified
        configuration and input.

        Args:
            input_file (str): The input data source. This can be:
                - A path to a single file (e.g., FITS, video, .txt).
                - The name of an ImageStreamIO stream (if using `-stream`).
                - A path to a text file containing a list of other files
                  (one per line), when used with the `filelist=True` kwarg
                  in the constructor.
            output_dir (str, optional): Path to the directory where output
                files will be saved. If None, a directory named
                `<input_file>.clusterdat` will be created automatically.

        Returns:
            dict: A dictionary containing the results of the run, including:
                - 'stdout' (str): The complete standard output from the binary.
                - 'output_dir' (str): The path to the output directory used.
                - 'log_file' (str): The path to the detailed run log.
                - 'total_clusters' (int): The number of clusters found.
                - 'assignments' (list[int]): A list of cluster IDs for each
                  input frame, if parsing was successful.
        """
        if output_dir is None:
            base = os.path.basename(input_file)
            base, _ = os.path.splitext(base)
            output_dir = f"{base}.clusterdat"

        cmd = [self.binary_path, str(self.rlim)]

        # Add options
        for k, v in self.options.items():
            if v is True:
                cmd.append(f"-{k}")
            elif v is not False and v is not None:
                cmd.append(f"-{k}")
                cmd.append(str(v))

        
        cmd.append("-outdir")
        cmd.append(output_dir)

        cmd.append(input_file)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Binary not found at '{self.binary_path}'. Ensure it's in your PATH or provide a full path.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"gric-cluster failed:\n{e.stderr}") from e


        res = self._parse_stdout(result.stdout)
        res['output_dir'] = output_dir
        res['log_file'] = os.path.join(output_dir, 'cluster_run.log')

        # Locate output file to parse assignments
        clustered_path = None
        if 'CLUSTERED_FILE' in res['stdout']:
            match = re.search(r"CLUSTERED_FILE: (.+)", res['stdout'])
            if match:
                clustered_path = match.group(1)

        if clustered_path and os.path.exists(clustered_path):
            res.update(self._read_clustered_file(clustered_path))
        else:
            # Fallback for older versions or if CLUSTERED_FILE is not in stdout
            membership_path = os.path.join(output_dir, "frame_membership.txt")
            if os.path.exists(membership_path):
                 res.update(self._read_membership_file(membership_path))


        return res

    def run_sequence(self, data):
        """
        A convenience wrapper to run clustering on an in-memory sequence of points.

        This method writes the data to a temporary text file and then calls
        the standard `run` method. The temporary files are cleaned up afterwards.

        Args:
            data (list[list[float]]): A list of points, where each point is a
                list of coordinates (e.g., `[[x1, y1], [x2, y2], ...]`).

        Returns:
            dict: The results dictionary from the `run` method.
        """
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp_path = tmp.name
            for point in data:
                line = " ".join(map(str, point))
                tmp.write(line + "\n")
        
        try:
            return self.run(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                # Also remove the generated clustered file
                base, _ = os.path.splitext(tmp_path)
                output_dir = f"{base}.clusterdat"
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.remove(tmp_path)
            

    def _parse_stdout(self, stdout):
        """Parses the stdout of gric-cluster for key statistics."""
        res = {'stdout': stdout}
        for line in stdout.split('\n'):
            if "Total clusters:" in line:
                res['total_clusters'] = int(line.split(':')[1].strip())
            if "STATS_CLUSTERS:" in line:
                res['total_clusters'] = int(line.split(':')[1].strip())
            if "Processing time:" in line:
                try:
                    res['time_ms'] = float(line.split(':')[1].strip().replace('ms',''))
                except:
                    pass
        return res

    def _read_clustered_file(self, filepath):
        """Parses a .clustered.txt file to get assignments."""
        assignments = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split()
                if len(parts) < 2: continue
                try:
                    cid = int(parts[1])
                    assignments.append(cid)
                except ValueError:
                    continue
        return {'assignments': assignments}

    def _read_membership_file(self, filepath):
        """Parses a frame_membership.txt file to get assignments."""
        assignments = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split()
                if len(parts) < 2: continue
                try:
                    cid = int(parts[1])
                    assignments.append(cid)
                except (ValueError, IndexError):
                    continue
        return {'assignments': assignments}


def plot_clusters(points_file, log_file, output_file=None, svg=False, font_size=None, binary_path="gric-plot"):
    """
    Calls the gric-plot binary to generate a visualization of clustering results.

    Args:
        points_file (str): The original input text file containing point coordinates.
        log_file (str): The log file generated by gric-cluster.
        output_file (str, optional): Path to save the resulting image. If None,
            the name is inferred from `points_file`.
        svg (bool, optional): If True, output an SVG image instead of a PNG.
            Defaults to False.
        font_size (float, optional): Font size for labels in the plot.
        binary_path (str, optional): Path to the `gric-plot` executable.
            Defaults to "gric-plot", assuming it's in the system's PATH.

    Returns:
        str or None: The path to the generated plot image file, or None if
            the path could not be determined from the output.
    """
    cmd = [binary_path]
    if svg:
        cmd.append("-svg")
    if font_size:
        cmd.extend(["-fs", str(font_size)])
    
    cmd.append(points_file)
    cmd.append(log_file)
    if output_file:
        cmd.append(output_file)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Binary not found at '{binary_path}'. Ensure it's in your PATH or provide a full path.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gric-plot failed:\n{e.stderr}") from e
    
    print(result.stdout)
    
    output_filename_match = re.search(r"Saving (PNG|SVG) output: (.+)", result.stdout)
    if output_filename_match:
        return output_filename_match.group(2).strip()
    return None


def plot_locate_results(log_file, output_file=None, binary_path="gric-locate-plot"):
    """
    Calls the gric-locate-plot binary to generate a visualization of location results.

    Args:
        log_file (str): The 'locate_run.log' file generated by gric-locate.
        output_file (str, optional): Path to save the resulting PNG image. If None,
            the name is inferred by the C binary (e.g., 'locate_histogram.png').
        binary_path (str, optional): Path to the `gric-locate-plot` executable.
            Defaults to "gric-locate-plot", assuming it's in the system's PATH.

    Returns:
        str or None: The path to the generated plot image file, or None if
            the path could not be determined from the output.
    """
    cmd = [binary_path, log_file]
    if output_file:
        cmd.append(output_file)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Binary not found at '{binary_path}'. Ensure it's in your PATH or provide a full path.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gric-locate-plot failed:\n{e.stderr}") from e
    
    print(result.stdout)
    
    # The new gric-locate-plot binary has a similar output format for the saved file
    output_filename_match = re.search(r"Saving PNG output: (.+)", result.stdout)
    if output_filename_match:
        return output_filename_match.group(1).strip()
    return None



def make_test_sequence(n_points, output_file, pattern,
                       repeats=None, noise=None, shuffle=False,
                       binary_path="gric-mktxtseq"):
    """
    Calls the gric-mktxtseq binary to generate a sequence of test points.

    Args:
        n_points (int): Number of points to generate for the base pattern.
        output_file (str): Path to save the output text file.
        pattern (str): The generation pattern, e.g., '2Drandom', '3Dsphere',
            '2Dwalk0.1', '2Dspiral5.0', '2Dcircle100'.
        repeats (int, optional): Number of times to repeat the base pattern.
        noise (float, optional): Radius of random noise to add to each point.
        shuffle (bool, optional): If True, shuffles the final order of points.
        binary_path (str, optional): Path to the `gric-mktxtseq` executable.
            Defaults to "gric-mktxtseq", assuming it's in the system's PATH.
    """
    cmd = [binary_path, str(n_points), output_file, pattern]
    if repeats:
        cmd.extend(["-repeat", str(repeats)])
    if noise:
        cmd.extend(["-noise", str(noise)])
    if shuffle:
        cmd.append("-shuffle")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Binary not found at '{binary_path}'. Ensure it's in your PATH or provide a full path.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gric-mktxtseq failed:\n{e.stderr}") from e
    
    print(result.stdout)

def make_clustered_file(input_file, membership_file, output_file, rlim=None,
                        binary_path="gric-mkclusteredfile"):
    """
    Calls gric-mkclusteredfile to reconstruct a clustered output file.

    This is useful for creating a single file that contains both the original
    data and the cluster assignments, suitable for some plotting tools.

    Args:
        input_file (str): Path to the original input text file (coordinates).
        membership_file (str): Path to the `frame_membership.txt` file.
        output_file (str): Path for the output clustered file.
        rlim (float, optional): Radius limit value to write into the header.
        binary_path (str, optional): Path to the `gric-mkclusteredfile` executable.
            Defaults to "gric-mkclusteredfile", assuming it's in the system's PATH.
    """
    cmd = [binary_path, input_file, membership_file, output_file]
    if rlim is not None:
        cmd.extend(["-rlim", str(rlim)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Binary not found at '{binary_path}'. Ensure it's in your PATH or provide a full path.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gric-mkclusteredfile failed:\n{e.stderr}") from e
    
    print(result.stdout)

def run_nd_model(dcc_file, dimensions, output_file,
                 temp=None, rate=None, iterations=None,
                 binary_path="gric-NDmodel"):
    """
    Calls gric-NDmodel to perform N-dimensional scaling on a distance matrix.

    This tool uses simulated annealing to find a low-dimensional embedding of
    the clusters based on their inter-cluster distances (`dcc.txt`).

    Args:
        dcc_file (str): Path to the input distance matrix file (`dcc.txt`).
        dimensions (int): The target number of dimensions for the output.
        output_file (str): Path to save the output file with ND coordinates.
        temp (float, optional): Initial temperature for simulated annealing.
        rate (float, optional): Cooling rate for simulated annealing.
        iterations (int, optional): Number of optimization iterations to run.
        binary_path (str, optional): Path to the `gric-NDmodel` executable.
            Defaults to "gric-NDmodel", assuming it's in the system's PATH.
    """
    cmd = [binary_path, dcc_file, str(dimensions), output_file]
    if temp is not None:
        cmd.extend(["-temp", str(temp)])
    if rate is not None:
        cmd.extend(["-rate", str(rate)])
    if iterations is not None:
        cmd.extend(["-iter", str(iterations)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Binary not found at '{binary_path}'. Ensure it's in your PATH or provide a full path.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gric-NDmodel failed:\n{e.stderr}") from e
    
    print(result.stdout)

def get_info(binary_path="gric-info"):
    """
    Calls the gric-info binary to get build and dependency information.

    Args:
        binary_path (str, optional): Path to the `gric-info` executable.
            Defaults to "gric-info", assuming it's in the system's PATH.

    Returns:
        str: The stdout from the command, containing the info dump.
    """
    cmd = [binary_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Binary not found at '{binary_path}'. Ensure it's in your PATH or provide a full path.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gric-info failed:\n{e.stderr}") from e
    
    return result.stdout

def get_frames_in_cluster(assignments, cluster_id):
    """
    Finds all frame indices that were assigned to a specific cluster.

    This is a pure Python helper function that processes the results from a
    completed `ImageCluster.run()` call.

    Args:
        assignments (list[int]): The list of cluster assignments returned
            by `ImageCluster.run()`. The list index corresponds to the frame index.
        cluster_id (int): The ID of the cluster to inspect.

    Returns:
        list[int]: A list of frame indices belonging to the specified cluster.
    """
    if not isinstance(assignments, list):
        raise TypeError("Input 'assignments' must be a list of integers.")
    
    return [i for i, assigned_cid in enumerate(assignments) if assigned_cid == cluster_id]

def load_anchors(anchors_file):
    """
    Loads an anchors.txt file into a dictionary of NumPy arrays.

    Args:
        anchors_file (str): Path to the `anchors.txt` file.

    Returns:
        dict[int, np.ndarray]: A dictionary mapping cluster IDs to their
            anchor coordinate vectors.
    """
    anchors = {}
    with open(anchors_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            try:
                # Format is: cluster_id frame_idx coord1 coord2 ...
                # In the new gric-locate, the format is just: id x y z ...
                cluster_id = int(parts[0])
                coords = np.array([float(p) for p in parts[1:] if p])
                # A check for the format from gric-cluster output is needed.
                # It has an extra column for frame_idx
                if len(coords) > 0 and len(parts) > 2:
                    # Heuristic: if the second part is a single digit, it's probably a frame index
                    if len(parts[1]) < 5: 
                         coords = np.array([float(p) for p in parts[2:] if p])
                
                anchors[cluster_id] = coords
            except (ValueError, IndexError):
                continue
    return anchors

def locate_points(anchors_file, dcc_file, new_input_file, num_neighbors=1, output_dir=None, binary_path="gric-locate"):
    """
    Calls the gric-locate binary to find the nearest clusters for new data.

    This function wraps the high-performance C program that implements the
    triangulation/pruning algorithm to efficiently classify new data points
    against a pre-existing cluster map.

    Args:
        anchors_file (str): Path to the `anchors.txt` or `anchors.fits` file from the original run.
        dcc_file (str): Path to the `dcc.txt` file from the original run.
        new_input_file (str): Path to the new data file to classify.
        num_neighbors (int, optional): The number of nearest clusters to find
            for each point. Defaults to 1.
        output_dir (str, optional): Directory to save the `locate_run.log` file.
            If None, the log is not guaranteed to be saved in a predictable location.
        binary_path (str, optional): Path to the `gric-locate` executable.
            Defaults to "gric-locate", assuming it's in the system's PATH.

    Returns:
        dict[int, list[tuple[int, float]]]:
            A dictionary mapping each frame index from the new input to a list of its
            nearest neighbors. Each neighbor is represented as a tuple of (cluster_id, distance).
            Note: The per-frame distance calculation count is no longer returned here;
            it is aggregated in 'locate_run.log'.
    """
    cmd = [
        binary_path,
        anchors_file,
        dcc_file,
        new_input_file,
        str(num_neighbors)
    ]
    if output_dir:
        cmd.append(output_dir)

    try:
        # Use text=True for string output and check=True to raise error on non-zero exit code
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Binary not found at '{binary_path}'. Ensure it's in your PATH or provide a full path.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gric-locate failed:\n{e.stderr}") from e

    # --- Parse the locate_run.log file ---
    locate_log_file = os.path.join(output_dir, "locate_run.log")
    total_frames_processed = 0
    distance_histogram = {} # Maps num_dist_calcs -> count_of_frames

    if os.path.exists(locate_log_file):
        with open(locate_log_file, 'r') as f:
            hist_parsing = False
            for line in f:
                if line.startswith("STATS_TOTAL_FRAMES_PROCESSED:"):
                    total_frames_processed = int(line.split(":")[1].strip())
                elif line.startswith("STATS_DIST_HIST_START"):
                    hist_parsing = True
                elif line.startswith("STATS_DIST_HIST_END"):
                    hist_parsing = False
                elif hist_parsing:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        try:
                            k = int(parts[0]) # num_dist_calcs
                            c = int(parts[1]) # count_of_frames
                            distance_histogram[k] = c
                        except ValueError:
                            pass
    else:
        # sys module needs to be imported for stderr, but for a one-off print,
        # a direct stderr write is fine or just print as is.
        # For this context, assuming sys is not imported at module level
        import sys
        print(f"Warning: locate_run.log not found at {locate_log_file}", file=sys.stderr)

    # --- Parse the output --- Expected format: "frame_idx: cluster_id (dist) cluster_id (dist) ..."
    locations = {}
    output = result.stdout
    for line in output.split('\n'):
        if not line or line.startswith('#'):
            continue
        
        parts = line.split(':', 1)
        if len(parts) < 2:
            continue
            
        try:
            frame_idx = int(parts[0])
            neighbors_string = parts[1].strip()

            parsed_neighbors = []
            if neighbors_string:
                # Use regex to find all "cluster_id (distance)" pairs
                neighbors = re.findall(r'(\d+) \((\d+\.\d+)\)', neighbors_string)
                
                # Convert to the correct types
                parsed_neighbors = [(int(cid), float(dist)) for cid, dist in neighbors]
            
            locations[frame_idx] = parsed_neighbors
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing line in gric-locate output: {line.strip()} - {e}", file=sys.stderr)
            continue
            
    return {
        'locations': locations,
        'locate_stats': {
            'total_frames_processed': total_frames_processed,
            'distance_histogram': distance_histogram
        }
    }

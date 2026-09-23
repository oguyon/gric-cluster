import subprocess
import os
import tempfile
import shutil
import re

class ImageCluster:
    def __init__(self, rlim, binary_path=None, **kwargs):
        self.rlim = rlim
        if binary_path is None:
            # Check build directory first, then fallback to PATH
            local_bin = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "build", "gric-cluster")
            )
            if os.path.exists(local_bin):
                self.binary_path = local_bin
            else:
                self.binary_path = "gric-cluster"
        else:
            self.binary_path = binary_path
        self.options = kwargs

    def run(self, input_file, output_dir=None):
        """
        Run clustering on an input file.

        Args:
            input_file (str): Path to input file.
            output_dir (str, optional): Output directory.

        Returns:
            dict: Results containing 'stdout', 'stats', and parsed 'assignments' if available.
        """
        cmd = [self.binary_path, str(self.rlim)]

        # Add options
        has_clustered = False
        has_txt = False
        for k, v in self.options.items():
            if k == 'clustered':
                has_clustered = True
            if k == 'txt':
                has_txt = True
            if v is True:
                cmd.append(f"-{k}")
            elif v is not False and v is not None:
                cmd.append(f"-{k}")
                cmd.append(str(v))

        if not has_clustered:
            cmd.append("-clustered")
        if not has_txt:
            cmd.append("-txt")

        if output_dir:
            cmd.append("-outdir")
            cmd.append(output_dir)

        cmd.append(input_file)

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"gric-cluster failed:\n{result.stderr}")

        res = self._parse_stdout(result.stdout)

        # Locate output file to parse assignments
        base = os.path.basename(input_file)
        for ext in ('.fits.fz', '.fits', '.mp4', '.txt'):
            if base.endswith(ext):
                base = base[:-len(ext)]
                break

        if output_dir:
            clustered_path = os.path.join(output_dir, base + ".clustered.txt")
        else:
            clustered_path = os.path.join(base + ".clusterdat", base + ".clustered.txt")

        if clustered_path and os.path.exists(clustered_path):
            res.update(self._read_clustered_file(clustered_path))

        return res

    def run_sequence(self, data):
        """
        Run clustering on a sequence of points (list of lists or numpy array).

        Args:
            data: List of coordinates [ [x,y,...], ... ]

        Returns:
            dict: Clustering results including 'assignments'.
        """
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp_path = tmp.name
            for point in data:
                line = " ".join(map(str, point))
                tmp.write(line + "\n")

        out_dir = tempfile.mkdtemp()
        try:
            return self.run(tmp_path, output_dir=out_dir)
        finally:
            if os.path.exists(tmp_path):
                # Also remove the generated clustered file
                clustered = tmp_path.replace('.txt', '.clustered.txt')
                if os.path.exists(clustered):
                    os.remove(clustered)
                os.remove(tmp_path)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)

    def _parse_stdout(self, stdout):
        res = {'stdout': stdout}
        for line in stdout.split('\n'):
            if "Total clusters:" in line:
                res['total_clusters'] = int(line.split(':')[1].strip())
            if "Processing time:" in line:
                try:
                    res['time_ms'] = float(line.split(':')[1].strip().replace('ms',''))
                except:
                    pass
        return res

    def _read_clustered_file(self, filepath):
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

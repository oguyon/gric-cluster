#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <ctype.h>

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

// Re-use the frame reading capabilities from the main project
#include "frameread.h"

// --- Data Structures ---

typedef struct {
    double *data;
    long dim;
} Point;

typedef struct {
    int id;
    Point point;
} Anchor;

// Used to hold final results and candidates during search
typedef struct {
    int id;
    double dist_sq; // Using squared distance is faster until the final output
} Result;

// Used for the initial pruning stage
typedef struct {
    int id;
    double lower_bound_sq;
} BoundedCandidate;

// --- qsort comparison functions ---

int compare_bounded_candidates(const void *a, const void *b) {
    BoundedCandidate *ca = (BoundedCandidate *)a;
    BoundedCandidate *cb = (BoundedCandidate *)b;
    if (ca->lower_bound_sq < cb->lower_bound_sq) return -1;
    if (ca->lower_bound_sq > cb->lower_bound_sq) return 1;
    return 0;
}

int compare_results(const void *a, const void *b) {
    Result *ca = (Result *)a;
    Result *cb = (Result *)b;
    if (ca->dist_sq < cb->dist_sq) return -1;
    if (ca->dist_sq > cb->dist_sq) return 1;
    return 0;
}


// --- Helper Functions ---

// Forward declaration
void write_locate_log(long *dist_counts, int max_dist_calcs, long total_frames, const char* out_dir);


double dist_sq(Point p1, Point p2) {
    double sum_sq = 0.0;
    for (long i = 0; i < p1.dim; i++) {
        double diff = p1.data[i] - p2.data[i];
        sum_sq += diff * diff;
    }
    return sum_sq;
}


void print_usage(const char *progname) {
    fprintf(stderr, "Usage: %s <anchors_file> <dcc_file> <new_input_file> <X> [output_dir]\n\n", progname);
    fprintf(stderr, "Description:\n");
    fprintf(stderr, "  Locates the X nearest clusters for each frame in a new input file\n");
    fprintf(stderr, "  based on a pre-computed cluster map (anchors and distance matrix).\n");
    fprintf(stderr, "  Input frames are flattened into 1D vectors for comparison.\n\n");
    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "  <anchors_file>    The 'anchors.txt' or 'anchors.fits' file from a gric-cluster run.\n");
    fprintf(stderr, "  <dcc_file>        The 'dcc.txt' file (distance matrix) from the same run.\n");
    fprintf(stderr, "  <new_input_file>  The new data to classify (e.g., a .txt or FITS file).\n");
    fprintf(stderr, "  <X>               The number of nearest clusters to find for each frame.\n");
    fprintf(stderr, "  [output_dir]      Optional: Directory to save 'locate_run.log'. Defaults to current directory.\n");
}

// --- Main ---

int main(int argc, char *argv[]) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    char *anchors_fname = argv[1];
    char *dcc_fname = argv[2];
    char *new_input_fname = argv[3];
    int num_neighbors = atoi(argv[4]);
    const char *output_dir = (argc > 5) ? argv[5] : ".";


    if (num_neighbors < 1) {
        fprintf(stderr, "Error: X (number of neighbors) must be at least 1.\n");
        return 1;
    }

    // --- Step 1: Load DCC matrix ---
    FILE *dcc_file = fopen(dcc_fname, "r");
    if (!dcc_file) {
        perror("Error opening dcc_file");
        return 1;
    }

    int max_id = -1;
    char line[2048];
    while (fgets(line, sizeof(line), dcc_file)) {
        int i, j;
        if (sscanf(line, "%d %d %*f", &i, &j) == 2) {
            if (i > max_id) max_id = i;
            if (j > max_id) max_id = j;
        }
    }
    rewind(dcc_file);
    if (max_id < 0) {
        fprintf(stderr, "Error: Could not find valid entries in dcc_file.\n");
        fclose(dcc_file);
        return 1;
    }
    int dcc_dim = max_id + 1;
    double *dcc_matrix = malloc(dcc_dim * dcc_dim * sizeof(double));
    if(!dcc_matrix) { perror("malloc dcc_matrix"); return 1; }
    for (int i = 0; i < dcc_dim * dcc_dim; i++) dcc_matrix[i] = -1.0;

    while (fgets(line, sizeof(line), dcc_file)) {
        int i, j;
        double dist;
        if (sscanf(line, "%d %d %lf", &i, &j, &dist) == 3) {
            if (i < dcc_dim && j < dcc_dim) {
                dcc_matrix[i * dcc_dim + j] = dist;
                dcc_matrix[j * dcc_dim + i] = dist;
            }
        }
    }
    fclose(dcc_file);

    // --- Step 2: Load anchors file ---
    long point_dim = 0; // This will be the dimension of the feature vector (flattened image size)
    int num_anchors_read = 0; // Count actual valid anchors read into `anchors` array
    Anchor *anchors = NULL;

    char *ext = strrchr(anchors_fname, '.');
    if (ext && (strcmp(ext, ".fits") == 0 || strcmp(ext, ".fit") == 0)) {
        // --- Load anchors from FITS file ---
        #ifdef USE_CFITSIO
        fitsfile *afptr = NULL;
        int status = 0;
        int naxis;
        long naxes[3];
        long fpixel[3] = {1, 1, 1}; // FITS fpixel is 1-indexed

        if (fits_open_file(&afptr, anchors_fname, READONLY, &status)) {
            fits_report_error(stderr, status);
            free(dcc_matrix);
            return 1;
        }
        if (fits_get_img_dim(afptr, &naxis, &status) || fits_get_img_size(afptr, 3, naxes, &status)) {
            fits_report_error(stderr, status);
            free(dcc_matrix);
            fits_close_file(afptr, &status);
            return 1;
        }
        
        int num_anchors_in_fits_file; // How many anchors are actually in the FITS file
        if (naxis == 3) { // Multiple anchors, each a 2D image
            point_dim = naxes[0] * naxes[1];
            num_anchors_in_fits_file = naxes[2];
        } else if (naxis == 2) { // Single anchor, a 2D image
            point_dim = naxes[0] * naxes[1];
            num_anchors_in_fits_file = 1;
        } else {
            fprintf(stderr, "Error: Anchors FITS file must be 2D or 3D (got %dD).\n", naxis);
            free(dcc_matrix);
            fits_close_file(afptr, &status);
            return 1;
        }
        
        if (num_anchors_in_fits_file != dcc_dim) {
            fprintf(stderr, "Warning: Number of anchors in FITS file (%d) does not match DCC matrix dimension (%d). This might lead to issues.\n", num_anchors_in_fits_file, dcc_dim);
        }

        anchors = malloc(dcc_dim * sizeof(Anchor)); // Allocate for all possible cluster IDs from DCC
        if(!anchors) { perror("malloc anchors"); free(dcc_matrix); fits_close_file(afptr, &status); return 1; }
        for (int i = 0; i < dcc_dim; i++) anchors[i].id = -1; // Mark as uninitialized

        for (int i = 0; i < num_anchors_in_fits_file && i < dcc_dim; i++) {
            anchors[i].id = i;
            anchors[i].point.dim = point_dim;
            anchors[i].point.data = malloc(point_dim * sizeof(double));
            if(!anchors[i].point.data) { 
                perror("malloc anchor data"); 
                // Need to free previously allocated anchor data as well
                for(int j=0; j<i; ++j) free(anchors[j].point.data);
                free(anchors);
                free(dcc_matrix); 
                fits_close_file(afptr, &status); 
                return 1; 
            } 
            
            fpixel[2] = i + 1; // FITS slices are 1-indexed for 3D
            if (fits_read_pix(afptr, TDOUBLE, fpixel, point_dim, NULL, anchors[i].point.data, NULL, &status)) {
                fits_report_error(stderr, status);
                // Need to free previously allocated anchor data
                for(int j=0; j<=i; ++j) free(anchors[j].point.data);
                free(anchors);
                free(dcc_matrix);
                fits_close_file(afptr, &status);
                return 1;
            }
            num_anchors_read++;
        }
        fits_close_file(afptr, &status);

        #else
        fprintf(stderr, "Error: CFITSIO support not compiled in. Cannot read FITS anchors file.\n");
        free(dcc_matrix);
        return 1;
        #endif

    } else {
        // --- Load anchors from TXT file (existing logic) ---
        FILE *txt_anchors_file = fopen(anchors_fname, "r");
        if (!txt_anchors_file) { perror("Error opening anchors_file as text"); free(dcc_matrix); return 1; }
        
        rewind(txt_anchors_file);
        int temp_num_anchors_in_file = 0; // Temp variable for counting lines in TXT
        while (fgets(line, sizeof(line), txt_anchors_file)) {
            if (line[0] == '#') continue; 
            if (point_dim == 0) { 
                char *p = line; char *next_p;
                while(*p != '\0' && *p != '\n' && isspace(*p) == 0) { 
                     strtod(p, &next_p); if (p == next_p) break; p = next_p; point_dim++;
                }
            }
            temp_num_anchors_in_file++;
        }
        rewind(txt_anchors_file);
        
        if (point_dim <= 0) { fprintf(stderr, "Error: Could not determine dimension from anchors file.\n"); free(dcc_matrix); fclose(txt_anchors_file); return 1; }
        if (temp_num_anchors_in_file != dcc_dim) { // Sanity check
            fprintf(stderr, "Warning: Number of anchors in file (%d) does not match DCC matrix dimension (%d). This might lead to issues.\n", temp_num_anchors_in_file, dcc_dim);
        }

        anchors = malloc(dcc_dim * sizeof(Anchor)); 
        if(!anchors) { perror("malloc anchors"); free(dcc_matrix); fclose(txt_anchors_file); return 1; }
        for (int i = 0; i < dcc_dim; i++) anchors[i].id = -1; 

        int current_cluster_id = 0; 
        for (int i=0; i < temp_num_anchors_in_file && i < dcc_dim; i++) {
            if(!fgets(line, sizeof(line), txt_anchors_file)) break; // Ensure we don't read past EOF
            while(line[0] == '#') { 
                if(!fgets(line, sizeof(line), txt_anchors_file)) break; // Handle EOF within comment block
            }
            if (line[0] == '#') { // In case of EOF after comments or last line was a comment
                 continue; // This will effectively skip the last cluster if it's a comment
            }
            
            char *p = line;
            anchors[current_cluster_id].id = current_cluster_id;
            anchors[current_cluster_id].point.dim = point_dim;
            anchors[current_cluster_id].point.data = malloc(point_dim * sizeof(double));
            if(!anchors[current_cluster_id].point.data) { 
                perror("malloc anchor data"); 
                // Free previously allocated data
                for(int j=0; j<current_cluster_id; ++j) free(anchors[j].point.data);
                free(anchors);
                free(dcc_matrix);
                fclose(txt_anchors_file);
                return 1;
            }
            for (long j = 0; j < point_dim; j++) {
                anchors[current_cluster_id].point.data[j] = strtod(p, &p);
            }
            num_anchors_read++;
            current_cluster_id++;
        }
        fclose(txt_anchors_file);
    }
    
    // --- Step 3: Initialize new input file ---
    if (init_frameread(new_input_fname, 0, 0, 0) != 0) {
        fprintf(stderr, "Error: Failed to open new input file '%s'\n", new_input_fname);
        free(dcc_matrix);
        if (anchors) {
            for (int i = 0; i < dcc_dim; i++) { // Free up to dcc_dim allocated
                if (anchors[i].id != -1) free(anchors[i].point.data);
            }
            free(anchors);
        }
        return 1;
    }
    
    long input_frame_total_dim;
    if (is_ascii_input_mode() || get_frame_height() == 1) { // If it's a list of points (ASCII or 1D features)
        input_frame_total_dim = get_frame_width();
    } else { // It's an image (FITS, MP4) - assume flattened image data
        input_frame_total_dim = get_frame_width() * get_frame_height();
    }
    
    if (input_frame_total_dim != point_dim) {
        fprintf(stderr, "Error: Dimension mismatch. Anchors (from '%s') are %ldD, but new input frames (from '%s') have %ldD elements (flattened image pixels or coordinates). These dimensions must match for comparison.\n",
                anchors_fname, point_dim, new_input_fname, input_frame_total_dim);
        // Clean up before returning
        close_frameread();
        free(dcc_matrix);
        if (anchors) {
            for (int i = 0; i < dcc_dim; i++) {
                if (anchors[i].id != -1) {
                    free(anchors[i].point.data);
                }
            }
            free(anchors);
        }
        return 1;
    }

    // --- Step 4: Main processing loop ---
    Frame *frame;
    long frame_idx = 0;
    const int N_REF = 3;

    // Allocate histogram for distance calculation counts
    // Assume max of dcc_dim calculations as a safe upper bound for histogram size
    long *dist_counts = (long *)calloc(dcc_dim + 1, sizeof(long));
    if(!dist_counts) { perror("calloc dist_counts"); /* cleanup */ return 1; }
    
    // Select reference anchors
    int ref_indices[N_REF];
    int refs_found = 0;
    for(int i=0; i<dcc_dim && refs_found < N_REF; ++i) {
        if(anchors[i].id != -1) ref_indices[refs_found++] = i;
    }
    
    if (refs_found == 0) {
        fprintf(stderr, "Error: No valid reference anchors found.\n");
        close_frameread();
        free(dcc_matrix);
        if (anchors) {
            for (int i = 0; i < dcc_dim; i++) {
                if (anchors[i].id != -1) {
                    free(anchors[i].point.data);
                }
            }
            free(anchors);
        }
        free(dist_counts);
        return 1;
    }

    while ((frame = getframe()) != NULL) {
        long dist_calculation_count = 0; // Reset for each frame

        Point current_point_flattened;
        current_point_flattened.data = frame->data; // This is the flattened pixel data or ASCII points
        current_point_flattened.dim = input_frame_total_dim; // This should match point_dim
        
        // A) Reference Distances (compute actual distances to N_REF anchors)
        double ref_dists_sq[N_REF];
        for(int i=0; i<refs_found; ++i) {
            ref_dists_sq[i] = dist_sq(current_point_flattened, anchors[ref_indices[i]].point);
            dist_calculation_count++;
        }

        // B) Candidate Generation and Bounding (using lower bounds from triangle inequality)
        BoundedCandidate *bounded_candidates = malloc(num_anchors_read * sizeof(BoundedCandidate));
        int candidate_count = 0;
        for (int i = 0; i < dcc_dim; i++) {
            if (anchors[i].id == -1) continue; // Skip uninitialized anchors

            double max_lower_bound = 0.0;
            for(int j=0; j<refs_found; ++j) {
                // Ensure the dcc_matrix entry is valid (not -1.0)
                if (dcc_matrix[i * dcc_dim + ref_indices[j]] < 0) continue; 
                
                double d_p_r = sqrt(ref_dists_sq[j]);
                double d_a_r = dcc_matrix[i * dcc_dim + ref_indices[j]];
                
                double bound = fabs(d_p_r - d_a_r);
                if (bound > max_lower_bound) {
                    max_lower_bound = bound;
                }
            }
            bounded_candidates[candidate_count].id = i;
            bounded_candidates[candidate_count].lower_bound_sq = max_lower_bound * max_lower_bound; // Store squared for comparison
            candidate_count++;
        }

        // C) Partial Sort (sort candidates by their lower bound distance)
        qsort(bounded_candidates, candidate_count, sizeof(BoundedCandidate), compare_bounded_candidates);

        // D) Iterative Search for Top X
        Result *top_x_results = malloc(num_neighbors * sizeof(Result));
        int results_found_count = 0;
        double dist_cutoff_sq = DBL_MAX; // Initialize with a very large distance squared

        for (int i = 0; i < candidate_count; i++) {
            BoundedCandidate current_cand = bounded_candidates[i];

            // Pruning Check: If the lower bound is already worse than our current Xth best, stop.
            // This is valid because the list is sorted by lower_bound_sq.
            if (current_cand.lower_bound_sq > dist_cutoff_sq) {
                break; 
            }

            // Full Calculation: If not pruned, compute the actual distance to the new point
            double actual_dist_sq = dist_sq(current_point_flattened, anchors[current_cand.id].point);
            dist_calculation_count++;

            // If this actual distance is better than our current Xth best
            if (actual_dist_sq < dist_cutoff_sq) {
                if (results_found_count < num_neighbors) {
                    // List is not full yet, add and increment count
                    top_x_results[results_found_count].id = current_cand.id;
                    top_x_results[results_found_count].dist_sq = actual_dist_sq;
                    results_found_count++;
                } else {
                    // List is full, replace the worst candidate (which is at results_found_count - 1 after sorting)
                    top_x_results[num_neighbors - 1].id = current_cand.id;
                    top_x_results[num_neighbors - 1].dist_sq = actual_dist_sq;
                }

                // Keep the top_x_results array sorted and update the cutoff
                qsort(top_x_results, results_found_count, sizeof(Result), compare_results);
                if (results_found_count == num_neighbors) { // If we have num_neighbors results
                    dist_cutoff_sq = top_x_results[num_neighbors - 1].dist_sq; // Xth worst distance
                }
            }
        }
        
        // Update histogram
        if (dist_calculation_count <= dcc_dim) {
            dist_counts[dist_calculation_count]++;
        }


        // E) Output Results (now without the dist_count per line)
        printf("%ld:", frame_idx);
        for (int i = 0; i < results_found_count; i++) {
            printf(" %d (%.4f)", top_x_results[i].id, sqrt(top_x_results[i].dist_sq));
        }
        printf("\n");

        free(bounded_candidates);
        free(top_x_results);
        free_frame(frame); // Free the frame data
        frame_idx++;
    }

    // --- Write Log and Cleanup ---
    write_locate_log(dist_counts, dcc_dim, frame_idx, output_dir);

    close_frameread();
    free(dcc_matrix);
    free(dist_counts);
    for (int i = 0; i < dcc_dim; i++) {
        if (anchors[i].id != -1) {
            free(anchors[i].point.data);
        }
    }
    free(anchors);

    return 0;
}


void write_locate_log(long *dist_counts, int max_dist_calcs, long total_frames, const char* out_dir) {
    char log_path[4096];
    snprintf(log_path, sizeof(log_path), "%s/locate_run.log", out_dir);
    FILE *f = fopen(log_path, "w");
    if (!f) {
        perror("Could not open locate_run.log for writing");
        return;
    }

    fprintf(f, "STATS_TOTAL_FRAMES_PROCESSED: %ld\n", total_frames);
    fprintf(f, "STATS_DIST_HIST_START\n");
    for (int k = 0; k <= max_dist_calcs; k++) {
        if (dist_counts[k] > 0) {
            // Format: num_calculations num_frames
            fprintf(f, "%d %ld\n", k, dist_counts[k]);
        }
    }
    fprintf(f, "STATS_DIST_HIST_END\n");
    
    fclose(f);
    fprintf(stderr, "locate_run.log written to %s\n", log_path);
}

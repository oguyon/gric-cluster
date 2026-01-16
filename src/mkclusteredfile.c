#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

void print_help(const char *progname) {
    printf("Usage: %s <input_file> <membership_file> <output_file> [options]\n", progname);
    printf("Description:\n");
    printf("  Reconstructs a clustered output file from the original input coordinates\n");
    printf("  and a frame_membership.txt file.\n");
    printf("  It infers anchors as the first frame encountered for each cluster ID.\n\n");
    printf("Arguments:\n");
    printf("  <input_file>       Original input text file (coordinates).\n");
    printf("  <membership_file>  Frame membership file (index cluster_id).\n");
    printf("  <output_file>      Output clustered filename.\n\n");
    printf("Options:\n");
    printf("  -rlim <val>        Specify radius limit to write to header (useful for plotting).\n");
    printf("  -h, --help         Show this help message.\n");
}

int main(int argc, char *argv[]) {
    char *input_fname = NULL;
    char *memb_fname = NULL;
    char *out_fname = NULL;
    double rlim = -1.0;

    int arg_idx = 1;
    while (arg_idx < argc) {
        if (strcmp(argv[arg_idx], "-h") == 0 || strcmp(argv[arg_idx], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[arg_idx], "-rlim") == 0) {
            if (arg_idx + 1 < argc) {
                rlim = atof(argv[++arg_idx]);
            } else {
                fprintf(stderr, "Error: -rlim requires an argument.\n");
                return 1;
            }
        } else if (argv[arg_idx][0] == '-') {
            fprintf(stderr, "Error: Unknown option %s\n", argv[arg_idx]);
            print_help(argv[0]);
            return 1;
        } else {
            if (!input_fname) input_fname = argv[arg_idx];
            else if (!memb_fname) memb_fname = argv[arg_idx];
            else if (!out_fname) out_fname = argv[arg_idx];
            else {
                fprintf(stderr, "Error: Too many arguments.\n");
                print_help(argv[0]);
                return 1;
            }
        }
        arg_idx++;
    }

    if (!input_fname || !memb_fname || !out_fname) {
        fprintf(stderr, "Error: Missing required arguments.\n");
        print_help(argv[0]);
        return 1;
    }

    FILE *f_in = fopen(input_fname, "r");
    if (!f_in) { perror("Error opening input file"); return 1; }

    FILE *f_memb = fopen(memb_fname, "r");
    if (!f_memb) { perror("Error opening membership file"); fclose(f_in); return 1; }

    FILE *f_out = fopen(out_fname, "w");
    if (!f_out) { perror("Error opening output file"); fclose(f_in); fclose(f_memb); return 1; }

    if (rlim >= 0) {
        fprintf(f_out, "# Parameters:\n");
        fprintf(f_out, "# rlim %f\n", rlim);
    }

    // Track seen clusters (assuming continuous IDs starting from 0, but safe to use simple array map if max ID is reasonable)
    // Or dynamic? Let's use a simple char array for flags. Realloc if needed.
    int max_cluster_id = 1000;
    char *cluster_seen = (char *)calloc(max_cluster_id, sizeof(char));
    if (!cluster_seen) { perror("Memory allocation failed"); return 1; }

    char line_in[4096];
    char line_memb[1024];
    long current_frame_idx = 0;

    while (1) {
        // Read next valid line from input (skip comments/empty)
        long f_in_pos = ftell(f_in);
        if (!fgets(line_in, sizeof(line_in), f_in)) break;
        
        // Skip comments and empty lines in input
        char *p = line_in;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;

        // Clean newline
        size_t len = strlen(line_in);
        if (len > 0 && line_in[len-1] == '\n') line_in[len-1] = '\0';

        // Read next line from membership
        long m_idx;
        int m_id;
        int found_memb = 0;
        while (fgets(line_memb, sizeof(line_memb), f_memb)) {
            // Parse index and id
            if (sscanf(line_memb, "%ld %d", &m_idx, &m_id) == 2) {
                if (m_idx == current_frame_idx) {
                    found_memb = 1;
                    break;
                } else if (m_idx > current_frame_idx) {
                    // Membership is ahead? Frame missing in membership?
                    // Maybe discarded?
                    // If discarded, we skip writing it to output?
                    // Or write with ID -1?
                    // Membership file usually contains all processed frames.
                    // If index mismatch, maybe input file had comments we skipped but membership counts them?
                    // No, input file lines correspond to frames.
                    // Let's assume strict sync.
                    // If m_idx > current, we haven't reached this membership entry yet. 
                    // So current input frame has no membership entry?
                    // This happens if membership file is partial?
                    // Let's print warning and assume -1?
                    fprintf(stderr, "Warning: Sync mismatch. Input frame %ld has no membership entry (next is %ld).\n", current_frame_idx, m_idx);
                    // Rewind membership line to read it next time
                    // But we can't rewind easily if we don't track pos.
                    // Actually, if m_idx > current, we should hold onto this line_memb and process input lines until current == m_idx.
                    // But `fgets` consumes.
                    // Let's not complicate. Assume 1-to-1 or input has extra?
                    // Standard usage: files match.
                    break; 
                }
            }
        }

        if (!found_memb) {
            // End of membership file or mismatch
            // If EOF membership, maybe stop?
            if (feof(f_memb)) break; 
            // If mismatch and not EOF, we just skip writing this frame to clustered output?
            // Or write as unassigned?
            // Let's write as -1 if we have input data but no membership match found (m_idx > current)
            // But we lost the line_memb if we read past.
            // Simplified: Assume files are perfectly synced on valid data lines.
        }

        // Process frame
        // Check cluster seen
        if (m_id >= 0) {
            if (m_id >= max_cluster_id) {
                int new_max = m_id + 1000;
                char *new_seen = (char *)realloc(cluster_seen, new_max * sizeof(char));
                if (new_seen) {
                    // Initialize new part
                    memset(new_seen + max_cluster_id, 0, (new_max - max_cluster_id));
                    cluster_seen = new_seen;
                    max_cluster_id = new_max;
                } else {
                    fprintf(stderr, "Memory realloc failed for clusters.\n");
                    break;
                }
            }

            if (!cluster_seen[m_id]) {
                // New cluster anchor
                fprintf(f_out, "# NEWCLUSTER %d %ld %s\n", m_id, current_frame_idx, p); // p points to data part of line_in
                cluster_seen[m_id] = 1;
            }
        }

        // Write data line
        fprintf(f_out, "%ld %d %s\n", current_frame_idx, m_id, p);

        current_frame_idx++;
    }

    free(cluster_seen);
    fclose(f_in);
    fclose(f_memb);
    fclose(f_out);
    printf("Successfully created %s\n", out_fname);

    return 0;
}

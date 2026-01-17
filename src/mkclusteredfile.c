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

    int max_cluster_id = 1000;
    char *cluster_seen = (char *)calloc(max_cluster_id, sizeof(char));
    if (!cluster_seen) { perror("Memory allocation failed"); return 1; }

    char line_in[4096];
    char line_memb[1024];
    long current_frame_idx = 0;

    while (1) {
        if (!fgets(line_in, sizeof(line_in), f_in)) break;
        
        char *p = line_in;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;

        size_t len = strlen(line_in);
        if (len > 0 && line_in[len-1] == '\n') line_in[len-1] = '\0';

        long m_idx;
        int m_id;
        int found_memb = 0;
        while (fgets(line_memb, sizeof(line_memb), f_memb)) {
            if (sscanf(line_memb, "%ld %d", &m_idx, &m_id) == 2) {
                if (m_idx == current_frame_idx) {
                    found_memb = 1;
                    break;
                } else if (m_idx > current_frame_idx) {
                    break; 
                }
            }
        }

        if (!found_memb) {
            if (feof(f_memb)) break; 
        }

        if (m_id >= 0) {
            if (m_id >= max_cluster_id) {
                int new_max = m_id + 1000;
                char *new_seen = (char *)realloc(cluster_seen, new_max * sizeof(char));
                if (new_seen) {
                    memset(new_seen + max_cluster_id, 0, (new_max - max_cluster_id));
                    cluster_seen = new_seen;
                    max_cluster_id = new_max;
                } else {
                    fprintf(stderr, "Memory realloc failed for clusters.\n");
                    break;
                }
            }

            if (!cluster_seen[m_id]) {
                fprintf(f_out, "# NEWCLUSTER %d %ld %s\n", m_id, current_frame_idx, p);
                cluster_seen[m_id] = 1;
            }
        }

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

/**
 * @file main.c
 * @brief Entry point for gric-knn: Out-of-Core Metric-Pruned k-NN Solver.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_defs.h"
#include "knn_cli.h"
#include "knn_engine.h"
#include "knn_loader.h"
#include "knn_writer.h"
#include "shared/cli_colors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(
    int   argc,
    char *argv[])
{
    cli_colors_init();

    KnnConfig config;
    int parse_rc = knn_cli_parse(argc, argv, &config);
    if (parse_rc != 0)
    {
        return (parse_rc == 2) ? 0 : 1;
    }

    knn_cli_print_banner(&config);

    struct timespec load_start, load_end;
    clock_gettime(CLOCK_MONOTONIC, &load_start);

    KnnModel model;
    if (knn_model_load(config.cluster_dir, config.input_data_path, &model,
                       config.use_double) != 0)
    {
        fprintf(stderr, "Error: Failed to load cluster model from '%s'\n", config.cluster_dir);
        return 1;
    }

    if (config.query_data_path != NULL)
    {
        double effective_rlim = (config.rlim_cutoff > 0.0) ?
            config.rlim_cutoff : model.model_rlim;

        if (config.refuse_unclustered == 0)
        {
            if (effective_rlim > 0.0)
            {
                config.refuse_unclustered = 1;
            }
        }
        else if (config.refuse_unclustered == -1)
        {
            config.refuse_unclustered = 0;
        }
    }
    else
    {
        config.refuse_unclustered = 0;
    }

    knn_model_cache_dataset(&model, &config);

    // Profile auto-discovery and loading
    char auto_prof_path[1024];
    const char *prof_to_load = config.prof_filename;
    if (prof_to_load == NULL && !config.no_prof)
    {
        if (gric_profile_find_auto(config.input_data_path,
                                   auto_prof_path,
                                   sizeof(auto_prof_path)))
        {
            prof_to_load = auto_prof_path;
        }
        else
        {
            char cluster_dir_prof[1024];
            snprintf(cluster_dir_prof, sizeof(cluster_dir_prof),
                     "%s/dataset.gricprof", config.cluster_dir);
            FILE *fchk = fopen(cluster_dir_prof, "r");
            if (fchk != NULL)
            {
                fclose(fchk);
                snprintf(auto_prof_path, sizeof(auto_prof_path), "%s", cluster_dir_prof);
                prof_to_load = auto_prof_path;
            }
        }
    }

    if (prof_to_load != NULL)
    {
        if (gric_profile_read_json(prof_to_load, &model.profile) == 0)
        {
            model.has_profile = 1;
            printf("%s[PROFILE]%s Auto-loaded dataset profile: %s\n",
                   ansi_bold_cyan, ansi_reset, prof_to_load);
            if (model.profile.use_sq16)
            {
                printf("  Range: [%.4f, %.4f], SQ16 scale: %.6f, suggested rlim: %.4f\n",
                       model.profile.sq16_params.min_val, model.profile.sq16_params.max_val,
                       model.profile.sq16_params.scale, model.profile.rlim_recommended);
            }
            else
            {
                printf("  Range: [%.4f, %.4f], SQ8 scale: %.6f, suggested rlim: %.4f\n",
                       model.profile.sq8_params.min_val, model.profile.sq8_params.max_val,
                       model.profile.sq8_params.scale, model.profile.rlim_recommended);
            }
        }
    }

    if (config.use_rq8)
    {
        if (knn_model_build_or_load_rq8(&model, &config) != 0)
        {
            fprintf(stderr, "Error: Failed to initialize RQ8 dataset buffer\n");
            knn_model_free(&model);
            return 1;
        }
    }
    else if (config.use_sq16)
    {
        if (knn_model_build_or_load_sq16(&model, &config) != 0)
        {
            fprintf(stderr, "Error: Failed to initialize SQ16 dataset buffer\n");
            knn_model_free(&model);
            return 1;
        }
    }
    else if (config.use_sq8)
    {
        if (knn_model_build_or_load_sq8(&model, &config) != 0)
        {
            fprintf(stderr, "Error: Failed to initialize SQ8 dataset buffer\n");
            knn_model_free(&model);
            return 1;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &load_end);
    double load_time_ms = (load_end.tv_sec - load_start.tv_sec) * 1000.0 +
                          (load_end.tv_nsec - load_start.tv_nsec) / 1000000.0;

    printf("Loaded Pass 1 Model in %.2f ms:\n", load_time_ms);
    printf("  Total Frames:    %ld\n", model.total_dataset_frames);
    printf("  Total Clusters:  %d\n", model.num_clusters);
    printf("  Frame Dimension: %ld x %ld (%ld elements)\n",
           model.frame_width, model.frame_height, model.frame_elements);
    if (config.rlim_cutoff > 0.0)
    {
        printf("  Radius Cutoff:   %.6f\n", config.rlim_cutoff);
    }
    else if (model.model_rlim > 0.0)
    {
        printf("  Cluster Radius:  %.6f (Inherited from Pass 1)\n", model.model_rlim);
    }
    if (config.query_data_path != NULL)
    {
        printf("  Cluster Refusal: %s\n",
               config.refuse_unclustered ?
                   "Enabled (Strict r_lim Membership)" : "Disabled (All Queries)");
    }
    printf("\n");

    KnnResults results;
    memset(&results, 0, sizeof(KnnResults));
    KnnTelemetry telemetry;
    memset(&telemetry, 0, sizeof(KnnTelemetry));

    if (knn_run_search(&config, &model, &results, &telemetry) != 0)
    {
        fprintf(stderr, "Error: k-NN search failed\n");
        knn_model_free(&model);
        return 1;
    }

    struct timespec write_start, write_end;
    clock_gettime(CLOCK_MONOTONIC, &write_start);

    if (knn_write_results(&config, &model, &results) != 0)
    {
        fprintf(stderr, "Error: Failed to write results\n");
    }

    clock_gettime(CLOCK_MONOTONIC, &write_end);
    double write_time_ms = (write_end.tv_sec - write_start.tv_sec) * 1000.0 +
                           (write_end.tv_nsec - write_start.tv_nsec) / 1000000.0;
    long total_queries = (results.num_queries > 0) ? results.num_queries :
                                                     model.total_dataset_frames;
    uint64_t total_brute_force = (uint64_t)total_queries *
                                 (uint64_t)model.total_dataset_frames;
    double prune_pct = 100.0 * (1.0 - (double)telemetry.framedist_calls /
                                       (double)total_brute_force);
    double fps = (telemetry.time_search_ms > 0.0) ?
                 (double)total_queries / (telemetry.time_search_ms / 1000.0) : 0.0;

    printf("\n%sgric-knn Search Summary:%s\n", ansi_bold_cyan, ansi_reset);
    printf("  Total Query Frames:        %ld\n", total_queries);
    printf("  Framedist Computations:    %lu (vs %lu brute-force)\n",
           (unsigned long)telemetry.framedist_calls, (unsigned long)total_brute_force);
    printf("  Metric Pruning Efficiency: %s%.4f%%%s calls pruned!\n",
           ansi_bold_green, prune_pct, ansi_reset);
    if (telemetry.out_of_cluster_rejected > 0)
    {
        printf("  Out-of-Cluster Refused:    %lu queries\n",
               (unsigned long)telemetry.out_of_cluster_rejected);
    }
    printf("  Level 1 Clusters Pruned:   %lu\n", (unsigned long)telemetry.level1_clusters_pruned);
    printf("  Level 2 Anchors Pruned:    %lu\n", (unsigned long)telemetry.level2_anchors_pruned);
    printf("  Level 3 Annular Pruned:    %lu\n", (unsigned long)telemetry.level3_annular_pruned);
    if (telemetry.clusters_graph_evaluated > 0)
    {
        printf("  Clusters Graph Evaluated:  %lu\n",
               (unsigned long)telemetry.clusters_graph_evaluated);
    }
    if (config.use_rq8)
    {
        uint64_t total_rq8_pruned = telemetry.rq8_members_pruned +
                                     telemetry.rq8_graph_pruned;
        printf("  RQ8 Evaluations:           %lu\n",
               (unsigned long)telemetry.rq8_evaluations);
        printf("  RQ8 Lower-Bound Pruned:    %lu\n",
               (unsigned long)total_rq8_pruned);
        printf("  RQ8 Member Pruned:         %lu\n",
               (unsigned long)telemetry.rq8_members_pruned);
        if (telemetry.rq8_graph_pruned > 0)
        {
            printf("  RQ8 Graph Pruned:          %lu\n",
                   (unsigned long)telemetry.rq8_graph_pruned);
        }
    }
    else if (config.use_sq16)
    {
        uint64_t total_sq16_pruned = telemetry.sq16_members_pruned +
                                     telemetry.sq16_graph_pruned;
        printf("  SQ16 Evaluations:          %lu\n",
               (unsigned long)telemetry.sq16_evaluations);
        printf("  SQ16 Lower-Bound Pruned:   %lu\n",
               (unsigned long)total_sq16_pruned);
        printf("  SQ16 Member Pruned:        %lu\n",
               (unsigned long)telemetry.sq16_members_pruned);
        printf("  SQ16 Graph Pruned:         %lu\n",
               (unsigned long)telemetry.sq16_graph_pruned);
        if (config.use_memo)
        {
            printf("  SQ16 Memo Hits:            %lu\n",
                   (unsigned long)telemetry.memo_hits);
            if (model.num_unique_frames > 0)
            {
                printf("  SQ16 Unique Frames:        %ld / %ld\n",
                       model.num_unique_frames, model.total_dataset_frames);
            }
        }
    }
    else if (config.use_sq8)
    {
        uint64_t total_sq8_pruned = telemetry.sq8_members_pruned +
                                    telemetry.sq8_graph_pruned;
        printf("  SQ8 Evaluations:           %lu\n",
               (unsigned long)telemetry.sq8_evaluations);
        printf("  SQ8 Lower-Bound Pruned:    %lu\n",
               (unsigned long)total_sq8_pruned);
        printf("  SQ8 Member Pruned:         %lu\n",
               (unsigned long)telemetry.sq8_members_pruned);
        printf("  SQ8 Graph Pruned:          %lu\n",
               (unsigned long)telemetry.sq8_graph_pruned);
    }
    if (telemetry.graph_seeds_evaluated > 0)
    {
        printf("  Graph Seeds Evaluated:     %lu\n",
               (unsigned long)telemetry.graph_seeds_evaluated);
    }
    if (telemetry.graph_edges_pruned > 0)
    {
        printf("  Graph Edges Pruned:        %lu\n",
               (unsigned long)telemetry.graph_edges_pruned);
    }
    printf("  Multi-Pivot Pruned:        %lu\n",
           (unsigned long)telemetry.multi_pivot_pruned);
    if (telemetry.angular_pruned > 0)
    {
        uint64_t total_graph_cands = telemetry.graph_seeds_evaluated +
                                     telemetry.graph_edges_pruned +
                                     telemetry.angular_pruned;
        double ang_pct = (total_graph_cands > 0) ?
            100.0 * (double)telemetry.angular_pruned / (double)total_graph_cands : 0.0;
        printf("  Angular Cosine Pruned:     %lu (%.1f%% of graph edges)\n",
               (unsigned long)telemetry.angular_pruned, ang_pct);
    }
    if (telemetry.trajectory_warmstarts > 0)
    {
        printf("  Trajectory Warmstarts:     %lu queries (%.1f%%)\n",
               (unsigned long)telemetry.trajectory_warmstarts,
               100.0 * (double)telemetry.trajectory_warmstarts /
                   (double)telemetry.total_queries);
    }
    if (telemetry.global_containment_hits > 0)
    {
        printf("  Global Containment Hits:   %lu queries (%.1f%%)\n",
               (unsigned long)telemetry.global_containment_hits,
               100.0 * (double)telemetry.global_containment_hits /
                   (double)telemetry.total_queries);
    }
    if (telemetry.reciprocal_reused > 0)
    {
        printf("  Reciprocal Reused:         %lu\n",
               (unsigned long)telemetry.reciprocal_reused);
    }
    if (telemetry.two_hop_evaluations > 0 || telemetry.two_hop_pruned > 0)
    {
        printf("  2-Hop Evaluated:           %lu\n",
               (unsigned long)telemetry.two_hop_evaluations);
        printf("  2-Hop Injected to Heap:    %lu\n",
               (unsigned long)telemetry.two_hop_injected);
        printf("  2-Hop Triangle Pruned:     %lu\n",
               (unsigned long)telemetry.two_hop_pruned);
    }
    printf("  Temporal Exclusions:       %lu\n", (unsigned long)telemetry.temporal_pruned);
    printf("  Search Wall Time:          %.2f ms (%.1f fps)\n", telemetry.time_search_ms, fps);
    printf("  Output Write Time:         %.2f ms\n", write_time_ms);
    printf("%sCompleted successfully.%s\n", ansi_bold_green, ansi_reset);

    knn_results_free(&results);
    knn_model_free(&model);
    return 0;
}

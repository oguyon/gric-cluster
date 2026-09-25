/**
 * @file tool_help.c
 * @brief MCP knowledge engine tools for documentation, suite catalog, and recipes.
 */

#include "mcp_tools.h"
#include "shared/help_topics.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * to_lowercase() - Convert string to lowercase in place.
 * @str: String buffer to convert.
 */
static void to_lowercase(
    char *str)
{
    for (size_t ii = 0; str[ii] != '\0'; ii++)
    {
        str[ii] = (char)tolower((unsigned char)str[ii]);
    }
} // to_lowercase

/**
 * struct SuiteProgramEntry - Information record for a GRIC suite executable.
 */
typedef struct
{
    const char *name;
    const char *category;
    const char *summary;
    const char *usage;
} SuiteProgramEntry;

static const SuiteProgramEntry SUITE_PROGRAMS[] = {
    {
        "gric-cluster",
        "Core Clustering",
        "High-performance distance-based clustering engine (offline or streaming).",
        "gric-cluster <rlim> <input> [options]"
    },
    {
        "gric-knn",
        "Nearest Neighbors",
        "Out-of-core metric-pruned k-nearest neighbors search solver.",
        "gric-knn <input_data> <cluster_dir> [options]"
    },
    {
        "gric-knn-avg",
        "Nearest Neighbors",
        "Reconstructs frames and feature averages using k-NN graph connectivity.",
        "gric-knn-avg <input_data> <knn_file> [options]"
    },
    {
        "milk-fpsexec-gric-cluster",
        "Milk Streaming",
        "Standalone Function Parameter Structure daemon for real-time stream clustering.",
        "milk-fpsexec-gric-cluster [options] [fpsname:]<command>"
    },
    {
        "gric-fps-cluster",
        "Milk Streaming",
        "Convenience symlink to milk-fpsexec-gric-cluster.",
        "gric-fps-cluster [options] [fpsname:]<command>"
    },
    {
        "gric-probe",
        "Tuning & Diagnostics",
        "Automated dataset geometry, noise floor, and parameter calibration probe.",
        "gric-probe [options] <dataset_path>"
    },
    {
        "gric-tune",
        "Tuning & Diagnostics",
        "Parameter space exploration and hyperparameter grid tuner.",
        "gric-tune <input_file> [options]"
    },
    {
        "gric-benchmark",
        "Tuning & Diagnostics",
        "Performance benchmarking across synthetic manifolds and algorithms.",
        "gric-benchmark [options]"
    },
    {
        "gric-dimdensity",
        "Tuning & Diagnostics",
        "Local intrinsic dimensionality (LID) and density spectrum estimator.",
        "gric-dimdensity <input_file> [options]"
    },
    {
        "gric-status",
        "Tuning & Diagnostics",
        "Real-time shared memory telemetry monitor (TUI dashboard).",
        "gric-status [shm_file]"
    },
    {
        "gric-plot",
        "Visualization & GUI",
        "Generates SVG/PNG diagnostic and summary plots of clustering runs.",
        "gric-plot <input_file> <run_log> <output_image>"
    },
    {
        "gric-server",
        "Visualization & GUI",
        "Native C HTTP micro-server powering the browser GUI and REST API.",
        "gric-server [options]"
    },
    {
        "gric-ascii2bin",
        "Format Conversion",
        "Encodes ASCII text tables and coordinates into self-describing .bin files.",
        "gric-ascii2bin -dim <D> <input.txt> <output.bin>"
    },
    {
        "gric-bin2ascii",
        "Format Conversion",
        "Decodes and inspects self-describing .bin files to ASCII.",
        "gric-bin2ascii <input.bin> [output.txt]"
    },
    {
        "gric-mktxtseq",
        "Generators & Simulation",
        "Generates synthetic coordinate sequences (walk, spiral, clusters).",
        "gric-mktxtseq <nframes> <outfile> <pattern>"
    },
    {
        "gric-gen-balls",
        "Generators & Simulation",
        "Generates multi-body bouncing ball physics simulation FITS video cubes.",
        "gric-gen-balls [options] <output.fits>"
    },
    {
        "gric-txt2stream",
        "Streaming I/O",
        "Ingests ASCII coordinate streams into ImageStreamIO shared memory.",
        "gric-txt2stream <stream_name> <dim> [options]"
    },
    {
        "gric-stream-to-pipe",
        "Streaming I/O",
        "Pipes raw frame data from live ImageStreamIO shared memory to stdout.",
        "gric-stream-to-pipe <stream_name> [options]"
    },
    { NULL, NULL, NULL, NULL }
};

int mcp_tool_help(
    const cJSON *args,
    cJSON       *res)
{
    const char *topic_str = NULL;
    const char *query_str = NULL;

    if (args != NULL)
    {
        cJSON *t_item = cJSON_GetObjectItemCaseSensitive(args, "topic");
        if (t_item != NULL && cJSON_IsString(t_item))
        {
            topic_str = t_item->valuestring;
        }

        cJSON *q_item = cJSON_GetObjectItemCaseSensitive(args, "query");
        if (q_item != NULL && cJSON_IsString(q_item))
        {
            query_str = q_item->valuestring;
        }
    }

    /* 1. Direct Topic Lookup */
    if (topic_str != NULL && strlen(topic_str) > 0 &&
        strcmp(topic_str, "list") != 0 && strcmp(topic_str, "all") != 0)
    {
        char norm_topic[128];
        strncpy(norm_topic, topic_str, sizeof(norm_topic) - 1);
        norm_topic[sizeof(norm_topic) - 1] = '\0';
        to_lowercase(norm_topic);

        /* Normalize leading dashes if user queried -rlim or --entropy */
        char *search_key = norm_topic;
        while (*search_key == '-')
        {
            search_key++;
        }

        const char *content = help_topic_lookup(search_key);
        if (content != NULL)
        {
            cJSON_AddStringToObject(res, "status", "FOUND");
            cJSON_AddStringToObject(res, "topic", search_key);
            cJSON_AddStringToObject(res, "content", content);
            return 0;
        }

        /* If exact lookup failed, fall back to searching for candidate matches */
        query_str = search_key;
    } // if topic_str

    /* 2. Search / Query / List All */
    const struct HelpTopicEntry *table = help_topic_get_table();
    if (table == NULL)
    {
        cJSON_AddStringToObject(res, "status", "ERROR");
        cJSON_AddStringToObject(res, "message", "Help database unavailable");
        return -1;
    }

    if (query_str != NULL && strlen(query_str) > 0)
    {
        char norm_q[128];
        strncpy(norm_q, query_str, sizeof(norm_q) - 1);
        norm_q[sizeof(norm_q) - 1] = '\0';
        to_lowercase(norm_q);

        cJSON *matches = cJSON_CreateArray();
        int match_count = 0;

        for (size_t ii = 0; table[ii].keyword != NULL; ii++)
        {
            char norm_k[128];
            strncpy(norm_k, table[ii].keyword, sizeof(norm_k) - 1);
            norm_k[sizeof(norm_k) - 1] = '\0';
            to_lowercase(norm_k);

            /* Check keyword substring or content match */
            if (strstr(norm_k, norm_q) != NULL ||
                (table[ii].content != NULL && strcasestr(table[ii].content, norm_q) != NULL))
            {
                cJSON *item = cJSON_CreateObject();
                cJSON_AddStringToObject(item, "topic", table[ii].keyword);

                /* Create concise preview line */
                char preview[160] = "";
                if (table[ii].content != NULL)
                {
                    const char *p = table[ii].content;
                    const char *nl = strchr(p, '\n');
                    if (nl != NULL && (size_t)(nl - p) < sizeof(preview))
                    {
                        strncpy(preview, p, (size_t)(nl - p));
                        preview[nl - p] = '\0';
                    }
                    else
                    {
                        strncpy(preview, p, sizeof(preview) - 1);
                        preview[sizeof(preview) - 1] = '\0';
                    }
                }
                cJSON_AddStringToObject(item, "preview", preview);
                cJSON_AddItemToArray(matches, item);
                match_count++;

                if (match_count >= 20)
                {
                    break;
                }
            } // if match
        } // for ii

        cJSON_AddStringToObject(res, "status", (match_count > 0) ? "MATCHES_FOUND" : "NOT_FOUND");
        cJSON_AddStringToObject(res, "query", query_str);
        cJSON_AddNumberToObject(res, "match_count", match_count);
        cJSON_AddItemToObject(res, "matches", matches);
        return 0;
    } // if query_str

    /* 3. Catalog of all available topics */
    cJSON *topics_arr = cJSON_CreateArray();
    size_t total = 0;

    for (size_t ii = 0; table[ii].keyword != NULL; ii++)
    {
        cJSON_AddItemToArray(topics_arr, cJSON_CreateString(table[ii].keyword));
        total++;
    }

    cJSON_AddStringToObject(res, "status", "CATALOG");
    cJSON_AddNumberToObject(res, "total_topics", (double)total);
    cJSON_AddItemToObject(res, "available_topics", topics_arr);
    return 0;
} // mcp_tool_help

int mcp_tool_list_suite(
    const cJSON *args,
    cJSON       *res)
{
    (void)args;
    cJSON *progs_arr = cJSON_CreateArray();

    for (size_t ii = 0; SUITE_PROGRAMS[ii].name != NULL; ii++)
    {
        cJSON *prog = cJSON_CreateObject();
        cJSON_AddStringToObject(prog, "name", SUITE_PROGRAMS[ii].name);
        cJSON_AddStringToObject(prog, "category", SUITE_PROGRAMS[ii].category);
        cJSON_AddStringToObject(prog, "summary", SUITE_PROGRAMS[ii].summary);
        cJSON_AddStringToObject(prog, "usage", SUITE_PROGRAMS[ii].usage);
        cJSON_AddItemToArray(progs_arr, prog);
    } // for ii

    cJSON_AddStringToObject(res, "status", "SUCCESS");
    cJSON_AddNumberToObject(res, "total_programs",
                            (double)(sizeof(SUITE_PROGRAMS) / sizeof(SUITE_PROGRAMS[0]) - 1));
    cJSON_AddItemToObject(res, "programs", progs_arr);
    return 0;
} // mcp_tool_list_suite

int mcp_tool_get_recipe(
    const cJSON *args,
    cJSON       *res)
{
    const char *recipe_name = NULL;
    if (args != NULL)
    {
        cJSON *r_item = cJSON_GetObjectItemCaseSensitive(args, "recipe");
        if (r_item != NULL && cJSON_IsString(r_item))
        {
            recipe_name = r_item->valuestring;
        }
    }

    if (recipe_name == NULL || strlen(recipe_name) == 0 || strcmp(recipe_name, "list") == 0)
    {
        cJSON *list = cJSON_CreateArray();
        cJSON_AddItemToArray(list, cJSON_CreateString("milk_realtime_streaming"));
        cJSON_AddItemToArray(list, cJSON_CreateString("high_dim_knn_search"));
        cJSON_AddItemToArray(list, cJSON_CreateString("image_cube_clustering"));
        cJSON_AddItemToArray(list, cJSON_CreateString("radius_calibration"));
        cJSON_AddItemToArray(list, cJSON_CreateString("synthetic_sequence"));

        cJSON_AddStringToObject(res, "status", "RECIPES_INDEX");
        cJSON_AddItemToObject(res, "available_recipes", list);
        return 0;
    }

    if (strcmp(recipe_name, "milk_realtime_streaming") == 0)
    {
        const char *steps =
            "# Recipe: Real-Time Stream Clustering with Milk Framework\n\n"
            "1. Inspect input stream status in shared memory:\n"
            "   Call gric_probe_shm(stream_name=\"cam_wfs\")\n\n"
            "2. Launch standalone FPS clustering daemon:\n"
            "   Call gric_fps_run(fps_name=\"gric01\", in_name=\"cam_wfs\", "
            "out_name=\"clust_wfs\", rlim=0.35, use_tmux=true)\n\n"
            "3. Monitor live loop rate and clustering telemetry:\n"
            "   Call gric_fps_status(fps_name=\"gric01\")\n"
            "   Call gric_probe_fps_streams(out_name=\"clust_wfs_assign\")\n\n"
            "4. Dynamically adjust parameters without stopping the stream:\n"
            "   Call gric_fps_set(fps_name=\"gric01\", param=\".rlim\", value=\"0.42\")\n"
            "   Call gric_fps_set(fps_name=\"gric01\", param=\".reset_state\", value=\"ON\")\n\n"
            "5. Stop stream processing cleanly:\n"
            "   Call gric_fps_stop(fps_name=\"gric01\")\n";

        cJSON_AddStringToObject(res, "recipe", recipe_name);
        cJSON_AddStringToObject(res, "title", "Real-Time Milk Stream Clustering");
        cJSON_AddStringToObject(res, "instructions", steps);
        return 0;
    }

    if (strcmp(recipe_name, "high_dim_knn_search") == 0)
    {
        const char *steps =
            "# Recipe: High-Dimensional Quantized kNN Indexing & Retrieval\n\n"
            "1. Encode raw text vectors to self-describing binary format:\n"
            "   $ gric-ascii2bin -dim 64 raw_vectors.txt data.bin\n\n"
            "2. Calibrate dataset geometry and quantization presets:\n"
            "   Call gric_probe_dataset(dataset_path=\"data.bin\")\n\n"
            "3. Run Pass 1 clustering with E8 lattice quantization (EQ16):\n"
            "   $ gric-cluster a1.0 data.bin -maxcl 1500 -eq16 -outdir run.clusterdat\n\n"
            "4. Verify clustering boundary invariants:\n"
            "   Call gric_verify_invariants(run_dir=\"run.clusterdat\", dataset=\"data.bin\")\n\n"
            "5. Run Pass 2 out-of-core metric-pruned kNN retrieval:\n"
            "   $ gric-knn data.bin run.clusterdat -k 10 -multipivot -angular\n";

        cJSON_AddStringToObject(res, "recipe", recipe_name);
        cJSON_AddStringToObject(res, "title", "High-Dimensional Quantized kNN");
        cJSON_AddStringToObject(res, "instructions", steps);
        return 0;
    }

    if (strcmp(recipe_name, "image_cube_clustering") == 0)
    {
        const char *steps =
            "# Recipe: FITS Image Cube Clustering with Spatial Tiling\n\n"
            "1. Probe image dimensions and noise floor:\n"
            "   Call gric_probe_dataset(dataset_path=\"image_cube.fits\")\n\n"
            "2. Cluster with 2x2 spatial tiling to capture localized patterns:\n"
            "   $ gric-cluster a1.2 image_cube.fits -tiles 2x2 -maxcl 1000 "
            "-outdir cube.clusterdat\n\n"
            "3. Inspect clustering telemetry and pruning efficiency:\n"
            "   Call gric_inspect_run(run_dir=\"cube.clusterdat\")\n";

        cJSON_AddStringToObject(res, "recipe", recipe_name);
        cJSON_AddStringToObject(res, "title", "Image Cube Tiled Clustering");
        cJSON_AddStringToObject(res, "instructions", steps);
        return 0;
    }

    if (strcmp(recipe_name, "radius_calibration") == 0)
    {
        const char *steps =
            "# Recipe: Automatic Radius Threshold Calibration\n\n"
            "1. Rapid distance spectrum scan:\n"
            "   $ gric-cluster -scandist input_data.bin\n\n"
            "2. Select adaptive median factor:\n"
            "   - a0.5 : Fine granularity, many small clusters (high fidelity)\n"
            "   - a1.0 : Balanced segmentation (recommended default)\n"
            "   - a1.5 : Coarse grouping, fewer clusters\n\n"
            "3. Or use gric_probe_dataset for percentile recommendations (D1%, D10%, D25%).\n";

        cJSON_AddStringToObject(res, "recipe", recipe_name);
        cJSON_AddStringToObject(res, "title", "Radius Calibration Guide");
        cJSON_AddStringToObject(res, "instructions", steps);
        return 0;
    }

    if (strcmp(recipe_name, "synthetic_sequence") == 0)
    {
        const char *steps =
            "# Recipe: Generating Synthetic Manifolds & Benchmarking\n\n"
            "1. Generate synthetic coordinate sequence:\n"
            "   $ gric-mktxtseq 5000 /tmp/spiral.txt 2Dspiral\n\n"
            "2. Cluster synthetic sequence:\n"
            "   $ gric-cluster 0.15 /tmp/spiral.txt -outdir /tmp/spiral.clusterdat\n\n"
            "3. Inspect results and plot:\n"
            "   Call gric_inspect_run(run_dir=\"/tmp/spiral.clusterdat\")\n"
            "   $ gric-plot /tmp/spiral.txt /tmp/spiral.clusterdat/cluster_run.log /tmp/plot.png\n";

        cJSON_AddStringToObject(res, "recipe", recipe_name);
        cJSON_AddStringToObject(res, "title", "Synthetic Sequence Generation");
        cJSON_AddStringToObject(res, "instructions", steps);
        return 0;
    }

    cJSON_AddStringToObject(res, "status", "UNKNOWN_RECIPE");
    cJSON_AddStringToObject(res, "requested", recipe_name);
    return -1;
} // mcp_tool_get_recipe

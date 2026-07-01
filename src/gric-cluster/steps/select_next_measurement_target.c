#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"

int select_next_measurement_target(
    ClusterConfig *config,
    ClusterState  *state,
    int           *k_search,
    const int     *pred_candidates,
    int            num_preds,
    int           *current_pred_idx)
{
    if (pred_candidates && *current_pred_idx < num_preds)
    {
        while (*current_pred_idx < num_preds)
        {
            int cj = pred_candidates[*current_pred_idx];
            (*current_pred_idx)++;
            if (cj >= 0 && cj < state->num_clusters && state->scratch.clmembflag[cj])
            {
                return cj;
            }
        }
    }

    if (!config->optim.gprob_mode)
    {
        while (*k_search < state->num_clusters &&
               state->scratch.clmembflag[state->scratch.probsortedclindex[*k_search]] == 0)
        {
            (*k_search)++;
        }
        if (*k_search >= state->num_clusters)
        {
            return -1;
        }
        int cj = state->scratch.probsortedclindex[*k_search];
        (*k_search)++;
        return cj;
    }
    else
    {
        double max_p = -1.0;
        int cj = -1;
        for (int i = 0; i < state->num_clusters; i++)
        {
            if (state->scratch.clmembflag[i])
            {
                double p = state->scratch.mixed_probs[i] *
                           state->scratch.current_gprobs[i];
                if (p > max_p)
                {
                    max_p = p;
                    cj = i;
                }
            }
        }
        return cj;
    }
}

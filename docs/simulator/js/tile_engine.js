/**
 * GRIC Simulator - tile_engine.js
 * Part of the GRIC Interactive Algorithm Simulator
 */

//  3. 1D SUBSPACE TILE ENGINE (Input Dimension Partitioning)
    // =========================================================================

    class Tile1D {
      constructor(axisName) {
        this.axisName = axisName;
        this.clusters = [];
        this.dcc = [];
        this.totalEvals = 0;
        this.naiveEvals = 0;
      }

      reset() {
        this.clusters = [];
        this.dcc = [];
        this.totalEvals = 0;
        this.naiveEvals = 0;
      }

      cluster1D(val, rThreshold, isExplain) {
        const K = this.clusters.length;
        if (K === 0) {
          this.clusters.push({ id: 0, val, members: 1, prob: 1.0 });
          this.dcc = [[0.0]];
          return { assigned: 0, evals: 0, isNew: true, evaluated: [], pruned: [], steps: [] };
        }

        this.naiveEvals += K;
        const clmembflag = new Array(K).fill(true);
        const evaluated = [];
        const pruned = [];
        const steps = [];
        let assigned = -1;
        let evals = 0;

        const order = [];
        for (let i = 0; i < K; i++) order.push(i);
        order.sort((a, b) => {
          let probA = this.clusters[a].prob || 1.0;
          let probB = this.clusters[b].prob || 1.0;
          if (useXTile && currentJointTuple) {
            const keyA = this.axisName === 'X' ? `${a}_${currentJointTuple.cy}` : `${currentJointTuple.cx}_${a}`;
            const keyB = this.axisName === 'X' ? `${b}_${currentJointTuple.cy}` : `${currentJointTuple.cx}_${b}`;
            const cntA = jointTuplesMap.has(keyA) ? jointTuplesMap.get(keyA).count : 0;
            const cntB = jointTuplesMap.has(keyB) ? jointTuplesMap.get(keyB).count : 0;
            probA += cntA * xtileDecay;
            probB += cntB * xtileDecay;
          }
          return probB - probA;
        });

        for (let target of order) {
          if (!clmembflag[target]) continue;
          evals++;
          distSampleCluster++;
          distSampleClusterLast++;
          const targetCl = this.clusters[target];
          const d = Math.abs(val - targetCl.val);
          const isMatch = d < rThreshold;
          evaluated.push({ target, dist: d, match: isMatch });

          let selectionReason = "";
          if (useXTile && (this.axisName === 'X' || this.axisName === 'Y' || this.axisName === 'Z') && currentJointTuple) {
            selectionReason = `Cross-tile subspace prior transferred from orthogonal axes (-xtile)`;
          } else if (useTM && this.prevAssigned >= 0 && this.transitionCounts[this.prevAssigned] && this.transitionCounts[this.prevAssigned][target] > 0) {
            const transP = this.transitionCounts[this.prevAssigned][target] / (this.clusters[this.prevAssigned].members || 1);
            selectionReason = `1D Markov transition prior from C${this.prevAssigned} (P=${transP.toFixed(3)})`;
          } else {
            selectionReason = `Highest 1D membership weight (prob=${(targetCl.prob || 1.0).toFixed(2)}, ${targetCl.members} members)`;
          }

          if (isExplain) {
            steps.push({
              type: isMatch ? 'match' : 'mismatch',
              title: `Tile ${this.axisName}: ${isMatch ? 'Match' : 'Mismatch'} on Anchor C${target}`,
              text: `1D distance |${val.toFixed(3)} - ${targetCl.val.toFixed(3)}| = ${d.toFixed(4)} ${isMatch ? '<=' : '>'} rlim (${rThreshold.toFixed(3)}).<br><b>Selection reason:</b> ${selectionReason}.`
            });
          }

          if (isMatch) {
            assigned = target;
            targetCl.members++;
            targetCl.prob = (targetCl.prob || 1.0) + 0.05;
            break;
          }

          clmembflag[target] = false;
          let prunedCount = 0;
          for (let cl = 0; cl < K; cl++) {
            if (!clmembflag[cl]) continue;
            const d_inter = this.dcc[target][cl];
            if (Math.abs(d - d_inter) > rThreshold) {
              clmembflag[cl] = false;
              pruned.push({ target: cl, reason: '1D-3P' });
              prunedCount++;
            }
          }
          if (prunedCount > 0) {
            pruneCount3P += prunedCount;
            if (isExplain) {
              steps.push({
                type: 'prune',
                title: `Tile ${this.axisName}: 1D-3P Pruning`,
                text: `1D triangle inequality pruned ${prunedCount} candidate(s).`
              });
            }
          }
        }

        this.totalEvals += evals;

        if (assigned === -1) {
          assigned = this.clusters.length;
          const newAnchor = { id: assigned, val, members: 1, prob: 1.0 };
          const newRow = [];
          for (let i = 0; i < this.clusters.length; i++) {
            const d_inter = Math.abs(val - this.clusters[i].val);
            distClusterCluster++;
            distClusterClusterLast++;
            this.dcc[i].push(d_inter);
            newRow.push(d_inter);
          }
          newRow.push(0.0);
          this.dcc.push(newRow);
          this.clusters.push(newAnchor);

          if (isExplain) {
            steps.push({
              type: 'new-cluster',
              title: `Tile ${this.axisName}: New 1D Anchor C${assigned}`,
              text: `Spawned new 1D anchor at coordinate ${val.toFixed(3)}.`
            });
          }
          return { assigned, evals, isNew: true, evaluated, pruned, steps };
        }

        return { assigned, evals, isNew: false, evaluated, pruned, steps };
      }
    }

    // =========================================================================

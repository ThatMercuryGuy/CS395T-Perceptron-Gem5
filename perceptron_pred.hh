#ifndef __CPU_PRED_PERCEPTRON_PRED_HH__
#define __CPU_PRED_PERCEPTRON_PRED_HH__

#include <vector>

#include "base/types.hh"
#include "cpu/pred/conditional.hh"
#include "params/PerceptronBP.hh"

namespace gem5
{

namespace branch_prediction
{

class PerceptronBP : public ConditionalPredictor
{
  private:
    const unsigned numPerceptrons;
    const unsigned historyLength;
    const int threshold;
    const unsigned weightBits;
    
    int maxWeight;
    int minWeight;

    std::vector<std::vector<int>> weightTable;
    std::vector<uint64_t> globalHistory;

    struct PerceptronHistory
    {
        uint64_t historyState;
        int outputY;
        unsigned tableIndex;
    };

  public:
    PerceptronBP(const PerceptronBPParams &params);

    bool lookup(ThreadID tid, Addr pc, void * &bp_history) override;
    void updateHistories(ThreadID tid, Addr pc, bool uncond, bool taken, Addr target, const StaticInstPtr &inst, void * &bp_history) override;
    void update(ThreadID tid, Addr pc, bool taken, void * &bp_history, bool squashed, const StaticInstPtr & inst, Addr target) override;
    void squash(ThreadID tid, void * &bp_history) override;
    void branchPlaceholder(ThreadID tid, Addr pc, bool uncond, void * &bp_history) override;
};

}
}

#endif
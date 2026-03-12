#include "cpu/pred/perceptron_pred.hh"

#include "base/bitfield.hh"
#include "base/intmath.hh"

namespace gem5
{

namespace branch_prediction
{

PerceptronBP::PerceptronBP(const PerceptronBPParams &params)
    : ConditionalPredictor(params),
      numPerceptrons(params.globalPredictorSize),
      historyLength(params.globalHistoryLength),
      threshold(params.threshold),
      weightBits(params.weightWidth)
{
    maxWeight = (1 << (weightBits - 1)) - 1;
    minWeight = -(1 << (weightBits - 1));

    weightTable.resize(numPerceptrons, std::vector<int>(historyLength + 1, 0));
    globalHistory.resize(params.numThreads, 0);
}

bool
PerceptronBP::lookup(ThreadID tid, Addr pc, void * &bp_history)
{
    //we don't really need the tid for anything

    //calculate index to read perceptron weight table 
    unsigned index = (pc >> instShiftAmt) % numPerceptrons;
    int y = weightTable[index][0]; //constant alpha

    //calculate int prediction
    for (int i = 0; i < historyLength; i++)
    {
        if ((globalHistory[tid] >> i) & 1) y += weightTable[index][i + 1];
        else y -= weightTable[index][i + 1];
    }

    PerceptronHistory *history = new PerceptronHistory();
    history->historyState = globalHistory[tid];
    history->outputY = y;
    history->tableIndex = index;
    bp_history = static_cast<void*>(history);

    //according to paper, positive sum is taken and vice versa
    if (y >= 0) return true;
    return false;
}

void
PerceptronBP::branchPlaceholder(ThreadID tid, Addr pc, bool uncond, void * &bp_history)
{
    PerceptronHistory *history = new PerceptronHistory();
    history->historyState = globalHistory[tid];
    history->outputY = 0;
    history->tableIndex = 0;
    bp_history = static_cast<void*>(history);
}

void
PerceptronBP::updateHistories(ThreadID tid, Addr pc, bool uncond, bool taken, Addr target, const StaticInstPtr &inst, void * &bp_history)
{
    if (taken) globalHistory[tid] = (globalHistory[tid] << 1) | 1;
    else globalHistory[tid] = (globalHistory[tid] << 1);
}

void
PerceptronBP::update(ThreadID tid, Addr pc, bool taken, void * &bp_history, bool squashed, const StaticInstPtr & inst, Addr target)
{
    if (bp_history)
    {
        PerceptronHistory *history = static_cast<PerceptronHistory *>(bp_history);

        if (!squashed && ((history->outputY >= 0) != taken || abs(history->outputY) <= threshold))
        {
            int t = taken ? 1 : -1;
            unsigned idx = history->tableIndex;

            weightTable[idx][0] += t;
            if (weightTable[idx][0] > maxWeight) weightTable[idx][0] = maxWeight;
            if (weightTable[idx][0] < minWeight) weightTable[idx][0] = minWeight;

            for (int i = 0; i < historyLength; i++)
            {
                int x_i = ((history->historyState >> i) & 1) ? 1 : -1;
                
                if (t == x_i) weightTable[idx][i + 1]++;
                else weightTable[idx][i + 1]--;

                if (weightTable[idx][i + 1] > maxWeight) weightTable[idx][i + 1] = maxWeight;
                if (weightTable[idx][i + 1] < minWeight) weightTable[idx][i + 1] = minWeight;
            }
        }

        delete history;
        bp_history = nullptr;
    }
}

void
PerceptronBP::squash(ThreadID tid, void * &bp_history)
{
    if (bp_history)
    {
        PerceptronHistory *history = static_cast<PerceptronHistory*>(bp_history);
        globalHistory[tid] = history->historyState;
        
        delete history;
        bp_history = nullptr;
    }
}

}
}
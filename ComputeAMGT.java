// Associative memory to provide medium term memory.
// Reads input address from one gathered source and scatters the result.
// Reads input from another source as an address and stores a further source.
package data.reservoir.compute.ai;

public final class ComputeAMGT extends Compute {
    
    private final AMGT memory;
    
    ComputeAMGT(Reservoir r, int density,float threshold) {
        super(r);
        memory = new AMGT(r.getComputeSize(), density,threshold);
    }
    
    @Override
    public void compute() {
        float[] workA = reservoir.getComputeBuffer(0);
        float[] workB = reservoir.getComputeBuffer(1);
        float[] workC = reservoir.getComputeBuffer(2);
        reservoir.gather(workA);
        reservoir.gather(workB);
        reservoir.gather(workC);
        memory.recallVec(workA, workA);
        reservoir.scatterGeneral(workA);
        memory.trainVec(workC, workB);
    }
    
    @Override
    public int buffersRequired() {
        return 3;
    }
    
    @Override
    public void resetHeldState() {
        memory.reset();
    }
    
    @Override
    public int nGather() {
        return 3;
    }
    
    @Override
    public int nScatterGeneral() {
        return 1;
    }
    
    @Override
    public int nCompute() {
        return 0;
    }
    
}

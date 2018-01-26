// Rescale the input section of the reservoir to a standard vector length
package data.reservoir.compute.ai;

public final class ComputeNormalizeInput extends Compute {

    public ComputeNormalizeInput(Reservoir r) {
        super(r);
    }

    @Override
    public void compute() {
        reservoir.normalizeInput();
    }

    @Override
    public int nGather() {
        return 0;
    }

    @Override
    public int nScatterGeneral() {
        return 0;
    }

    @Override
    public int nCompute() {
        return 0;
    }

    @Override
    public int buffersRequired() {
        return 0;
    }
  
}

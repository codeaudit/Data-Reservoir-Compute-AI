// Rescale the input section of the reservoir to a standard vector length
package data.reservoir.compute.ai;

public final class ComputeBiasWrite extends Compute {

    private final int writeLocation;

    public ComputeBiasWrite(Reservoir r, int writeLocation) {
        super(r);
        this.writeLocation = writeLocation;
    }

    @Override
    public void compute() {
        float[] workA = reservoir.getComputeBuffer(0);
        int n = workA.length;
        float sc = 2f / n;
        for (int i = 0; i < n; i++) {
            workA[i] = sc * i - 1f + 0.5f * sc;
        }
        reservoir.scatterWrite(workA, writeLocation);
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
        return 1;
    }

}

// Put some random numbers into the write section of the data reservoir.
// It can make evolution problematic but can also allow the network to
// discover randomized algorithms.  Or that is the idea anyway.
package data.reservoir.compute.ai;

public final class ComputeRndWrite extends Compute {

    private final RNG rng;
    private final float scale;
    private final int writeLocation;

    public ComputeRndWrite(Reservoir r, float scale, int writeLocation) {
        super(r);
        rng = new RNG();
        this.scale = scale;
        this.writeLocation = writeLocation;
    }

    @Override
    public void compute() {
        float[] workA = reservoir.getComputeBuffer(0);
        for (int i = 0; i < workA.length; i++) {
            workA[i] = scale * rng.nextFloatSym();
        }
        reservoir.scatterWrite(workA, writeLocation);
    }

    @Override
    public int buffersRequired() {
        return 1;
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

}

// Calculates a single neural network layer with x signed square root activation function.
// Uses a fast approximation
package data.reservoir.compute.ai;

public final class ComputeLayerSqRt extends Compute {

    private final int density;

    ComputeLayerSqRt(Reservoir r, int density) {
        super(r);
        this.density = density;
    }

    @Override
    public void compute() {
        float[] workA = reservoir.getComputeBuffer(0);
        float[] workB = reservoir.getComputeBuffer(1);
        reservoir.gather(workA);
        VecOps.adjust(workA, workA);
        reservoir.multiplyWithWeights(workB, workA);
        for (int i = 1; i < density; i++) {
            reservoir.randomProjection(workA);
            reservoir.multiplyWithWeightsAddTo(workB, workA);
        }
        VecOps.scale(workB, workB, 1f / (float) Math.sqrt(density));
        VecOps.signedSqRt(workB, workB);
        reservoir.scatterGeneral(workB);
    }

    @Override
    public int buffersRequired() {
        return 2;
    }

    @Override
    public int nGather() {
        return 1;
    }

    @Override
    public int nScatterGeneral() {
        return 1;
    }

    @Override
    public int nCompute() {
        return density;
    }

}

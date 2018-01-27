// Calculates a single neural network layer with square root activation function.
// High density version for where density is greater than the number of 
// dimensions (computeSize.)
package data.reservoir.compute.ai;

import java.util.Arrays;

public final class ComputeLayerSqRtHD extends Compute {

    private final int density;

    ComputeLayerSqRtHD(Reservoir r, int density) {
        super(r);
        this.density = density;
    }

    @Override
    public void compute() {
        float[] workA = reservoir.getComputeBuffer(0);
        float[] workB = reservoir.getComputeBuffer(1);
        float[] workC = reservoir.getComputeBuffer(2);
        reservoir.gather(workA);
        VecOps.adjust(workA, workA);
        Arrays.fill(workC, 0f);
        for (int i = 0; i < density; i++) {
            reservoir.randomProjection(workA);
            VecOps.signedSqRt(workB, workA);  // approximate signed square root
            reservoir.multiplyWithWeightsAddTo(workC, workB);
        }
        VecOps.scale(workC, workC, 1f / (float) Math.sqrt(density));
        reservoir.scatterGeneral(workC);
    }

    @Override
    public int buffersRequired() {
        return 3;
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

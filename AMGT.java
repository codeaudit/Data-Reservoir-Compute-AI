/* Associative memory threshold gated class
   Self gating. Only stores information over a threshold magnitude.
   And then minus that magnitude.
   Should allow long term memory.
 */
package data.reservoir.compute.ai;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.Arrays;

public final class AMGT implements Serializable {

    private final int vecLen;
    private final int density;
    private final float threshold;
    private final long hash;
    private final float[][] weights;
    private transient float[][] bipolar;
    private final float[] workA;
    private final float[] workB;

    // vecLen must be 2,4,8,16,32.....
    // density is the maximum number of vector pairs that can be associated with
    // repeated training.
    public AMGT(int vecLen, int density, float threshold) {
        this.vecLen = vecLen;
        this.density = density;
        this.threshold = threshold;
        hash = System.nanoTime();
        weights = new float[density][vecLen];
        bipolar = new float[density][vecLen];
        workA = new float[vecLen];
        workB = new float[vecLen];
    }

    public void recallVec(float[] resultVec, float[] inVec) {
        System.arraycopy(inVec, 0, workA, 0, vecLen);
        Arrays.fill(resultVec, 0f);
        for (int i = 0; i < density; i++) {
            WHT.fastRP(workA, hash + i);
            WHT.signOf(bipolar[i], workA);
            VecOps.multiplyAddTo(resultVec, weights[i], bipolar[i]);
        }
    }

    public void trainVec(float[] targetVec, float[] inVec) {
        float rate = 1f / density;
        recallVec(workB, inVec);
        VecOps.truncate(workA, targetVec, threshold);  // lob off the threshold magnitude.
        for (int i = 0; i < vecLen; i++) {
            workB[i] = (workA[i] - workB[i]) * rate;   //get the error term in workB
        }
        for (int i = 0; i < density; i++) {            // correct the weights 
            for (int j = 0; j < vecLen; j++) {
                if (workA[j] != 0f) {   // if not gated out by truncation update the weight
                    weights[i][j] += workB[j] * bipolar[i][j];
                }
            }
        }
    }

    public void reset() {
        for (float[] x : weights) {
            Arrays.fill(x, 0f);
        }
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        bipolar = new float[density][vecLen];
    }
}

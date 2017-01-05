package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.regression.regression_tree.Node;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jinghanyang on 12/26/16.
 */
public class TreeInfo {
    public Vector vector;
    public RegressionTree regressionTree;
    public List<Integer> wordVecIds;
    public List<Integer> dummyIds;
    public int leafId;
    public List<Integer> dummyFeatureIndexes;

    public TreeInfo(RegressionTree regressionTree, Vector datapoint) {
        this.vector = datapoint;
        this.regressionTree = regressionTree;
        this.wordVecIds = new ArrayList<Integer>();
        this.dummyIds = new ArrayList<Integer>();
        this.dummyFeatureIndexes = new ArrayList<Integer>();
        int finalId = nodeCheck(regressionTree.getRoot());
        this.leafId = finalId;
    }
    public int nodeCheck(Node node){
        int id = node.id;
        double threshold = node.getThreshold();
        int featureIndex = node.getFeatureIndex();
        if(featureIndex<300){
            this.wordVecIds.add(id);
        }else {
            this.dummyIds.add(id);
            this.dummyFeatureIndexes.add(featureIndex);
        }
        if (node.leaf) {
            return node.id;
        }else {
            if(this.vector.get(featureIndex)<=threshold){
                return nodeCheck(node.getLeftChild());
            }else {
                return nodeCheck(node.getRightChild());
            }
        }
    }
//
//    main() {
//        TreeInfo obj = new TreeInfo();

//    }
}

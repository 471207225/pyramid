package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.dataset.SparseDataSet;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.GradientValueOptimizer;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoostOptimizer;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import edu.neu.ccs.pyramid.regression.linear_regression.SquareLoss;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * Created by chengli on 11/20/16.
 */
public class JinghanTry3 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");

        }
        Config config = new Config(args[0]);
        System.out.println(config);
        train(config);
    }

    public static void train(Config config) throws Exception{

        DataSet train_docstoword = loadword2vecMatrix(config.getString("input.word2vec"));
//        DataSet test_docstoword = loadDocMatrix(config.getString("input.testData"), config);

        double [] train_labels = loadlabels(config.getString("input.trainLabel"), config);


        String excellent = "0.0321825 0.0539471 0.0992927 -0.0832675 -0.000512374 0.0674834 0.0246569 0.0294338 0.091537 -0.0634545 -0.050433 0.0213652 0.0398635 -0.00501282 -0.0104254 0.0124003 -0.0131236 0.0354248 -0.00844028 0.046553 -0.0468409 0.0106379 0.0625142 0.0297577 0.0924253 -0.063427 0.00392243 -0.0268176 -0.0573974 0.00552436 -0.05915 0.0539379 -0.0405931 -0.0762427 0.112903 0.064104 -0.035022 0.0499215 -0.0440251 -0.00416941 0.0939134 0.0399762 0.0237441 -0.0146216 0.0587286 0.0286235 -0.0368914 0.0586366 -0.0693823 -0.0735858 -0.0798048 0.0440956 -0.0581654 0.120932 0.0696294 -0.0215987 0.0325409 0.0753451 -0.110362 -0.0183705 0.0234748 -0.0170097 0.0443503 -0.116759 0.0751401 0.0138974 -0.10477 0.0800403 0.0466164 0.0530043 -0.030835 -0.00701226 -0.0505618 -0.0338349 0.0382627 -0.0102276 -0.0657329 0.0760658 0.0445084 -0.0366826 -0.058441 0.0487677 0.0255777 -0.0300729 -1.65222e-05 0.0272723 0.0546337 0.030568 -0.0730008 -0.0146992 -0.038227 -0.125683 -0.00809925 -0.0281269 -0.0300183 -0.0054489 -0.00445629 0.0150926 0.00595796 -0.0449214 0.112337 0.0842202 0.134064 -0.0570071 -0.0542159 0.0345854 -0.112499 0.0567177 -0.0538593 -0.00361067 0.0158978 -0.106642 -0.0935506 0.0176671 -0.165172 0.0348708 -0.0353731 0.0209057 0.0141422 0.0731311 0.0167408 0.0181434 0.0647786 0.0856397 -0.0778322 0.0666069 -0.0492442 0.012557 0.00774029 0.0110807 0.074418 0.0552813 -0.00300535 -0.0483167 0.00211262 0.0527255 0.100803 -0.0250688 0.0062285 -0.0687703 -0.0621329 0.0441249 -0.0638663 -0.0478753 0.00565079 0.131931 0.0693929 0.0393221 0.0008917 0.0622015 -0.00671735 0.0348428 -0.0294625 0.0109483 -0.000621195 -0.0302992 -0.0689784 -0.111338 0.121243 0.0252578 -0.0171483 0.00491344 -0.0197123 0.019865 -0.00434499 0.0702032 0.0566522 -0.0542501 0.0118529 0.0154858 0.0276096 0.0332827 -0.0122224 -0.150395 0.0913295 -0.0252088 -0.0520647 0.0135917 -0.0471353 -0.0458645 0.0539141 -0.00341873 -0.0477719 -0.0276479 0.0919565 0.0818114 0.1287 0.00850494 -0.00425617 -0.0711885 0.0710582 0.141059 -0.132379 0.032149 0.014088 -0.00791652 0.0591335 0.0477362 0.0257646 0.0690699 -0.0197989 -0.0689657 0.061973 -0.0333673 -0.0368927 -0.0368221 -0.0587932 -0.0139462 -0.0477406 -0.060535 0.0649116 0.0501605 0.00594743 -0.139395 -0.00161513 0.0396209 -0.00304721 0.0232281 -0.0673759 -0.0442746 -0.0039196 -0.0243623 0.0108633 0.0626738 0.00521995 0.0184113 -0.0768095 0.0914265 0.0380155 0.0250778 0.0421014 0.0425228 -0.00496919 0.0170051 -0.0248989 0.0764546 0.0308852 -0.07632 -0.0715343 0.139169 0.0744283 -0.0221138 0.000115149 0.0289811 -0.0669768 0.00100503 -0.0236128 -0.00255442 0.0181001 0.0300676 -0.000709654 -0.0365018 0.00732605 -0.0830522 -0.0434454 -0.0717019 -0.00267346 0.0529516 0.0410755 -0.0157103 0.0892408 -0.04442 -0.1015 0.00301575 0.00563756 0.112663 0.0600498 -0.0777162 0.0288293 -0.0324092 0.0393601 -0.00843264 0.00390135 0.0681497 0.0400292 0.0204291 0.00772007 0.0197063 0.028401 0.0473645 0.0422868 -0.0628948 -0.0996632 0.0235304 0.0477761 0.0649427 0.0371432 -0.201946 0.00699931 -0.00450987 -0.00579748 0.0108124 -0.00799496 -0.00334801 0.0924824 -0.021592 0.0302682 0.0721657 -0.115423 0.0496256 ";
        String terrible = "0.00238741 -0.0777599 0.222141 -0.0154371 0.0771727 0.032222 0.0880818 0.0136024 -0.0224897 0.0020099 -0.023885 0.0586544 0.0597272 -0.0778771 0.139016 -0.0105515 0.0420436 0.0244911 -0.00140189 0.114481 -0.0458677 0.0777632 -0.0728873 0.0259644 0.063984 -0.175236 -0.00794863 -0.0845098 -0.0138979 -0.0115022 -0.0466959 -0.0722631 -0.0418846 -0.063272 0.0188792 -0.00500983 0.0723643 -0.00880037 -0.0189719 -0.0958869 0.158138 -0.0743425 -0.00146478 -0.0315363 -0.0618628 0.0302134 -0.0574783 -0.0130158 0.0206121 -0.0458012 -0.109382 0.0264561 0.061575 -0.112281 0.0237843 -0.0332714 0.0349422 -0.0172838 -0.0693578 -0.0147753 0.0384257 -0.0286352 -0.0902214 -0.0359291 0.0540313 0.0415376 0.00354583 0.0682583 0.0199456 0.0746569 -0.0115069 0.0145684 -0.0523399 -0.0894753 0.0141087 -0.017832 -0.0566444 -0.0204111 0.0557908 -0.0874708 -0.0173642 -0.0484887 0.0627994 -0.0323284 0.113299 -0.0113173 0.0277486 0.0724752 -0.0581123 -0.0156623 0.0169152 -0.103703 -0.12464 -0.011543 0.037368 0.0187192 -0.0430093 0.0497969 0.0764484 0.0281275 0.068157 0.153125 0.0209015 0.000921787 -0.00348939 0.0079308 -0.0246638 0.0716233 0.0409894 0.0165788 -0.0252504 0.0541375 -0.0357581 -0.0350453 -0.121434 0.019994 -0.00375951 -0.0165565 0.0595255 0.0527371 -0.108987 0.065449 0.0691634 -0.0215072 -0.000988503 -0.00432013 -0.0436134 -0.014321 0.053769 0.0201755 -0.0182435 -0.00249897 0.0151342 0.0334732 0.0767691 -0.029875 0.0592648 0.0682063 0.00456453 -0.0266632 -0.0435917 0.0923425 -0.0167688 -0.0108612 0.0248131 0.00205507 -0.0900548 -0.0264521 0.0192797 0.00720387 -0.0217191 0.0538613 0.0403829 0.0517373 -0.0212922 -0.0029675 -0.0412405 -0.0728205 -0.0210561 -0.0156279 -0.0721686 0.00900264 -0.0136358 -0.0395352 0.00592356 0.0470412 -0.0190803 -0.0351851 0.0335366 0.036239 -0.0425058 0.0108722 -0.0428273 -0.0851411 0.00107943 0.00473979 0.0619823 -0.0268649 -0.0412148 -0.0163642 -0.00250039 -0.00311098 -0.0184059 -0.0125345 0.100822 -0.0438575 0.0744207 -0.0266469 0.0781005 -0.0054638 0.0307539 0.0505167 -0.157285 0.106691 0.0889329 -0.0521235 0.00139183 -0.00094314 -0.0603271 -0.0493292 -0.0165473 -0.0450314 0.0294959 -0.0137492 -0.10731 0.0505421 -0.0716076 -0.0273773 0.0366976 -0.127354 0.136879 0.115819 0.0135489 -0.0283146 0.0307816 0.0292061 0.0513612 -0.080907 -0.0427027 0.0218366 -0.00478522 0.0309952 -0.00418625 0.0541616 -0.0685329 0.0947767 -0.0815277 0.0212201 -0.0169686 0.0137298 0.106214 -0.00754339 -0.080587 0.109351 -0.107719 -0.0289385 -0.0517937 -0.0686755 -0.00945573 0.136505 -0.000420062 -0.0615897 0.0191247 -0.0493136 -0.00380854 -0.0583585 -0.0231735 -0.00711277 -0.031853 0.0327058 -0.00402268 -0.0265844 -0.00628709 -0.0350074 -0.119991 -0.0348273 -0.0222141 0.113791 -0.0167376 0.0559408 -0.0368074 0.0246099 -0.00636458 0.021051 0.0389704 -0.0477464 0.0860059 -0.0443413 0.0353228 -0.0949731 -0.00306644 0.0295212 0.0373198 0.0320189 -0.0208702 0.0290283 -0.00983686 -0.00891026 0.0632437 -0.00338114 0.0768 0.031285 -0.0795047 -0.017637 -0.0142972 -0.0844337 -0.0548016 -0.105268 -0.0121448 -0.0175369 -0.000198181 -0.0695205 -0.0895038 -0.0467935 -0.0358396 -0.014209 0.0399594 -0.0893139 -0.0893562 0.0643794 ";
        double[] exArr = Arrays.stream(excellent.split(" ")).mapToDouble(s->Double.parseDouble(s)).toArray();
        double[] teArr = Arrays.stream(terrible.split(" ")).mapToDouble(s->Double.parseDouble(s)).toArray();
        Vector exVector = new DenseVector(exArr);
        Vector teVector = new DenseVector(teArr);
        LSBoost lsBoost = new LSBoost();

        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(config.getInt("train.numLeaves"));
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSBoostOptimizer optimizer = new LSBoostOptimizer(lsBoost, train_docstoword, regTreeFactory, train_labels);
        optimizer.setShrinkage(0.1);
        optimizer.initialize();

        for (int iter=0;iter<config.getInt("iterations");iter++){
            optimizer.iterate();
            System.out.println("iteration "+iter);
            System.out.println("training RMSE = "+ RMSE.rmse(train_labels, lsBoost.predict(train_docstoword)));
//            System.out.println("test RMSE = "+ RMSE.rmse(test_labels, lsBoost.predict(test_docstoword)));
            System.out.println("excellent"+lsBoost.predict(exVector));
            System.out.println("terrible"+lsBoost.predict(teVector));
        }



    }


    public static DataSet loadword2vecMatrix(String path) throws Exception{
        DataSet denseDataSet = DataSetBuilder.getBuilder().numDataPoints(4998).numFeatures(300).build();
        try(BufferedReader br = new BufferedReader(new FileReader(path))){
            String line = null;
            int dataIndex = 0;
            while ((line = br.readLine()) !=null){
                String [] lineEle = line.split(" ");
                for (int j=0; j<denseDataSet.getNumFeatures();j++){
                    denseDataSet.setFeatureValue(dataIndex,j,Double.parseDouble(lineEle[j]));
                }
                dataIndex += 1;
            }
        }

        return denseDataSet;
    }

//    public static DataSet loadDocMatrix(String path, Config config) throws Exception{
//        int numData = (int) Files.lines(Paths.get(path)).count();
//        int numFeatures = Files.lines(Paths.get(path)).findFirst().get().split(" ").length;
//        DataSet dataSet = new SparseDataSet(numData, numFeatures, false);
//        try (BufferedReader br = new BufferedReader(new FileReader(path));
//        ) {
//            String line = null;
//            int dataIndex = 0;
//            while ((line = br.readLine()) != null) {
//
//                if(dataIndex == 0){
//                    dataIndex += 1;
//                    String[] featureNames = line.split(" ");
//                    FeatureList featureList = new FeatureList();
//                    for (String featureName: featureNames){
//                        Feature feature = new Feature();
//                        feature.setName(featureName);
//                        featureList.add(feature);
//                    }
//                    dataSet.setFeatureList(featureList);
//                    continue;
//                }
//                String [] lineEle = line.split(" ");
//                for (int j=0;j<dataSet.getNumFeatures();j++){
//                    dataSet.setFeatureValue(dataIndex-1, j, Double.valueOf(lineEle[j]));
//                }
//                dataIndex += 1;
//            }
//        }
//        return dataSet;
//    }
    public static double[] loadlabels(String Path, Config config) throws Exception{
        double [] labels = new double[4998];
        try(BufferedReader br = new BufferedReader(new FileReader(Path))){
            String line = null;
            int lineIndex = 0;
            while ((line = br.readLine()) != null) {

                labels[lineIndex] = Double.parseDouble(line);
                lineIndex += 1;
            }
        }
        return labels;
    }
}

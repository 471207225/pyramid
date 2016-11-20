package edu.neu.ccs.pyramid.regression.linear_regression;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.optimization.Optimizer;
import junit.framework.TestCase;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

import java.io.File;

/**
 * Created by chengli on 11/20/16.
 */
public class SquareLossTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration config = ctx.getConfiguration();
        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.OFF);
        ctx.updateLoggers();
        test1();
    }

    private static void test1() throws Exception{
        RegDataSet dataSet = TRECFormat.loadRegDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.REG_DENSE, true);
        RegDataSet testSet = TRECFormat.loadRegDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.REG_DENSE, true);

        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures());
        SquareLoss squareLoss = new SquareLoss(linearRegression, dataSet, dataSet.getLabels(), 1);
        System.out.println(squareLoss.getGradient());
        LBFGS optimizer = new LBFGS(squareLoss);
//        GradientDescent gradientDescent = new GradientDescent(squareLoss);
        for (int iter=0;iter<100;iter++){
            optimizer.iterate();
            System.out.println("training RMSE = "+ RMSE.rmse(linearRegression, dataSet));
            System.out.println("test RMSE = "+ RMSE.rmse(linearRegression, testSet));
        }
    }

}
package edu.neu.ccs.pyramid.application;
import edu.neu.ccs.pyramid.jinghan.ArrayIndexComparator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.lang.*;
import java.util.Comparator;

/**
 * Created by jinghanyang on 12/26/16.
 */
public class jinghan_ArrayDemo {
    public static void main(String[] args) {

        // initializing unsorted int array

        String excellent = "0.0321825 0.0539471 0.0992927 -0.0832675 -0.000512374";
        double[] exArr = Arrays.stream(excellent.split(" ")).mapToDouble(s->Double.parseDouble(s)).toArray();
        Vector exVector = new DenseVector(exArr);
        System.out.println(exVector.get(0));

        int iArr[] = {2, 1, -9, 6, 4};
        Integer[] array = {9,-2,10,3,-5,34,-22,7};
        double[] doubles = new double[]{3.1,0,-0.1,6.1 ,2.2, 4.1, -5.1};
//        String[] out = new String[n];
//        for(int i = 0; i < n; i++) {
//            out[i] = in.get(i);
//        }

        ArrayIndexComparator comparator = new ArrayIndexComparator(doubles);
        Integer[] indexes = comparator.createIndexArray();
        Arrays.sort(indexes, comparator);
        System.out.println("sorted index is " + Arrays.toString(indexes));
        System.out.println("sorted element is ");
        for(int ind : indexes){
            System.out.println(doubles[ind]);
        }

        // try to return index
//        String[] newArray = {"Zrance", "Arance", "France", "Italy"};
//        ArrayIndexComparator comparator = new ArrayIndexComparator(newArray);
//        Integer[] indexes = comparator.createIndexArray();
//        Arrays.sort(indexes, comparator);
//        for(int ind : indexes){
//            System.out.println("index is " + ind);
//        }



//        Arrays.sort(myDoubleArray, Comparator.comparingDouble());

        Arrays.sort(array, Comparator.comparingInt(Math::abs));
        System.out.println(Arrays.toString(array));

//        Arrays.sort(myDoubleArray, (a, b)->(Double.compare(Math.abs(a),Math.abs(b))));
        // let us print all the elements available in list
        for (int number : iArr) {
            System.out.println("Number = " + number);
            System.out.println(Math.abs(number));
        }

        // sorting array
        Arrays.sort(iArr);

        // let us print all the elements available in list
        System.out.println("The sorted int array is:");
        for (int number : iArr) {
            System.out.println("Number = " + number);
        }



        // double
        double[] myDoubleArray = new double[]{1.2,-1.3,1.4,-2.5};
    }
}

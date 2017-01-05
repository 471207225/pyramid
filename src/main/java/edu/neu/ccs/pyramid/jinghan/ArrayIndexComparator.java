package edu.neu.ccs.pyramid.jinghan;

import java.util.Comparator;

/**
 * Created by jinghanyang on 12/26/16.
 */
public class ArrayIndexComparator implements Comparator<Integer>{
    public double[] doubleArray;
    public String[] array;

    public ArrayIndexComparator(double[] array){
        this.doubleArray = array;
    }

    public Integer[] createIndexArray()
    {
        Integer[] indexes = new Integer[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++)
        {
            indexes[i] = i; // Autoboxing
        }
        return indexes;
    }

    @Override
    public int compare(Integer index1, Integer index2)
    {
        return Double.compare(Math.abs(doubleArray[index2]), Math.abs(doubleArray[index1]));
    }
}

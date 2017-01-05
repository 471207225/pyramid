package edu.neu.ccs.pyramid.jinghan;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by jinghanyang on 12/26/16.
 */
public class LeafNodeCount {
    public Map<Integer, List<Integer>> map;
    public LeafNodeCount(List<Integer> list) {
        this.map = new HashMap<Integer, List<Integer>>();
        for(int i=0; i<list.size(); i++){
            if(!map.containsKey(list.get(i))){
                List<Integer> leafIDs = new ArrayList<Integer>();
                leafIDs.add(i);
                map.put(list.get(i),leafIDs);
            }else{
                map.get(list.get(i)).add(i);
            }

        }


//        Map<String, String> map = new HashMap<String, String>();
//        map.put("dog", "type of animal");
//        System.out.println(map.get("dog"));

    }
}

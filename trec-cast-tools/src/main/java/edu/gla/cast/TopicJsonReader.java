package edu.gla.cast;

import cast.topics.TopicDef.Topic;
import com.google.protobuf.util.JsonFormat;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Read in a topic JSON file and create a protocol buffer topic representation.
 *
 */
public class TopicJsonReader {


  /**
   * Parses a topic JSON file and produces a list of Topic objects.
   */
  public List<Topic> readJsonTopics(String topicFile) throws Exception {
    FileInputStream fileInputStream = new FileInputStream(topicFile);
    List<Topic> topicList = new ArrayList();

    try {
      Reader reader = new InputStreamReader(fileInputStream);
      JSONParser parser = new JSONParser();
      JSONArray array = (JSONArray) parser.parse(reader);
      System.out.println(array);

      JsonFormat.Parser formatParser = JsonFormat.parser();
      Iterator<JSONObject> iterator = array.iterator();
      while (iterator.hasNext()) {
        // This is a bit annoying -- is there a better way of going
        // from jsonobject to proto except via string?
        String jsonString = iterator.next().toJSONString();
        Topic.Builder builder = Topic.newBuilder();
        formatParser.merge(jsonString, builder);
        topicList.add(builder.build());
      }
    } finally {
      fileInputStream.close();
    }
    return topicList;
  }


  public static void main(String[] args) throws Exception{
    System.out.println("Loading topics.");
    TopicJsonReader topicTextToProto = new TopicJsonReader();
    List<Topic> topicList = topicTextToProto.readJsonTopics(args[0]);
    System.out.println("Number of topics:" + topicList.size());
    for (Topic topic : topicList) {
      System.out.println(topic.toString());
    }
  }
}
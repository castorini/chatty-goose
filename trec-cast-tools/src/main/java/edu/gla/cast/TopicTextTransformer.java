package edu.gla.cast;

import java.io.FileOutputStream;
import java.io.FileWriter;
import java.util.List;

import cast.topics.TopicDef.*;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import com.google.protobuf.util.JsonFormat;

/**
 * Read in a topic text file and create a standard protocol buffer
 * and json file outputs.
 *
 */
public class TopicTextTransformer {

  /**
   * Write the topic files to a pretty printed JSON file.
   * The individual topics are wrapped in an outer array.
   *
   * @param topics
   * @param outputFile
   * @throws Exception
   */
  public void writeTopicToJsonFile(List<Topic> topics, String outputFile) throws Exception {
    FileWriter writer = new FileWriter(outputFile);
    try {
      JSONArray list = new JSONArray();
      JSONParser parser = new JSONParser();
      for (Topic topic : topics) {
        String jsonString = JsonFormat.printer()
                .preservingProtoFieldNames()
                .print(topic);

        JSONObject object = (JSONObject) parser.parse(jsonString);
        list.add(object);
      }
      Gson gson = new GsonBuilder().setPrettyPrinting().create();
      JsonParser jp = new JsonParser();
      JsonElement je = jp.parse(list.toJSONString());
      String prettyJsonString = gson.toJson(je);

      writer.write(prettyJsonString);
    } finally {
      writer.close();
    }
  }

  /**
   * Write the topics to a binary file that is delimited.
   *
   * @param topics
   * @param outputFile
   * @throws Exception
   */
  public void writeTopicToProtoFile(List<Topic> topics, String outputFile) throws Exception {
    FileOutputStream outputStream = new FileOutputStream(outputFile);
    try {
      for (Topic topic : topics) {
        topic.writeDelimitedTo(outputStream);
      }
    } finally {
      outputStream.close();
    }
  }

  public static void main(String[] args) throws Exception{
    System.out.println("Loading topics.");
    TopicTextReader reader = new TopicTextReader();
    List<Topic> topicList = reader.parseTopicTextFile(args[0]);
    System.out.println("Number of topics:" + topicList.size());
    TopicTextTransformer transformer = new TopicTextTransformer();
    transformer.writeTopicToJsonFile(topicList, args[1]);
    transformer.writeTopicToProtoFile(topicList, args[2]);

  }
}
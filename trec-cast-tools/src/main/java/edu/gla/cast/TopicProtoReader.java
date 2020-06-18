package edu.gla.cast;

import cast.topics.TopicDef.Topic;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Read in a Topic file encoded in binary protocol buffer format.
 *
 */
public class TopicProtoReader {


  /**
   * Parses a topic protocol buffer file and produces a list of Topic objects.
   */
  public List<Topic> readTopics(String topicFile) throws Exception {
    FileInputStream fileInputStream = new FileInputStream(topicFile);
    List<Topic> topicList = new ArrayList();
    try {
      while(true) {
        Topic topic = Topic.parseDelimitedFrom(fileInputStream);
        if (topic == null) {
          break;
        } else {
          topicList.add(topic);
        }
      }
    } finally {
      fileInputStream.close();
    }
    return topicList;
  }


  public static void main(String[] args) throws Exception{
    System.out.println("Loading topics.");
    TopicProtoReader topicTextToProto = new TopicProtoReader();
    List<Topic> topicList = topicTextToProto.readTopics(args[0]);

    // Simply print out the topics loaded.
    System.out.println("Number of topics:" + topicList.size());
    for (Topic topic : topicList) {
      System.out.println(topic.toString());
    }
  }
}
import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  final String baseUrl = 'http://localhost:5000';

  Future<String> getTopicResponse(String topic) async {
    final response = await http.post(
      Uri.parse('$baseUrl/get_topic_response'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'topic': topic}),
    );
    if (response.statusCode == 200) {
      return jsonDecode(response.body)['response'];
    } else {
      throw Exception('Failed to get topic response: ${response.body}');
    }
  }

  Future<String> getAiResponse(String prompt, String topic) async {
    final response = await http.post(
      Uri.parse('$baseUrl/get_ai_response'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'prompt': prompt, 'topic': topic}),
    );
    if (response.statusCode == 200) {
      return jsonDecode(response.body)['response'];
    } else {
      throw Exception('Failed to get AI response: ${response.body}');
    }
  }
}

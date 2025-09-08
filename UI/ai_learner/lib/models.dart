class ChatMessage {
  final String content;
  final bool isUser;

  ChatMessage({required this.content, required this.isUser});

  Map<String, dynamic> toJson() {
    return {'content': content, 'isUser': isUser};
  }

  factory ChatMessage.fromJson(Map<String, dynamic> json) {
    return ChatMessage(content: json['content'], isUser: json['isUser']);
  }
}

class ChatHistory {
  final String id;
  final String topic;
  List<ChatMessage> messages;
  final DateTime timestamp;

  ChatHistory({
    required this.id,
    required this.topic,
    required this.messages,
    required this.timestamp,
  });

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'topic': topic,
      'messages': messages.map((m) => m.toJson()).toList(),
      'timestamp': timestamp.toIso8601String(),
    };
  }

  factory ChatHistory.fromJson(Map<String, dynamic> json) {
    return ChatHistory(
      id: json['id'],
      topic: json['topic'],
      messages: (json['messages'] as List)
          .map((m) => ChatMessage.fromJson(m))
          .toList(),
      timestamp: DateTime.parse(json['timestamp']),
    );
  }
}

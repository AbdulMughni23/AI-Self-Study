import 'package:flutter/material.dart';
import 'package:flutter_html/flutter_html.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import 'api_service.dart';
import 'models.dart';
import 'chat_messages.dart';

void main() {
  runApp(RAGLearningApp());
}

class RAGLearningApp extends StatelessWidget {
  const RAGLearningApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RAG Learning System',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        scaffoldBackgroundColor: Colors.grey[100],
        fontFamily: 'Roboto',
      ),
      home: ChatScreen(),
    );
  }
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final ApiService _apiService = ApiService();
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  List<ChatMessage> _messages = [];
  List<ChatHistory> _chatHistories = [];
  String? _selectedTopic;
  bool _isLoading = false;
  String? _errorMessage;
  String? _currentChatId;

  final List<String> _topics = [
    'Motion',
    'Energy',
    'Momentum',
  ]; // Hardcoded topics

  @override
  void initState() {
    super.initState();
    _loadChatHistories();
  }

  Future<void> _loadChatHistories() async {
    final prefs = await SharedPreferences.getInstance();
    final historiesJson = prefs.getString('chat_histories');
    if (historiesJson != null) {
      final List<dynamic> decoded = jsonDecode(historiesJson);
      setState(() {
        _chatHistories = decoded
            .map((json) => ChatHistory.fromJson(json))
            .toList();
      });
    }
  }

  Future<void> _saveChatHistories() async {
    final prefs = await SharedPreferences.getInstance();
    final historiesJson = jsonEncode(
      _chatHistories.map((h) => h.toJson()).toList(),
    );
    await prefs.setString('chat_histories', historiesJson);
  }

  void _startNewChat(String topic) {
    final newChat = ChatHistory(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      topic: topic,
      messages: [],
      timestamp: DateTime.now(),
    );
    setState(() {
      _chatHistories.add(newChat);
      _currentChatId = newChat.id;
      _messages = [];
      _selectedTopic = topic;
      _errorMessage = null;
    });
    _saveChatHistories();
    _getInitialResponse(topic);
  }

  void _loadChat(String chatId) {
    final chat = _chatHistories.firstWhere((h) => h.id == chatId);
    setState(() {
      _currentChatId = chatId;
      _messages = chat.messages;
      _selectedTopic = chat.topic;
      _errorMessage = null;
    });
    _scrollToBottom();
  }

  Future<void> _getInitialResponse(String topic) async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });
    try {
      final response = await _apiService.getTopicResponse(topic);
      _addMessage(ChatMessage(content: response, isUser: false));
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to connect to backend: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
      _scrollToBottom();
    }
  }

  Future<void> _sendMessage() async {
    if (_controller.text.isEmpty || _selectedTopic == null) return;

    final userMessage = _controller.text;
    _addMessage(ChatMessage(content: userMessage, isUser: true));
    _controller.clear();

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final response = await _apiService.getAiResponse(
        userMessage,
        _selectedTopic!,
      );
      _addMessage(ChatMessage(content: response, isUser: false));
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to connect to backend: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
      _scrollToBottom();
    }
  }

  void _addMessage(ChatMessage message) {
    setState(() {
      _messages.add(message);
      if (_currentChatId != null) {
        final chatIndex = _chatHistories.indexWhere(
          (h) => h.id == _currentChatId,
        );
        if (chatIndex != -1) {
          _chatHistories[chatIndex].messages = List.from(_messages);
        }
      }
    });
    _saveChatHistories();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: DropdownButton<String>(
          hint: Text('Select Topic', style: TextStyle(color: Colors.white)),
          value: _selectedTopic,
          icon: Icon(Icons.arrow_downward, color: Colors.white),
          underline: SizedBox(),
          items: _topics.map((String topic) {
            return DropdownMenuItem<String>(value: topic, child: Text(topic));
          }).toList(),
          onChanged: (String? newTopic) {
            if (newTopic != null) {
              _startNewChat(newTopic);
            }
          },
        ),
      ),
      drawer: Drawer(
        child: ListView(
          children: [
            DrawerHeader(
              decoration: BoxDecoration(color: Colors.blue),
              child: Text('Chat History'),
            ),
            ..._chatHistories.map(
              (chat) => ListTile(
                title: Text(
                  '${chat.topic} - ${chat.timestamp.toString().substring(0, 10)}',
                ),
                onTap: () {
                  _loadChat(chat.id);
                  Navigator.pop(context);
                },
              ),
            ),
          ],
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: _selectedTopic == null
                ? Center(
                    child: Text(
                      'Please select a topic to start the chat.',
                      style: TextStyle(fontSize: 18, color: Colors.red),
                    ),
                  )
                : ListView.builder(
                    controller: _scrollController,
                    padding: EdgeInsets.all(8.0),
                    itemCount: _messages.length,
                    itemBuilder: (context, index) {
                      return ChatMessageWidget(message: _messages[index]);
                    },
                  ),
          ),
          if (_errorMessage != null)
            Padding(
              padding: EdgeInsets.all(8.0),
              child: Text(_errorMessage!, style: TextStyle(color: Colors.red)),
            ),
          if (_isLoading)
            Padding(
              padding: EdgeInsets.all(8.0),
              child: SpinKitCircle(color: Colors.blue),
            ),
          Padding(
            padding: EdgeInsets.all(8.0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: 'Type your prompt...',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
                IconButton(
                  icon: Icon(Icons.send),
                  onPressed: _selectedTopic != null ? _sendMessage : null,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

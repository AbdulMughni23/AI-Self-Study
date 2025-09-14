import 'package:flutter/material.dart';
// import 'package:flutter_html/flutter_html.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import 'api_service.dart';
import 'models.dart';
import 'chat_messages.dart';
import 'package:dropdown_search/dropdown_search.dart';
import 'package:google_fonts/google_fonts.dart';

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

  final List<String> _topics = ['Motion', 'Energy', 'Momentum'];

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
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Widget _buildHistoryList() {
    return SizedBox(
      height: 400, // Fixed height to prevent overflow
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'History',
            style: GoogleFonts.inter(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: Colors.black,
            ),
          ),
          Opacity(
            opacity: 0.3,
            child: Divider(thickness: 2, color: Color(0xFF002C5F)),
          ),
          Text(
            'Completed',
            style: GoogleFonts.inter(
              fontSize: 14,
              fontWeight: FontWeight.normal,
              color: Colors.black,
            ),
          ),
          SizedBox(height: 10),
          Expanded(
            child: ListView.builder(
              shrinkWrap: true,
              itemCount: _chatHistories.length,
              itemBuilder: (context, index) {
                final chat = _chatHistories[index];
                return Padding(
                  padding: EdgeInsets.only(bottom: 8),
                  child: ListTile(
                    contentPadding: EdgeInsets.zero,
                    leading: Icon(
                      Icons.chat_bubble_outline_outlined,
                      color: Color(0xFF002C5F),
                      size: 18,
                    ),
                    title: Text(
                      chat.topic,
                      style: TextStyle(
                        fontFamily: 'Inter',
                        fontWeight: FontWeight.w600,
                        fontSize: 12,
                      ),
                    ),
                    subtitle: Text(
                      chat.timestamp.toString().substring(0, 10),
                      style: TextStyle(fontSize: 10),
                    ),
                    trailing: IconButton(
                      icon: Icon(
                        Icons.delete_outline,
                        color: Colors.red,
                        size: 16,
                      ),
                      onPressed: () =>
                          _showDeleteConfirmation(chat.id, chat.topic),
                      tooltip: 'Delete chat',
                    ),
                    onTap: () => _loadChat(chat.id),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          // Top Bar
          Container(
            decoration: BoxDecoration(color: Color(0xFFD9D9D9)),
            padding: EdgeInsets.symmetric(horizontal: 20, vertical: 20),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'AI Learner Chatbot',
                  style: TextStyle(
                    fontSize: 42.0,
                    fontFamily: 'InterTight',
                    fontWeight: FontWeight.w600,
                    color: Colors.black,
                  ),
                ),
                Row(
                  children: [
                    Text(
                      'Hi, Paul',
                      style: TextStyle(
                        fontSize: 16.0,
                        fontFamily: 'InterTight',
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    SizedBox(width: 20),
                    Container(
                      width: 40,
                      height: 40,
                      decoration: BoxDecoration(shape: BoxShape.circle),
                      child: ClipOval(
                        child: Image.network(
                          'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w0NTYyMDF8MHwxfHNlYXJjaHw3fHxwZXJzb258ZW58MHx8fHwxNzU3MTczNTc4fDA&ixlib=rb-4.0.1&q=80&w=1080',
                          fit: BoxFit.cover,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          // Main Body
          Expanded(
            child: Row(
              children: [
                // Left Sidebar
                Expanded(
                  flex: 1,
                  child: Container(
                    width: 300,
                    margin: EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Color(0xFFD9D9D9),
                      boxShadow: [
                        BoxShadow(
                          blurRadius: 3,
                          color: Color(0x33000000),
                          offset: Offset(5, 5),
                        ),
                      ],
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Padding(
                      padding: EdgeInsets.all(20),
                      child: Column(
                        children: [
                          DropdownSearch<String>(
                            popupProps: PopupProps.menu(
                              showSearchBox: true,
                              searchFieldProps: TextFieldProps(
                                style: GoogleFonts.inter(
                                  fontSize: 16,
                                  fontWeight: FontWeight.normal,
                                ),
                                decoration: InputDecoration(
                                  hintText: 'Search...',
                                  hintStyle: GoogleFonts.inter(
                                    color: Color(0xFF002C5F),
                                  ),
                                ),
                              ),
                            ),
                            items: _topics,
                            onChanged: (String? newTopic) {
                              if (newTopic != null) _startNewChat(newTopic);
                            },
                            dropdownDecoratorProps: DropDownDecoratorProps(
                              dropdownSearchDecoration: InputDecoration(
                                hintText: 'Select Topic...',
                                hintStyle: GoogleFonts.roboto(
                                  color: Colors.white,
                                  fontSize: 16,
                                ),
                                filled: true,
                                fillColor: Color(0xFF002C5F),
                                border: OutlineInputBorder(
                                  borderRadius: BorderRadius.circular(20),
                                  borderSide: BorderSide.none,
                                ),
                                contentPadding: EdgeInsets.symmetric(
                                  horizontal: 12,
                                  vertical: 16,
                                ),
                              ),
                            ),
                            // Add this dropdownBuilder property
                            dropdownBuilder: (context, selectedItem) {
                              return Text(
                                selectedItem ?? 'Select Topic...',
                                style: GoogleFonts.roboto(
                                  color: Colors
                                      .white, // Change this to your desired text color
                                  fontSize: 16,
                                ),
                              );
                            },
                            dropdownButtonProps: DropdownButtonProps(
                              icon: Icon(
                                Icons.arrow_drop_down,
                                color: Colors
                                    .white, // Change to your desired color
                              ),
                            ),
                          ),
                          SizedBox(height: 20),
                          Expanded(child: _buildHistoryList()),
                        ],
                      ),
                    ),
                  ),
                ),
                // Chat Area
                Expanded(
                  flex: 3,
                  child: Container(
                    margin: EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Color(0xFFD9D9D9),
                      boxShadow: [
                        BoxShadow(
                          blurRadius: 3,
                          color: Color(0x33000000),
                          offset: Offset(5, 5),
                        ),
                      ],
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisSize: MainAxisSize.max,
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: [
                            Padding(
                              padding: EdgeInsets.only(
                                left: 20,
                                top: 10,
                                bottom: 10,
                                right: 20,
                              ),
                              child: Text(
                                _selectedTopic != null
                                    ? _selectedTopic!
                                    : 'Select Topic',
                                style: GoogleFonts.interTight(
                                  fontSize: 45,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ),
                            Padding(
                              padding: EdgeInsetsDirectional.fromSTEB(
                                0,
                                0,
                                20,
                                0,
                              ),
                              child: ElevatedButton(
                                onPressed: () {
                                  // print('Button pressed ...');
                                },
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: Color(
                                    0xFF002C5F,
                                  ), // Primary color from your previous examples
                                  foregroundColor: Colors.white,
                                  minimumSize: Size(0, 40),
                                  padding: EdgeInsets.symmetric(horizontal: 16),
                                  textStyle: GoogleFonts.interTight(
                                    fontSize: 14.0,
                                    fontWeight: FontWeight.w500,
                                    fontStyle: FontStyle.normal,
                                    color: Colors.white,
                                    letterSpacing: 0.0,
                                  ),
                                  elevation: 0,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(20),
                                  ),
                                ),
                                child: Text('Questions on this Topic'),
                              ),
                            ),
                          ],
                        ),

                        // Chat Messages Area
                        Expanded(
                          child: Container(
                            // margin: EdgeInsets.symmetric(horizontal: 10),
                            margin: EdgeInsets.all(10),
                            padding: EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: Color(0xFFF7F7F7),
                              borderRadius: BorderRadius.circular(10),
                              border: Border.all(
                                color: Color(0xFF002C5F),
                                width: 1,
                              ),
                            ),
                            child: _selectedTopic == null
                                ? Center(
                                    child: Text(
                                      'Please select a topic to start the chat.',
                                      style: TextStyle(
                                        fontSize: 18,
                                        color: Colors.grey[600],
                                      ),
                                    ),
                                  )
                                : Column(
                                    children: [
                                      Expanded(
                                        child: ListView.builder(
                                          controller: _scrollController,
                                          itemCount: _messages.length,
                                          itemBuilder: (context, index) {
                                            return ChatMessageWidget(
                                              message: _messages[index],
                                            );
                                          },
                                        ),
                                      ),
                                      if (_errorMessage != null)
                                        Padding(
                                          padding: EdgeInsets.all(8.0),
                                          child: Text(
                                            _errorMessage!,
                                            style: TextStyle(color: Colors.red),
                                          ),
                                        ),
                                      if (_isLoading)
                                        Padding(
                                          padding: EdgeInsets.all(8.0),
                                          child: SpinKitCircle(
                                            color: Color(0xFF002C5F),
                                          ),
                                        ),
                                    ],
                                  ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                // Right Sidebar Panel
                Expanded(
                  flex: 2,
                  child: Container(
                    margin: EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Color(0xFFD9D9D9),
                      boxShadow: [
                        BoxShadow(
                          blurRadius: 3,
                          color: Color(0x33000000),
                          offset: Offset(5, 5),
                        ),
                      ],
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Padding(
                      padding: EdgeInsets.all(20),
                      child: Column(
                        children: [
                          Text(
                            'Ask Questions Related to the current topic',
                            style: GoogleFonts.interTight(
                              fontSize: 24,
                              fontWeight: FontWeight.w600,
                              color: Colors.black,
                            ),
                          ),
                          SizedBox(height: 20),
                          // Current Topic Display
                          if (_selectedTopic != null)
                            Container(
                              width: double.infinity,
                              padding: EdgeInsets.all(16),
                              decoration: BoxDecoration(
                                color: Color(0xFF002C5F),
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'Current Topic:',
                                    style: GoogleFonts.inter(
                                      fontSize: 14,
                                      fontWeight: FontWeight.normal,
                                      color: Colors.white,
                                    ),
                                  ),
                                  SizedBox(height: 8),
                                  Text(
                                    _selectedTopic!,
                                    style: GoogleFonts.inter(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.white,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          SizedBox(height: 20),
                          // Quick Actions or Information Panel
                          Expanded(
                            child: Container(
                              width: double.infinity,
                              padding: EdgeInsets.all(16),
                              decoration: BoxDecoration(
                                color: Color(0xFFF7F7F7),
                                borderRadius: BorderRadius.circular(10),
                                border: Border.all(
                                  color: Color(0xFF002C5F),
                                  width: 1,
                                ),
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'Quick Tips',
                                    style: GoogleFonts.inter(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                      color: Color(0xFF002C5F),
                                    ),
                                  ),
                                  SizedBox(height: 16),
                                  if (_selectedTopic != null) ...[
                                    _buildTipItem(
                                      'Ask specific questions for better answers',
                                    ),
                                    _buildTipItem(
                                      'Use examples in your questions',
                                    ),
                                    _buildTipItem(
                                      'Request explanations at your level',
                                    ),
                                    _buildTipItem('Ask for practice problems'),
                                  ] else
                                    Text(
                                      'Select a topic to see relevant tips and resources.',
                                      style: GoogleFonts.inter(
                                        fontSize: 14,
                                        color: Colors.grey[600],
                                      ),
                                    ),
                                  Spacer(),
                                  // Message Count
                                  if (_messages.isNotEmpty)
                                    Container(
                                      padding: EdgeInsets.symmetric(
                                        horizontal: 12,
                                        vertical: 8,
                                      ),
                                      decoration: BoxDecoration(
                                        color: Color(
                                          0xFF002C5F,
                                        ).withOpacity(0.1),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: Row(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          Icon(
                                            Icons.chat_bubble_outline,
                                            size: 16,
                                            color: Color(0xFF002C5F),
                                          ),
                                          SizedBox(width: 8),
                                          Text(
                                            '${_messages.length} messages',
                                            style: GoogleFonts.inter(
                                              fontSize: 12,
                                              color: Color(0xFF002C5F),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  // Input Field and Send Button
                                  Container(
                                    margin: EdgeInsets.only(top: 10),
                                    padding: EdgeInsets.all(8),
                                    decoration: BoxDecoration(
                                      color: Color(0xFFF2F2F2),
                                      borderRadius: BorderRadius.circular(12),
                                      boxShadow: [
                                        BoxShadow(
                                          blurRadius: 3,
                                          color: Color(0x33000000),
                                          offset: Offset(0, 1),
                                        ),
                                      ],
                                    ),
                                    child: Row(
                                      children: [
                                        Expanded(
                                          child: TextField(
                                            controller: _controller,
                                            decoration: InputDecoration(
                                              hintText: 'Type your question...',
                                              border: OutlineInputBorder(
                                                borderRadius:
                                                    BorderRadius.circular(20),
                                                borderSide: BorderSide.none,
                                              ),
                                              filled: true,
                                              fillColor: Colors.white,
                                              contentPadding:
                                                  EdgeInsets.symmetric(
                                                    horizontal: 16,
                                                    vertical: 12,
                                                  ),
                                            ),
                                            onSubmitted: (_) => _sendMessage(),
                                          ),
                                        ),
                                        SizedBox(width: 8),
                                        IconButton(
                                          icon: Icon(
                                            Icons.send_rounded,
                                            color: Color(0xFF002C5F),
                                            size: 24,
                                          ),
                                          onPressed: _selectedTopic != null
                                              ? _sendMessage
                                              : null,
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _showDeleteConfirmation(String chatId, String topic) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Delete Chat'),
          content: Text(
            'Are you sure you want to delete the chat for "$topic"?',
          ),
          actions: [
            TextButton(
              child: Text('Cancel'),
              onPressed: () => Navigator.of(context).pop(),
            ),
            TextButton(
              child: Text('Delete', style: TextStyle(color: Colors.red)),
              onPressed: () {
                setState(() {
                  _chatHistories.removeWhere((h) => h.id == chatId);
                  if (_currentChatId == chatId) {
                    _currentChatId = null;
                    _messages = [];
                    _selectedTopic = null;
                  }
                });
                _saveChatHistories();
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  Widget _buildTipItem(String tip) {
    return Padding(
      padding: EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 6,
            height: 6,
            margin: EdgeInsets.only(top: 6, right: 12),
            decoration: BoxDecoration(
              color: Color(0xFF002C5F),
              shape: BoxShape.circle,
            ),
          ),
          Expanded(
            child: Text(
              tip,
              style: GoogleFonts.inter(
                fontSize: 13,
                color: Colors.grey[700],
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

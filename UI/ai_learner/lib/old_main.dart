// import 'package:flutter/material.dart';
// // import 'package:flutter_html/flutter_html.dart';
// import 'package:flutter_spinkit/flutter_spinkit.dart';
// import 'package:shared_preferences/shared_preferences.dart';
// import 'dart:convert';
// import 'api_service.dart';
// import 'models.dart';
// import 'chat_messages.dart';
// import 'package:dropdown_search/dropdown_search.dart';
// import 'package:google_fonts/google_fonts.dart';

// void main() {
//   runApp(RAGLearningApp());
// }

// class RAGLearningApp extends StatelessWidget {
//   const RAGLearningApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       title: 'RAG Learning System',
//       theme: ThemeData(
//         primarySwatch: Colors.blue,
//         scaffoldBackgroundColor: Colors.grey[100],
//         fontFamily: 'Roboto',
//       ),
//       home: ChatScreen(),
//     );
//   }
// }

// class ChatScreen extends StatefulWidget {
//   const ChatScreen({super.key});

//   @override
//   _ChatScreenState createState() => _ChatScreenState();
// }

// class _ChatScreenState extends State<ChatScreen> {
//   final ApiService _apiService = ApiService();
//   final TextEditingController _controller = TextEditingController();
//   final ScrollController _scrollController = ScrollController();
//   List<ChatMessage> _messages = [];
//   List<ChatHistory> _chatHistories = [];
//   String? _selectedTopic;
//   bool _isLoading = false;
//   String? _errorMessage;
//   String? _currentChatId;

//   final List<String> _topics = [
//     'Motion',
//     'Energy',
//     'Momentum',
//   ]; // Hardcoded topics

//   @override
//   void initState() {
//     super.initState();
//     _loadChatHistories();
//   }

//   Future<void> _loadChatHistories() async {
//     final prefs = await SharedPreferences.getInstance();
//     final historiesJson = prefs.getString('chat_histories');
//     if (historiesJson != null) {
//       final List<dynamic> decoded = jsonDecode(historiesJson);
//       setState(() {
//         _chatHistories = decoded
//             .map((json) => ChatHistory.fromJson(json))
//             .toList();
//       });
//     }
//   }

//   Future<void> _saveChatHistories() async {
//     final prefs = await SharedPreferences.getInstance();
//     final historiesJson = jsonEncode(
//       _chatHistories.map((h) => h.toJson()).toList(),
//     );
//     await prefs.setString('chat_histories', historiesJson);
//   }

//   void _startNewChat(String topic) {
//     final newChat = ChatHistory(
//       id: DateTime.now().millisecondsSinceEpoch.toString(),
//       topic: topic,
//       messages: [],
//       timestamp: DateTime.now(),
//     );
//     setState(() {
//       _chatHistories.add(newChat);
//       _currentChatId = newChat.id;
//       _messages = [];
//       _selectedTopic = topic;
//       _errorMessage = null;
//     });
//     _saveChatHistories();
//     _getInitialResponse(topic);
//   }

//   void _loadChat(String chatId) {
//     final chat = _chatHistories.firstWhere((h) => h.id == chatId);
//     setState(() {
//       _currentChatId = chatId;
//       _messages = chat.messages;
//       _selectedTopic = chat.topic;
//       _errorMessage = null;
//     });
//     _scrollToBottom();
//   }

//   Future<void> _getInitialResponse(String topic) async {
//     setState(() {
//       _isLoading = true;
//       _errorMessage = null;
//     });
//     try {
//       final response = await _apiService.getTopicResponse(topic);
//       _addMessage(ChatMessage(content: response, isUser: false));
//     } catch (e) {
//       setState(() {
//         _errorMessage = 'Failed to connect to backend: $e';
//       });
//     } finally {
//       setState(() {
//         _isLoading = false;
//       });
//       _scrollToBottom();
//     }
//   }

//   Future<void> _sendMessage() async {
//     if (_controller.text.isEmpty || _selectedTopic == null) return;

//     final userMessage = _controller.text;
//     _addMessage(ChatMessage(content: userMessage, isUser: true));
//     _controller.clear();

//     setState(() {
//       _isLoading = true;
//       _errorMessage = null;
//     });

//     try {
//       final response = await _apiService.getAiResponse(
//         userMessage,
//         _selectedTopic!,
//       );
//       _addMessage(ChatMessage(content: response, isUser: false));
//     } catch (e) {
//       setState(() {
//         _errorMessage = 'Failed to connect to backend: $e';
//       });
//     } finally {
//       setState(() {
//         _isLoading = false;
//       });
//       _scrollToBottom();
//     }
//   }

//   void _addMessage(ChatMessage message) {
//     setState(() {
//       _messages.add(message);
//       if (_currentChatId != null) {
//         final chatIndex = _chatHistories.indexWhere(
//           (h) => h.id == _currentChatId,
//         );
//         if (chatIndex != -1) {
//           _chatHistories[chatIndex].messages = List.from(_messages);
//         }
//       }
//     });
//     _saveChatHistories();
//   }

//   void _scrollToBottom() {
//     WidgetsBinding.instance.addPostFrameCallback((_) {
//       _scrollController.animateTo(
//         _scrollController.position.maxScrollExtent,
//         duration: Duration(milliseconds: 300),
//         curve: Curves.easeOut,
//       );
//     });
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: DropdownButton<String>(
//           hint: Text('Select Topic', style: TextStyle(color: Colors.white)),
//           value: _selectedTopic,
//           icon: Icon(Icons.arrow_downward, color: Colors.white),
//           underline: SizedBox(),
//           items: _topics.map((String topic) {
//             return DropdownMenuItem<String>(value: topic, child: Text(topic));
//           }).toList(),
//           onChanged: (String? newTopic) {
//             if (newTopic != null) {
//               _startNewChat(newTopic);
//             }
//           },
//         ),
//       ),
//       drawer: Drawer(
//         child: ListView(
//           children: [
//             DrawerHeader(
//               decoration: BoxDecoration(color: Colors.blue),
//               child: Text('Chat History'),
//             ),
//             ..._chatHistories.map(
//               (chat) => ListTile(
//                 title: Text(
//                   '${chat.topic} - ${chat.timestamp.toString().substring(0, 10)}',
//                 ),
//                 onTap: () {
//                   _loadChat(chat.id);
//                   Navigator.pop(context);
//                 },
//               ),
//             ),
//           ],
//         ),
//       ),
//       body: Column(
//         children: [
//           // top bar
//           // Generated code for this Row Widget...
//           Row(
//             mainAxisSize: MainAxisSize.max,
//             mainAxisAlignment: MainAxisAlignment.spaceBetween,
//             children: [
//               Padding(
//                 padding: EdgeInsetsDirectional.fromSTEB(20, 0, 0, 0),
//                 child: Text(
//                   'AI Learner Chatbot',
//                   style: TextStyle(
//                     fontSize: 24.0, // Typical headlineMedium size
//                     fontFamily: 'InterTight',
//                     fontWeight: FontWeight
//                         .w500, // Medium weight (common for headlineMedium)
//                     fontStyle: FontStyle.normal,
//                     color: Colors.black, // Or your primary text color
//                     letterSpacing: 0.0,
//                   ),
//                 ),
//               ),
//               Padding(
//                 padding: EdgeInsetsDirectional.fromSTEB(0, 0, 20, 0),
//                 child: Row(
//                   mainAxisSize: MainAxisSize.max,
//                   children: [
//                     Padding(
//                       padding: EdgeInsetsDirectional.fromSTEB(0, 0, 20, 0),
//                       child: Text(
//                         'Hi, Paul',
//                         style: TextStyle(
//                           fontSize: 16.0, // Typical titleSmall size
//                           fontFamily: 'InterTight',
//                           fontWeight: FontWeight
//                               .w500, // Medium weight (common for titleSmall)
//                           fontStyle: FontStyle.normal,
//                           letterSpacing: 0.0,
//                         ),
//                       ),
//                     ),
//                     Container(
//                       width: MediaQuery.sizeOf(context).width * 0.03,
//                       height: MediaQuery.sizeOf(context).width * 0.03,
//                       clipBehavior: Clip.antiAlias,
//                       decoration: BoxDecoration(shape: BoxShape.circle),
//                       child: Image.network(
//                         'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w0NTYyMDF8MHwxfHNlYXJjaHw3fHxwZXJzb258ZW58MHx8fHwxNzU3MTczNTc4fDA&ixlib=rb-4.1.0&q=80&w=1080',
//                         fit: BoxFit.cover,
//                       ),
//                     ),
//                   ],
//                 ),
//               ),
//             ],
//           ),
//           // end of top bar

//           // Body start
//           // Generated code for this Body Widget...
//           Row(
//             mainAxisSize: MainAxisSize.max,
//             mainAxisAlignment: MainAxisAlignment.spaceAround,
//             children: [
//               Padding(
//                 padding: EdgeInsets.all(10),
//                 child: Column(
//                   mainAxisSize: MainAxisSize.max,
//                   children: [
//                     Container(
//                       width: MediaQuery.sizeOf(context).width * 0.2,
//                       height: MediaQuery.sizeOf(context).height * 0.85,
//                       decoration: BoxDecoration(
//                         color: Color(0x33000000),
//                         boxShadow: [
//                           BoxShadow(
//                             blurRadius: 3,
//                             color: Color(0x33000000), //alternate color
//                             offset: Offset(5, 5),
//                           ),
//                         ],
//                         borderRadius: BorderRadius.circular(10),
//                       ),
//                       child: Column(
//                         mainAxisSize: MainAxisSize.max,
//                         crossAxisAlignment: CrossAxisAlignment.center,
//                         children: [
//                           Padding(
//                             padding: EdgeInsetsDirectional.fromSTEB(
//                               20,
//                               20,
//                               20,
//                               10,
//                             ),

//                             child: DropdownSearch<String>(
//                               popupProps: PopupProps.menu(
//                                 showSearchBox: true,
//                                 searchFieldProps: TextFieldProps(
//                                   style: GoogleFonts.inter(
//                                     fontSize: 16,
//                                     fontWeight: FontWeight.normal,
//                                   ),
//                                   cursorColor: Colors.white,
//                                   decoration: InputDecoration(
//                                     hintText: 'Search...',
//                                     hintStyle: GoogleFonts.inter(
//                                       color: Colors.grey[600],
//                                     ),
//                                   ),
//                                 ),
//                               ),
//                               items: _topics,
//                               onChanged: (String? newTopic) {
//                                 if (newTopic != null) _startNewChat(newTopic);
//                               },
//                               dropdownDecoratorProps: DropDownDecoratorProps(
//                                 dropdownSearchDecoration: InputDecoration(
//                                   hintText: 'Select Topic...',
//                                   hintStyle: GoogleFonts.roboto(
//                                     color: Colors.white,
//                                     fontSize: 16,
//                                   ),
//                                   filled: true,
//                                   fillColor: Color(0xFF002C5F), // primary color
//                                   border: OutlineInputBorder(
//                                     borderRadius: BorderRadius.circular(20),
//                                     borderSide: BorderSide.none,
//                                   ),
//                                   contentPadding: EdgeInsets.symmetric(
//                                     horizontal: 12,
//                                     vertical: 0,
//                                   ),
//                                 ),
//                               ),
//                               clearButtonProps: ClearButtonProps(
//                                 icon: Icon(Icons.clear, color: Colors.white),
//                               ),
//                               dropdownButtonProps: DropdownButtonProps(
//                                 icon: Icon(
//                                   Icons.keyboard_arrow_down_rounded,
//                                   color: Colors.white,
//                                   size: 24,
//                                 ),
//                               ),
//                             ),
//                           ),
//                           Container(
//                             width: 275,
//                             height: 698.8,
//                             decoration: BoxDecoration(
//                               color: Color(0x33000000), // alternate color
//                             ),
//                             child: Padding(
//                               padding: EdgeInsetsDirectional.fromSTEB(
//                                 10,
//                                 20,
//                                 0,
//                                 0,
//                               ),
//                               child: Column(
//                                 mainAxisSize: MainAxisSize.max,
//                                 crossAxisAlignment: CrossAxisAlignment.start,
//                                 children: [
//                                   Text(
//                                     'History',
//                                     style: GoogleFonts.inter(
//                                       fontSize:
//                                           16, // Typical bodyLarge font size
//                                       fontWeight: FontWeight.bold,
//                                       fontStyle: FontStyle
//                                           .normal, // Default font style
//                                       letterSpacing: 0.0,
//                                       color: Colors
//                                           .black, // Add your preferred text color
//                                     ),
//                                   ),
//                                   Opacity(
//                                     opacity: 0.3,
//                                     child: Divider(
//                                       thickness: 2,
//                                       color: Color(
//                                         0xFF002C5F,
//                                       ), // primary color,
//                                     ),
//                                   ),
//                                   Padding(
//                                     padding: EdgeInsetsDirectional.fromSTEB(
//                                       0,
//                                       0,
//                                       0,
//                                       10,
//                                     ),
//                                     child: Text(
//                                       'Completed',
//                                       style: GoogleFonts.inter(
//                                         fontSize:
//                                             14, // Typical bodyMedium font size
//                                         fontWeight: FontWeight
//                                             .normal, // Default bodyMedium weight
//                                         fontStyle: FontStyle
//                                             .normal, // Default font style
//                                         letterSpacing: 0.0,
//                                         color: Colors
//                                             .black, // Add your preferred text color
//                                       ),
//                                     ),
//                                   ),

//                                   ///---------------------------------history list------------------------------
//                                   ListView(
//                                     children: [
//                                       DrawerHeader(
//                                         decoration: BoxDecoration(
//                                           color: Colors.blue,
//                                         ),
//                                         child: Text('Chat History'),
//                                       ),
//                                       ..._chatHistories.map(
//                                         (chat) => Padding(
//                                           padding: EdgeInsets.only(top: 10),
//                                           child: ListTile(
//                                             contentPadding: EdgeInsets.zero,
//                                             title: Row(
//                                               mainAxisSize: MainAxisSize.max,
//                                               children: [
//                                                 Padding(
//                                                   padding: EdgeInsets.only(
//                                                     right: 5,
//                                                   ),
//                                                   child: Icon(
//                                                     Icons
//                                                         .chat_bubble_outline_outlined,
//                                                     color: Color(
//                                                       0xFF002C5F,
//                                                     ), // primary color
//                                                     size: 18,
//                                                   ),
//                                                 ),
//                                                 Text(
//                                                   '${chat.topic} - ${chat.timestamp.toString().substring(0, 10)}',
//                                                   style: TextStyle(
//                                                     fontFamily: 'Inter',
//                                                     fontWeight: FontWeight.w600,
//                                                     letterSpacing: 0.0,
//                                                   ),
//                                                 ),
//                                               ],
//                                             ),
//                                             onTap: () {
//                                               _loadChat(chat.id);
//                                               Navigator.pop(context);
//                                             },
//                                           ),
//                                         ),
//                                       ),
//                                     ],
//                                   ),
//                                 ],
//                               ),
//                             ),
//                           ),
//                         ],
//                       ),
//                     ),
//                   ],
//                 ),
//               ),

//               // middle section implementation
//               Expanded(
//                 child: _selectedTopic == null
//                     ? Center(
//                         child: Text(
//                           'Please select a topic to start the chat.',
//                           style: TextStyle(fontSize: 18, color: Colors.red),
//                         ),
//                       )
//                     : ListView.builder(
//                         controller: _scrollController,
//                         padding: EdgeInsets.all(8.0),
//                         itemCount: _messages.length,
//                         itemBuilder: (context, index) {
//                           return ChatMessageWidget(message: _messages[index]);
//                         },
//                       ),
//               ),
//               if (_errorMessage != null)
//                 Padding(
//                   padding: EdgeInsets.all(8.0),
//                   child: Text(
//                     _errorMessage!,
//                     style: TextStyle(color: Colors.red),
//                   ),
//                 ),
//               if (_isLoading)
//                 Padding(
//                   padding: EdgeInsets.all(8.0),
//                   child: SpinKitCircle(color: Colors.blue),
//                 ),

//               // ------------------------- end of middle section -------------------------
//               Padding(
//                 padding: EdgeInsets.all(10),
//                 child: Column(
//                   mainAxisSize: MainAxisSize.max,
//                   children: [
//                     Container(
//                       width: MediaQuery.sizeOf(context).width * 0.258,
//                       height: MediaQuery.sizeOf(context).height * 0.85,
//                       decoration: BoxDecoration(
//                         color: Color(0x33000000),
//                         boxShadow: [
//                           BoxShadow(
//                             blurRadius: 3,
//                             color: Color(0x33000000),
//                             offset: Offset(5, 5),
//                           ),
//                         ],
//                         borderRadius: BorderRadius.circular(10),
//                       ),
//                       child: Column(
//                         mainAxisSize: MainAxisSize.max,
//                         children: [
//                           Padding(
//                             padding: EdgeInsetsDirectional.fromSTEB(
//                               10,
//                               10,
//                               10,
//                               10,
//                             ),
//                             child: Text(
//                               'Ask Any Question on the Current Topic',
//                               style: GoogleFonts.interTight(
//                                 fontSize: 24,
//                                 fontWeight: FontWeight.w500,
//                                 fontStyle: FontStyle.normal,
//                                 letterSpacing: 0.0,
//                               ),
//                             ),
//                           ),
//                           Expanded(
//                             child: Container(
//                               decoration: BoxDecoration(),
//                               child: ListView(
//                                 padding: EdgeInsets.zero,
//                                 reverse: true,
//                                 scrollDirection: Axis.vertical,
//                                 children: [
//                                   Card(
//                                     clipBehavior: Clip.antiAliasWithSaveLayer,
//                                     color: Color(0x33000000),
//                                     elevation: 0,
//                                     shape: RoundedRectangleBorder(
//                                       borderRadius: BorderRadius.circular(8),
//                                     ),
//                                     child: Column(
//                                       mainAxisSize: MainAxisSize.max,
//                                       children: [
//                                         Padding(
//                                           padding:
//                                               EdgeInsetsDirectional.fromSTEB(
//                                                 0,
//                                                 10,
//                                                 0,
//                                                 10,
//                                               ),
//                                           child: Container(
//                                             constraints: BoxConstraints(
//                                               maxWidth: () {
//                                                 if (MediaQuery.sizeOf(
//                                                       context,
//                                                     ).width >=
//                                                     1170.0) {
//                                                   return 700.0;
//                                                 } else if (MediaQuery.sizeOf(
//                                                       context,
//                                                     ).width <=
//                                                     470.0) {
//                                                   return 330.0;
//                                                 } else {
//                                                   return 530.0;
//                                                 }
//                                               }(),
//                                             ),
//                                             decoration: BoxDecoration(
//                                               color: Color(
//                                                 0xFFF7F7F7,
//                                               ), // primary background color
//                                               borderRadius:
//                                                   BorderRadius.circular(10),
//                                               border: Border.all(
//                                                 color: Color(
//                                                   0xFF002C5F,
//                                                 ), // primary color
//                                                 width: 2,
//                                               ),
//                                             ),
//                                             child: Padding(
//                                               padding:
//                                                   EdgeInsetsDirectional.fromSTEB(
//                                                     12,
//                                                     8,
//                                                     12,
//                                                     8,
//                                                   ),
//                                               //---------------chat column----------------
//                                               child: Column(
//                                                 children: [
//                                                   Expanded(
//                                                     child:
//                                                         _selectedTopic == null
//                                                         ? Center(
//                                                             child: Text(
//                                                               'Please select a topic to start the chat.',
//                                                               style: TextStyle(
//                                                                 fontSize: 18,
//                                                                 color:
//                                                                     Colors.red,
//                                                               ),
//                                                             ),
//                                                           )
//                                                         : ListView.builder(
//                                                             controller:
//                                                                 _scrollController,
//                                                             padding:
//                                                                 EdgeInsets.all(
//                                                                   8.0,
//                                                                 ),
//                                                             itemCount: _messages
//                                                                 .length,
//                                                             itemBuilder:
//                                                                 (
//                                                                   context,
//                                                                   index,
//                                                                 ) {
//                                                                   return ChatMessageWidget(
//                                                                     message:
//                                                                         _messages[index],
//                                                                   );
//                                                                 },
//                                                           ),
//                                                   ),
//                                                   if (_errorMessage != null)
//                                                     Padding(
//                                                       padding: EdgeInsets.all(
//                                                         8.0,
//                                                       ),
//                                                       child: Text(
//                                                         _errorMessage!,
//                                                         style: TextStyle(
//                                                           color: Colors.red,
//                                                         ),
//                                                       ),
//                                                     ),
//                                                   if (_isLoading)
//                                                     Padding(
//                                                       padding: EdgeInsets.all(
//                                                         8.0,
//                                                       ),
//                                                       child: SpinKitCircle(
//                                                         color: Color(
//                                                           0xFF002C5F,
//                                                         ),
//                                                       ), //primary color
//                                                     ),
//                                                 ],
//                                               ),

//                                               // ------------ end chat column -------------
//                                             ),
//                                           ),
//                                         ),
//                                       ],
//                                     ),
//                                   ),
//                                 ],
//                               ),
//                             ),
//                           ),

//                           Padding(
//                             padding: EdgeInsets.all(12),
//                             child: Container(
//                               width: double.infinity,
//                               decoration: BoxDecoration(
//                                 color: Color(
//                                   0xFFF2F2F2,
//                                 ), // Secondary background color
//                                 boxShadow: [
//                                   BoxShadow(
//                                     blurRadius: 3,
//                                     color: Color(0x33000000),
//                                     offset: Offset(0, 1),
//                                   ),
//                                 ],
//                                 borderRadius: BorderRadius.circular(12),
//                               ),
//                               child: Stack(
//                                 children: [
//                                   SizedBox(
//                                     width: double.infinity,
//                                     child: Row(
//                                       children: [
//                                         Expanded(
//                                           child: TextField(
//                                             controller: _controller,
//                                             decoration: InputDecoration(
//                                               hintText: 'Type your prompt...',
//                                               border: OutlineInputBorder(
//                                                 borderRadius:
//                                                     BorderRadius.circular(20),
//                                               ),
//                                             ),
//                                             onSubmitted: (_) => _sendMessage(),
//                                           ),
//                                         ),
//                                         IconButton(
//                                           icon: Icon(Icons.send),
//                                           onPressed: _selectedTopic != null
//                                               ? _sendMessage
//                                               : null,
//                                         ),
//                                       ],
//                                     ),
//                                   ),
//                                   Align(
//                                     alignment: AlignmentDirectional(1, 0),
//                                     child: IconButton(
//                                       icon: Icon(Icons.send_rounded, size: 30),
//                                       onPressed: () async {},
//                                     ),
//                                   ),
//                                 ],
//                               ),
//                             ),
//                           ),
//                         ],
//                       ),
//                     ),
//                   ],
//                 ),
//               ),
//             ],
//           ),

//           // Body end
//           Expanded(
//             child: _selectedTopic == null
//                 ? Center(
//                     child: Text(
//                       'Please select a topic to start the chat.',
//                       style: TextStyle(fontSize: 18, color: Colors.red),
//                     ),
//                   )
//                 : ListView.builder(
//                     controller: _scrollController,
//                     padding: EdgeInsets.all(8.0),
//                     itemCount: _messages.length,
//                     itemBuilder: (context, index) {
//                       return ChatMessageWidget(message: _messages[index]);
//                     },
//                   ),
//           ),
//           if (_errorMessage != null)
//             Padding(
//               padding: EdgeInsets.all(8.0),
//               child: Text(_errorMessage!, style: TextStyle(color: Colors.red)),
//             ),
//           if (_isLoading)
//             Padding(
//               padding: EdgeInsets.all(8.0),
//               child: SpinKitCircle(color: Colors.blue),
//             ),
//           Padding(
//             padding: EdgeInsets.all(8.0),
//             child: Row(
//               children: [
//                 Expanded(
//                   child: TextField(
//                     controller: _controller,
//                     decoration: InputDecoration(
//                       hintText: 'Type your prompt...',
//                       border: OutlineInputBorder(
//                         borderRadius: BorderRadius.circular(20),
//                       ),
//                     ),
//                     onSubmitted: (_) => _sendMessage(),
//                   ),
//                 ),
//                 IconButton(
//                   icon: Icon(Icons.send),
//                   onPressed: _selectedTopic != null ? _sendMessage : null,
//                 ),
//               ],
//             ),
//           ),
//         ],
//       ),
//     );
//   }
// }

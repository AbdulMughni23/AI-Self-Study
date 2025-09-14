import 'package:flutter/material.dart';
import 'package:flutter_html/flutter_html.dart';
import 'models.dart';
import 'package:google_fonts/google_fonts.dart';

class ChatMessageWidget extends StatelessWidget {
  final ChatMessage message;

  const ChatMessageWidget({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Padding(
        padding: EdgeInsetsDirectional.fromSTEB(0, 10, 0, 10),
        child: Column(
          mainAxisSize: MainAxisSize.max,
          children: [
            Card(
              clipBehavior: Clip.antiAliasWithSaveLayer,
              color: Colors.white,
              elevation: 0,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.max,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: EdgeInsetsDirectional.fromSTEB(0, 10, 0, 10),
                    child: Container(
                      // constraints: BoxConstraints(
                      // maxWidth: () {
                      //   if (MediaQuery.sizeOf(context).width >= 1170.0) {
                      //     return 700.0;
                      //   } else if (MediaQuery.sizeOf(context).width <=
                      //       470.0) {
                      //     return 330.0;
                      //   } else {
                      //     return 530.0;
                      //   }
                      // }(),
                      // ),
                      decoration: BoxDecoration(
                        color: Color(0xFFF7F7F7),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(
                          color: Color(0xFF002C5F), // primary color
                          width: 1,
                        ),
                      ),
                      child: Padding(
                        padding: EdgeInsetsDirectional.fromSTEB(12, 8, 12, 8),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Html(
                              data: message.content,
                              style: {'body': Style(margin: Margins.zero)},
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      // Next Topic Button
                      Padding(
                        padding: EdgeInsets.only(left: 0),
                        child: ElevatedButton(
                          onPressed: () {
                            // print('Button pressed ...');
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Color(0xFF002C5F), // Primary color
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
                          child: Text('Next Topic: [Topic Name]'),
                        ),
                      ),

                      // Explain Again Button
                      Padding(
                        padding: EdgeInsets.only(right: 0),
                        child: ElevatedButton.icon(
                          onPressed: () {
                            // print('Button pressed ...');
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Color(0xFF002C5F), // Primary color
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
                          icon: Icon(Icons.refresh_sharp, size: 15),
                          label: Text('Explain Again'),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:flutter/gestures.dart';
import 'dart:ui';
import 'package:google_fonts/google_fonts.dart';
import 'package:myapp/utils.dart';

class Scene extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    double baseWidth = 390;
    double fem = MediaQuery.of(context).size.width / baseWidth;
    double ffem = fem * 0.97;
    return Container(
      width: double.infinity,
      child: Container(
        // loginpageW1w (1:25)
        padding: EdgeInsets.fromLTRB(24 * fem, 24 * fem, 24 * fem, 16 * fem),
        width: double.infinity,
        decoration: BoxDecoration(
          color: Color(0xffffffff),
          borderRadius: BorderRadius.circular(30 * fem),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Container(
              // image18qw (1:27)
              margin:
                  EdgeInsets.fromLTRB(0 * fem, 0 * fem, 3 * fem, 26.5 * fem),
              width: 234 * fem,
              height: 188 * fem,
              child: Image.asset(
                'assets/page-1/images/image-1.png',
                fit: BoxFit.cover,
              ),
            ),
            Container(
              // loginformdGu (1:28)
              margin: EdgeInsets.fromLTRB(0 * fem, 0 * fem, 0 * fem, 126 * fem),
              width: double.infinity,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    // loginLx1 (1:30)
                    margin: EdgeInsets.fromLTRB(
                        0 * fem, 0 * fem, 0 * fem, 23.5 * fem),
                    child: Text(
                      'Login',
                      style: SafeGoogleFont(
                        'Roboto',
                        fontSize: 32 * ffem,
                        fontWeight: FontWeight.w700,
                        height: 1.6000000238 * ffem / fem,
                        letterSpacing: 0.32 * fem,
                        color: Color(0xff000000),
                      ),
                    ),
                  ),
                  Container(
                    // formQgy (1:31)
                    margin: EdgeInsets.fromLTRB(
                        0 * fem, 0 * fem, 0 * fem, 42 * fem),
                    width: double.infinity,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Container(
                          // inputJGZ (1:32)
                          width: double.infinity,
                          height: 42 * fem,
                          decoration: BoxDecoration(
                            boxShadow: [
                              BoxShadow(
                                color: Color(0x3f000000),
                                offset: Offset(0 * fem, 4 * fem),
                                blurRadius: 2 * fem,
                              ),
                            ],
                          ),
                          child: Container(
                            // inputfFX (I1:32;114:115)
                            padding: EdgeInsets.fromLTRB(
                                14.1 * fem, 14 * fem, 14.1 * fem, 14 * fem),
                            width: double.infinity,
                            height: double.infinity,
                            decoration: BoxDecoration(
                              border: Border.all(color: Color(0xffd4d7e3)),
                              color: Color(0xfff2f6fa),
                              borderRadius: BorderRadius.circular(8 * fem),
                            ),
                            child: Text(
                              'Bajaj Id',
                              style: SafeGoogleFont(
                                'Roboto',
                                fontSize: 14 * ffem,
                                fontWeight: FontWeight.w400,
                                height: 1 * ffem / fem,
                                letterSpacing: 0.14 * fem,
                                color: Color(0xff8796ad),
                              ),
                            ),
                          ),
                        ),
                        SizedBox(
                          height: 16 * fem,
                        ),
                        Container(
                          // input65w (1:33)
                          width: double.infinity,
                          height: 42 * fem,
                          decoration: BoxDecoration(
                            boxShadow: [
                              BoxShadow(
                                color: Color(0x3f000000),
                                offset: Offset(0 * fem, 4 * fem),
                                blurRadius: 2 * fem,
                              ),
                            ],
                          ),
                          child: Container(
                            // inputc4H (I1:33;114:115)
                            padding: EdgeInsets.fromLTRB(
                                14.1 * fem, 14 * fem, 14.1 * fem, 14 * fem),
                            width: double.infinity,
                            height: double.infinity,
                            decoration: BoxDecoration(
                              border: Border.all(color: Color(0xffd4d7e3)),
                              color: Color(0xfff2f6fa),
                              borderRadius: BorderRadius.circular(8 * fem),
                            ),
                            child: Text(
                              'Password',
                              style: SafeGoogleFont(
                                'Roboto',
                                fontSize: 14 * ffem,
                                fontWeight: FontWeight.w400,
                                height: 1 * ffem / fem,
                                letterSpacing: 0.14 * fem,
                                color: Color(0xff8796ad),
                              ),
                            ),
                          ),
                        ),
                        SizedBox(
                          height: 16 * fem,
                        ),
                        Center(
                          // forgotpassword3QV (1:34)
                          child: Container(
                            width: double.infinity,
                            child: Text(
                              'Forgot Password?',
                              textAlign: TextAlign.center,
                              style: SafeGoogleFont(
                                'Roboto',
                                fontSize: 14 * ffem,
                                fontWeight: FontWeight.w400,
                                height: 1 * ffem / fem,
                                letterSpacing: 0.14 * fem,
                                color: Color(0xff1d4ae8),
                              ),
                            ),
                          ),
                        ),
                        SizedBox(
                          height: 16 * fem,
                        ),
                        Container(
                          // mainbutton8Rw (1:35)
                          width: double.infinity,
                          height: 44 * fem,
                          decoration: BoxDecoration(
                            color: Color(0xff0056ff),
                            borderRadius: BorderRadius.circular(12 * fem),
                            boxShadow: [
                              BoxShadow(
                                color: Color(0x3f000000),
                                offset: Offset(0 * fem, 4 * fem),
                                blurRadius: 2 * fem,
                              ),
                            ],
                          ),
                          child: Center(
                            child: Center(
                              child: Text(
                                'Login',
                                textAlign: TextAlign.center,
                                style: SafeGoogleFont(
                                  'Roboto',
                                  fontSize: 16 * ffem,
                                  fontWeight: FontWeight.w400,
                                  height: 1 * ffem / fem,
                                  letterSpacing: 0.16 * fem,
                                  color: Color(0xffffffff),
                                ),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    // oro2H (1:37)
                    margin: EdgeInsets.fromLTRB(
                        0 * fem, 0 * fem, 0 * fem, 34 * fem),
                    width: double.infinity,
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Container(
                          // line2KWR (1:38)
                          margin: EdgeInsets.fromLTRB(
                              0 * fem, 1 * fem, 0 * fem, 0 * fem),
                          width: 109.5 * fem,
                          height: 1 * fem,
                          decoration: BoxDecoration(
                            color: Color(0xffcfdfe2),
                          ),
                        ),
                        SizedBox(
                          width: 16 * fem,
                        ),
                        Center(
                          // orsigninwithckR (1:39)
                          child: Text(
                            'Or sign in with',
                            textAlign: TextAlign.center,
                            style: SafeGoogleFont(
                              'Roboto',
                              fontSize: 14 * ffem,
                              fontWeight: FontWeight.w400,
                              height: 1 * ffem / fem,
                              letterSpacing: 0.14 * fem,
                              color: Color(0xff294956),
                            ),
                          ),
                        ),
                        SizedBox(
                          width: 16 * fem,
                        ),
                        Container(
                          // line1vm7 (1:40)
                          margin: EdgeInsets.fromLTRB(
                              0 * fem, 1 * fem, 0 * fem, 0 * fem),
                          width: 109.5 * fem,
                          height: 1 * fem,
                          decoration: BoxDecoration(
                            color: Color(0xffcfdfe2),
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    // frame2FoP (1:41)
                    margin: EdgeInsets.fromLTRB(
                        0 * fem, 0 * fem, 0 * fem, 24 * fem),
                    width: double.infinity,
                    height: 44 * fem,
                    decoration: BoxDecoration(
                      color: Color(0xfff3f9f8),
                      borderRadius: BorderRadius.circular(12 * fem),
                      boxShadow: [
                        BoxShadow(
                          color: Color(0x19000000),
                          offset: Offset(0 * fem, 4 * fem),
                          blurRadius: 2 * fem,
                        ),
                      ],
                    ),
                    child: Stack(
                      children: [
                        Positioned(
                          // bajajaccountwAR (1:43)
                          left: 93 * fem,
                          top: 11.5 * fem,
                          child: Center(
                            child: Align(
                              child: SizedBox(
                                width: 105 * fem,
                                height: 26 * fem,
                                child: RichText(
                                  textAlign: TextAlign.center,
                                  text: TextSpan(
                                    style: SafeGoogleFont(
                                      'Roboto',
                                      fontSize: 16 * ffem,
                                      fontWeight: FontWeight.w400,
                                      height: 1.6000000238 * ffem / fem,
                                      letterSpacing: 0.16 * fem,
                                      color: Color(0xffffffff),
                                    ),
                                    children: [
                                      TextSpan(
                                        text: '                     ',
                                        style: SafeGoogleFont(
                                          'Roboto',
                                          fontSize: 16 * ffem,
                                          fontWeight: FontWeight.w300,
                                          height: 1.6000000238 * ffem / fem,
                                          letterSpacing: 0.16 * fem,
                                          color: Color(0xffffffff),
                                        ),
                                      ),
                                      TextSpan(
                                        text: 'Bajaj Account',
                                        style: SafeGoogleFont(
                                          'Roboto',
                                          fontSize: 16 * ffem,
                                          fontWeight: FontWeight.w300,
                                          height: 1.6000000238 * ffem / fem,
                                          letterSpacing: 0.16 * fem,
                                          color: Color(0xff000000),
                                        ),
                                      ),
                                      TextSpan(
                                        text: ' ',
                                        style: SafeGoogleFont(
                                          'Roboto',
                                          fontSize: 16 * ffem,
                                          fontWeight: FontWeight.w300,
                                          height: 1.6000000238 * ffem / fem,
                                          letterSpacing: 0.16 * fem,
                                          color: Color(0xffffffff),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ),
                        Positioned(
                          // arcticonsbajajfinservdUV (1:44)
                          left: 84.0833332539 * fem,
                          top: 4.0833129883 * fem,
                          child: Align(
                            child: SizedBox(
                              width: 35.83 * fem,
                              height: 35.83 * fem,
                              child: Image.asset(
                                'assets/page-1/images/arcticons-bajaj-finserv-Nbb.png',
                                width: 35.83 * fem,
                                height: 35.83 * fem,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  Center(
                    // donthaveanaccountsignupuS1 (1:47)
                    child: Container(
                      width: double.infinity,
                      child: RichText(
                        textAlign: TextAlign.center,
                        text: TextSpan(
                          style: SafeGoogleFont(
                            'Roboto',
                            fontSize: 16 * ffem,
                            fontWeight: FontWeight.w400,
                            height: 1.6000000238 * ffem / fem,
                            letterSpacing: 0.16 * fem,
                            color: Color(0xff112b30),
                          ),
                          children: [
                            TextSpan(
                              text: 'Don\'t have an account? ',
                              style: SafeGoogleFont(
                                'Roboto',
                                fontSize: 16 * ffem,
                                fontWeight: FontWeight.w400,
                                height: 1.6000000238 * ffem / fem,
                                letterSpacing: 0.16 * fem,
                                color: Color(0xff313957),
                              ),
                            ),
                            TextSpan(
                              text: 'Sign up',
                              style: SafeGoogleFont(
                                'Roboto',
                                fontSize: 16 * ffem,
                                fontWeight: FontWeight.w400,
                                height: 1.6000000238 * ffem / fem,
                                letterSpacing: 0.16 * fem,
                                color: Color(0xff1d4ae8),
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
            Center(
              // allrightsreservedd93 (1:48)
              child: Text(
                '© 2023 ALL RIGHTS RESERVED',
                textAlign: TextAlign.center,
                style: SafeGoogleFont(
                  'Roboto',
                  fontSize: 14 * ffem,
                  fontWeight: FontWeight.w400,
                  height: 1 * ffem / fem,
                  letterSpacing: 0.14 * fem,
                  color: Color(0xff959cb5),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
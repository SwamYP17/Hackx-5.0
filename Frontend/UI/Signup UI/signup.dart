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
        // signuppageEC1 (1:49)
        padding: EdgeInsets.fromLTRB(24*fem, 24*fem, 24*fem, 19*fem),
        width: double.infinity,
        decoration: BoxDecoration (
          color: Color(0xffffffff),
          borderRadius: BorderRadius.circular(30*fem),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Container(
              // image2KjF (1:51)
              margin: EdgeInsets.fromLTRB(0*fem, 0*fem, 13*fem, 26.5*fem),
              width: 234*fem,
              height: 185*fem,
              child: Image.asset(
                'assets/page-1/images/image-2.png',
                fit: BoxFit.cover,
              ),
            ),
            Container(
              // loginformEbK (1:52)
              margin: EdgeInsets.fromLTRB(0*fem, 0*fem, 0*fem, 40*fem),
              width: double.infinity,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    // signupNBj (1:54)
                    margin: EdgeInsets.fromLTRB(0*fem, 0*fem, 0*fem, 23.5*fem),
                    child: Text(
                      'Sign up',
                      style: SafeGoogleFont (
                        'Roboto',
                        fontSize: 32*ffem,
                        fontWeight: FontWeight.w700,
                        height: 1.6000000238*ffem/fem,
                        letterSpacing: 0.32*fem,
                        color: Color(0xff000000),
                      ),
                    ),
                  ),
                  Container(
                    // formeuw (1:55)
                    margin: EdgeInsets.fromLTRB(0*fem, 0*fem, 0*fem, 42*fem),
                    width: double.infinity,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Container(
                          // inputyxD (1:56)
                          width: double.infinity,
                          height: 42*fem,
                          decoration: BoxDecoration (
                            boxShadow: [
                              BoxShadow(
                                color: Color(0x3f000000),
                                offset: Offset(0*fem, 4*fem),
                                blurRadius: 2*fem,
                              ),
                            ],
                          ),
                          child: Container(
                            // inputhtD (I1:56;114:115)
                            padding: EdgeInsets.fromLTRB(14.1*fem, 14*fem, 14.1*fem, 14*fem),
                            width: double.infinity,
                            height: double.infinity,
                            decoration: BoxDecoration (
                              border: Border.all(color: Color(0xffd4d7e3)),
                              color: Color(0xfff2f6fa),
                              borderRadius: BorderRadius.circular(8*fem),
                            ),
                            child: Text(
                              'Bajaj Id',
                              style: SafeGoogleFont (
                                'Roboto',
                                fontSize: 14*ffem,
                                fontWeight: FontWeight.w400,
                                height: 1*ffem/fem,
                                letterSpacing: 0.14*fem,
                                color: Color(0xff8796ad),
                              ),
                            ),
                          ),
                        ),
                        SizedBox(
                          height: 16*fem,
                        ),
                        Container(
                          // inputYP3 (1:57)
                          width: double.infinity,
                          height: 42*fem,
                          child: Container(
                            // inputgk9 (I1:57;114:115)
                            padding: EdgeInsets.fromLTRB(14.1*fem, 14*fem, 14.1*fem, 14*fem),
                            width: double.infinity,
                            height: double.infinity,
                            decoration: BoxDecoration (
                              border: Border.all(color: Color(0xffd4d7e3)),
                              color: Color(0xfff2f6fa),
                              borderRadius: BorderRadius.circular(8*fem),
                              boxShadow: [
                                BoxShadow(
                                  color: Color(0x3f000000),
                                  offset: Offset(0*fem, 4*fem),
                                  blurRadius: 2*fem,
                                ),
                              ],
                            ),
                            child: Text(
                              'Full name',
                              style: SafeGoogleFont (
                                'Roboto',
                                fontSize: 14*ffem,
                                fontWeight: FontWeight.w400,
                                height: 1*ffem/fem,
                                letterSpacing: 0.14*fem,
                                color: Color(0xff8796ad),
                              ),
                            ),
                          ),
                        ),
                        SizedBox(
                          height: 16*fem,
                        ),
                        Container(
                          // input6os (1:58)
                          width: double.infinity,
                          height: 42*fem,
                          child: Container(
                            // inputro3 (I1:58;114:115)
                            padding: EdgeInsets.fromLTRB(14.1*fem, 14*fem, 14.1*fem, 14*fem),
                            width: double.infinity,
                            height: double.infinity,
                            decoration: BoxDecoration (
                              border: Border.all(color: Color(0xffd4d7e3)),
                              color: Color(0xfff2f6fa),
                              borderRadius: BorderRadius.circular(8*fem),
                              boxShadow: [
                                BoxShadow(
                                  color: Color(0x3f000000),
                                  offset: Offset(0*fem, 4*fem),
                                  blurRadius: 2*fem,
                                ),
                              ],
                            ),
                            child: Text(
                              'Password',
                              style: SafeGoogleFont (
                                'Roboto',
                                fontSize: 14*ffem,
                                fontWeight: FontWeight.w400,
                                height: 1*ffem/fem,
                                letterSpacing: 0.14*fem,
                                color: Color(0xff8796ad),
                              ),
                            ),
                          ),
                        ),
                        SizedBox(
                          height: 16*fem,
                        ),
                        Container(
                          // inputGrm (1:59)
                          width: double.infinity,
                          height: 42*fem,
                          child: Container(
                            // inputRjf (I1:59;114:115)
                            padding: EdgeInsets.fromLTRB(14.1*fem, 14*fem, 14.1*fem, 14*fem),
                            width: double.infinity,
                            height: double.infinity,
                            decoration: BoxDecoration (
                              border: Border.all(color: Color(0xffd4d7e3)),
                              color: Color(0xfff2f6fa),
                              borderRadius: BorderRadius.circular(8*fem),
                              boxShadow: [
                                BoxShadow(
                                  color: Color(0x3f000000),
                                  offset: Offset(0*fem, 4*fem),
                                  blurRadius: 2*fem,
                                ),
                              ],
                            ),
                            child: Text(
                              'Confirm Password',
                              style: SafeGoogleFont (
                                'Roboto',
                                fontSize: 14*ffem,
                                fontWeight: FontWeight.w400,
                                height: 1*ffem/fem,
                                letterSpacing: 0.14*fem,
                                color: Color(0xff8796ad),
                              ),
                            ),
                          ),
                        ),
                        SizedBox(
                          height: 16*fem,
                        ),
                        Container(
                          // mainbuttonFCu (1:60)
                          width: double.infinity,
                          height: 44*fem,
                          decoration: BoxDecoration (
                            color: Color(0xff0056ff),
                            borderRadius: BorderRadius.circular(12*fem),
                            boxShadow: [
                              BoxShadow(
                                color: Color(0x3f000000),
                                offset: Offset(0*fem, 4*fem),
                                blurRadius: 2*fem,
                              ),
                            ],
                          ),
                          child: Center(
                            child: Center(
                              child: Text(
                                'Sign up',
                                textAlign: TextAlign.center,
                                style: SafeGoogleFont (
                                  'Roboto',
                                  fontSize: 16*ffem,
                                  fontWeight: FontWeight.w400,
                                  height: 1*ffem/fem,
                                  letterSpacing: 0.16*fem,
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
                    // orWPj (1:62)
                    margin: EdgeInsets.fromLTRB(0*fem, 0*fem, 0*fem, 34*fem),
                    width: double.infinity,
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Container(
                          // line2SoB (1:63)
                          margin: EdgeInsets.fromLTRB(0*fem, 1*fem, 0*fem, 0*fem),
                          width: 107.5*fem,
                          height: 1*fem,
                          decoration: BoxDecoration (
                            color: Color(0xffcfdfe2),
                          ),
                        ),
                        SizedBox(
                          width: 16*fem,
                        ),
                        Center(
                          // orsignupwithYrD (1:64)
                          child: Text(
                            'Or sign up with',
                            textAlign: TextAlign.center,
                            style: SafeGoogleFont (
                              'Roboto',
                              fontSize: 14*ffem,
                              fontWeight: FontWeight.w400,
                              height: 1*ffem/fem,
                              letterSpacing: 0.14*fem,
                              color: Color(0xff294956),
                            ),
                          ),
                        ),
                        SizedBox(
                          width: 16*fem,
                        ),
                        Container(
                          // line1G1X (1:65)
                          margin: EdgeInsets.fromLTRB(0*fem, 1*fem, 0*fem, 0*fem),
                          width: 107.5*fem,
                          height: 1*fem,
                          decoration: BoxDecoration (
                            color: Color(0xffcfdfe2),
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    // frame2PM3 (1:66)
                    margin: EdgeInsets.fromLTRB(0*fem, 0*fem, 0*fem, 24*fem),
                    width: double.infinity,
                    height: 44*fem,
                    decoration: BoxDecoration (
                      color: Color(0xfff3f9f8),
                      borderRadius: BorderRadius.circular(12*fem),
                      boxShadow: [
                        BoxShadow(
                          color: Color(0x19000000),
                          offset: Offset(0*fem, 4*fem),
                          blurRadius: 2*fem,
                        ),
                      ],
                    ),
                    child: Stack(
                      children: [
                        Positioned(
                          // bajajaccounttYh (1:68)
                          left: 93*fem,
                          top: 11.5*fem,
                          child: Center(
                            child: Align(
                              child: SizedBox(
                                width: 105*fem,
                                height: 26*fem,
                                child: RichText(
                                  textAlign: TextAlign.center,
                                  text: TextSpan(
                                    style: SafeGoogleFont (
                                      'Roboto',
                                      fontSize: 16*ffem,
                                      fontWeight: FontWeight.w400,
                                      height: 1.6000000238*ffem/fem,
                                      letterSpacing: 0.16*fem,
                                      color: Color(0xffffffff),
                                    ),
                                    children: [
                                      TextSpan(
                                        text: '                     ',
                                        style: SafeGoogleFont (
                                          'Roboto',
                                          fontSize: 16*ffem,
                                          fontWeight: FontWeight.w300,
                                          height: 1.6000000238*ffem/fem,
                                          letterSpacing: 0.16*fem,
                                          color: Color(0xffffffff),
                                        ),
                                      ),
                                      TextSpan(
                                        text: 'Bajaj Account',
                                        style: SafeGoogleFont (
                                          'Roboto',
                                          fontSize: 16*ffem,
                                          fontWeight: FontWeight.w300,
                                          height: 1.6000000238*ffem/fem,
                                          letterSpacing: 0.16*fem,
                                          color: Color(0xff000000),
                                        ),
                                      ),
                                      TextSpan(
                                        text: ' ',
                                        style: SafeGoogleFont (
                                          'Roboto',
                                          fontSize: 16*ffem,
                                          fontWeight: FontWeight.w300,
                                          height: 1.6000000238*ffem/fem,
                                          letterSpacing: 0.16*fem,
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
                          // arcticonsbajajfinservxaV (1:69)
                          left: 87.0833435059*fem,
                          top: 4.0833129883*fem,
                          child: Align(
                            child: SizedBox(
                              width: 35.83*fem,
                              height: 35.83*fem,
                              child: Image.asset(
                                'assets/page-1/images/arcticons-bajaj-finserv.png',
                                width: 35.83*fem,
                                height: 35.83*fem,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  Center(
                    // alreadyhaveanaccountloginrQy (1:72)
                    child: Container(
                      width: double.infinity,
                      child: RichText(
                        textAlign: TextAlign.center,
                        text: TextSpan(
                          style: SafeGoogleFont (
                            'Roboto',
                            fontSize: 16*ffem,
                            fontWeight: FontWeight.w400,
                            height: 1.6000000238*ffem/fem,
                            letterSpacing: 0.16*fem,
                            color: Color(0xff112b30),
                          ),
                          children: [
                            TextSpan(
                              text: 'Already have an account? ',
                              style: SafeGoogleFont (
                                'Roboto',
                                fontSize: 16*ffem,
                                fontWeight: FontWeight.w400,
                                height: 1.6000000238*ffem/fem,
                                letterSpacing: 0.16*fem,
                                color: Color(0xff313957),
                              ),
                            ),
                            TextSpan(
                              text: 'Login',
                              style: SafeGoogleFont (
                                'Roboto',
                                fontSize: 16*ffem,
                                fontWeight: FontWeight.w400,
                                height: 1.6000000238*ffem/fem,
                                letterSpacing: 0.16*fem,
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
              // allrightsreservedu1s (1:73)
              child: Text(
                '© 2023 ALL RIGHTS RESERVED',
                textAlign: TextAlign.center,
                style: SafeGoogleFont (
                  'Roboto',
                  fontSize: 14*ffem,
                  fontWeight: FontWeight.w400,
                  height: 1*ffem/fem,
                  letterSpacing: 0.14*fem,
                  color: Color(0xff959cb5),
                ),
              ),
            ),
          ],
        ),
      ),
          );
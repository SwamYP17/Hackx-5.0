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
        // mainpageQ1o (1:74)
        padding: EdgeInsets.fromLTRB(0*fem, 48*fem, 0*fem, 0*fem),
        width: double.infinity,
        decoration: BoxDecoration (
          color: Color(0xff000000),
          borderRadius: BorderRadius.circular(30*fem),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Container(
              // scandocumentK8m (1:107)
              margin: EdgeInsets.fromLTRB(41*fem, 0*fem, 70.5*fem, 412*fem),
              width: double.infinity,
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Container(
                    // back3Kf (1:108)
                    margin: EdgeInsets.fromLTRB(0*fem, 0*fem, 72.7*fem, 1*fem),
                    width: 13.3*fem,
                    height: 13*fem,
                    child: Image.asset(
                      'assets/page-1/images/back.png',
                      width: 13.3*fem,
                      height: 13*fem,
                    ),
                  ),
                  Container(
                    // scandocumentSm7 (1:119)
                    margin: EdgeInsets.fromLTRB(0*fem, 0*fem, 41.5*fem, 0*fem),
                    child: Text(
                      'Scan Document',
                      style: SafeGoogleFont (
                        'Inter',
                        fontSize: 16*ffem,
                        fontWeight: FontWeight.w600,
                        height: 1.2125*ffem/fem,
                        color: Color(0xffffffff),
                      ),
                    ),
                  ),
                  Container(
                    // group9Qd (1:111)
                    width: 28*fem,
                    height: 28*fem,
                    child: Image.asset(
                      'assets/page-1/images/group.png',
                      width: 28*fem,
                      height: 28*fem,
                    ),
                  ),
                ],
              ),
            ),
            Container(
              // autogroupuq5wfds (21B1PjhfTGBxKzs8gqUQ5w)
              padding: EdgeInsets.fromLTRB(24*fem, 40*fem, 29*fem, 42*fem),
              width: double.infinity,
              decoration: BoxDecoration (
                color: Color(0xffffffff),
                borderRadius: BorderRadius.only (
                  topLeft: Radius.circular(20*fem),
                  topRight: Radius.circular(20*fem),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Container(
                    // youcanaddsomethingAKj (1:93)
                    margin: EdgeInsets.fromLTRB(0*fem, 0*fem, 96*fem, 76*fem),
                    child: Text(
                      'Add a document or an image',
                      style: SafeGoogleFont (
                        'Inter',
                        fontSize: 16*ffem,
                        fontWeight: FontWeight.w700,
                        height: 1.2125*ffem/fem,
                        color: Color(0xff2c3d7a),
                      ),
                    ),
                  ),
                  Container(
                    // autogroupyv7priM (21B1aQ4ZmzwLGqrtrUyV7P)
                    margin: EdgeInsets.fromLTRB(16*fem, 0*fem, 49*fem, 19*fem),
                    width: double.infinity,
                    height: 40*fem,
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Container(
                          // documentNwb (1:101)
                          margin: EdgeInsets.fromLTRB(0*fem, 0*fem, 56*fem, 0*fem),
                          height: double.infinity,
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: [
                              Container(
                                // autogroup8ugu7eH (21B1ryRHFKNBLzyKZm8ugu)
                                margin: EdgeInsets.fromLTRB(0*fem, 0*fem, 11*fem, 0*fem),
                                width: 40*fem,
                                height: 40*fem,
                                child: Image.asset(
                                  'assets/page-1/images/auto-group-8ugu.png',
                                  width: 40*fem,
                                  height: 40*fem,
                                ),
                              ),
                              Container(
                                // documentq4V (1:102)
                                margin: EdgeInsets.fromLTRB(0*fem, 1*fem, 0*fem, 0*fem),
                                child: Text(
                                  'Document',
                                  style: SafeGoogleFont (
                                    'Inter',
                                    fontSize: 14*ffem,
                                    fontWeight: FontWeight.w500,
                                    height: 1.2125*ffem/fem,
                                    color: Color(0xff2c3d7a),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                        Container(
                          // gallerykx9 (1:94)
                          height: double.infinity,
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: [
                              Container(
                                // autogroup8cc1vLq (21B1hjBgkP5QVB3c6N8cC1)
                                margin: EdgeInsets.fromLTRB(0*fem, 0*fem, 8*fem, 0*fem),
                                width: 40*fem,
                                height: 40*fem,
                                child: Image.asset(
                                  'assets/page-1/images/auto-group-8cc1.png',
                                  width: 40*fem,
                                  height: 40*fem,
                                ),
                              ),
                              Container(
                                // barcodeDqj (1:95)
                                margin: EdgeInsets.fromLTRB(0*fem, 1*fem, 0*fem, 0*fem),
                                child: Text(
                                  'Gallery',
                                  style: SafeGoogleFont (
                                    'Inter',
                                    fontSize: 14*ffem,
                                    fontWeight: FontWeight.w500,
                                    height: 1.2125*ffem/fem,
                                    color: Color(0xff2c3d7a),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    // autogroupmepb9Db (21B1xyFHgMfL4j2p2wMePb)
                    width: 337*fem,
                    height: 119*fem,
                    child: Image.asset(
                      'assets/page-1/images/auto-group-mepb.png',
                      width: 337*fem,
                      height: 119*fem,
                    ),
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
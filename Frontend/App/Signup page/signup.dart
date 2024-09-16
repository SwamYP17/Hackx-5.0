import 'package:flutter/material.dart';
//import 'main.dart';
class SignupPage extends StatelessWidget {
  // Declare a global ScaffoldMessengerState variable
  final scaffoldMessengerKey = GlobalKey<ScaffoldMessengerState>();

  SignupPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ScaffoldMessenger(
      key: scaffoldMessengerKey,
      child: Scaffold(
        appBar: AppBar(
          title: Text("Signup Page"),
          centerTitle: true,
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              TextField(
                decoration: InputDecoration(
                  labelText: 'Bajaj ID',
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 20),
              TextField(
                decoration: InputDecoration(
                  labelText: 'Full Name',
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 20),
              TextField(
                decoration: InputDecoration(
                  labelText: 'Password',
                  border: OutlineInputBorder(),
                ),
                obscureText: true,
              ),
              SizedBox(height: 20),
              TextField(
                decoration: InputDecoration(
                  labelText: 'Confirm Password',
                  border: OutlineInputBorder(),
                ),
                obscureText: true,
              ),
              SizedBox(height: 20),
              ElevatedButton(
  onPressed: () {
    // Perform signup action.

    // After signup is successful, pass the success message back to LoginPage
    Navigator.pop(context, "Signup successful. You can login.");
  },
  child: Text("Signup"),
  style: ElevatedButton.styleFrom(
    primary: Colors.blue,
  ),
),
            ],
          ),
        ),
      ),
    );
  }
}
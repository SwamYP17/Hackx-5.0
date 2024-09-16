import 'package:flutter/material.dart';
import 'main.dart';
import 'Signup_page.dart'; // Import the signup page

class LoginPage extends StatelessWidget {
  // Controller to handle text input
  final TextEditingController bajajIdController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();

  LoginPage({Key? key});

  // Function to handle the login logic
  void handleLogin(BuildContext context) {
    String bajajId = bajajIdController.text;
    String password = passwordController.text;

    // Replace this with your actual login logic.
    // For demonstration purposes, we're using a hardcoded check for the login credentials.
    if (bajajId == '11223344' && password == 'abhishek') {
      // Navigate to the HomePage when the login is successful
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => const MyHomePage()),
      );
    } else {
      // Show an error message or handle login failure here
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Login Failed'),
          content: const Text('Invalid credentials. Please try again.'),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('OK'),
            ),
          ],
        ),
      );
    }
  }

 @override
Widget build(BuildContext context) {
 return Scaffold(
    appBar: AppBar(
      title: const Text('AuthentiScan'),
    ),
    body: Column(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Expanded(
          child: Image.asset(
            'images/Doc-p.png', // Replace with your image path
            fit: BoxFit.cover,
          ),
        ),
        Center(
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Shadow effect for Bajaj ID input field
                Container(
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(8),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.grey.withOpacity(0.3),
                        spreadRadius: 3,
                        blurRadius: 5,
                        offset: Offset(0, 3), // changes position of shadow
                      ),
                    ],
                  ),
                  child: TextField(
                    controller: bajajIdController,
                    decoration: const InputDecoration(
                      hintText: 'Bajaj ID', // Placeholder for Bajaj ID
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(height: 20.0),
                // Shadow effect for Password input field
                Container(
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(8),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.grey.withOpacity(0.3),
                        spreadRadius: 3,
                        blurRadius: 5,
                        offset: Offset(0, 3), // changes position of shadow
                      ),
                    ],
                  ),
                  child: TextField(
                    controller: passwordController,
                    obscureText: true,
                    decoration: const InputDecoration(
                      hintText: 'Password', // Placeholder for Password
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(height: 20.0),
                ElevatedButton(
                  onPressed: () => handleLogin(context), // Call handleLogin function
                  child: const Text('Login'),
                ),
                const SizedBox(height: 20.0),
                Row(
                  children: [
                    Expanded(child: Divider(thickness: 1)),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 8.0),
                      child: const Text(
                        'Sign in with',
                        textAlign: TextAlign.center,
                        style: TextStyle(fontSize: 16.0),
                      ),
                    ),
                    Expanded(child: Divider(thickness: 2)),
                  ],
                ),
                const SizedBox(height: 10.0),
                ElevatedButton(
                  onPressed: () {
                    // TODO: Implement "Sign in with" functionality
                    // This function will be called when the button is pressed.
                  },
                  child: const Text('Login with Bajaj ID'),
                ),
                const SizedBox(height: 20.0),
                const Text(
                  "Don't have an account?", // Message "Don't have an account?"
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16.0),
                ),
                const SizedBox(height: 10.0),
                GestureDetector(
                  onTap: () async {
                    // Navigate to the signup page and wait for a result (success message)
                    final result = await Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => SignupPage()),
                    );

                    // If a success message is received, show it using a SnackBar
                    if (result != null && result is String) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text(result),
                          duration: Duration(seconds: 4),
                        ),
                      );
                    }
                  },
                  child: const Text(
                    'Sign up',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 16.0, color: Colors.blue),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    ),
  );
}
}
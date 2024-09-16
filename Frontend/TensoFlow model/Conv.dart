import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteModel {
  late Interpreter _interpreter;

  Future<void> loadModel() async {
    try {
      final interpreterOptions = InterpreterOptions();
      _interpreter = await Interpreter.fromAsset('your_model.tflite',
          options: interpreterOptions);
    } catch (e) {
      print('Error loading TensorFlow Lite model: $e');
    }
  }

  Future<List<dynamic>> runInference(String inputText) async {
    try {
      final input = [inputText]; // Assuming your model takes a list of strings as input
      final output = List.filled(1, <double>[]); // Initialize the output list
      _interpreter.run(input, output);
      return output;
    } catch (e) {
      print('Error running inference: $e');
      return [];
    }
  }
}
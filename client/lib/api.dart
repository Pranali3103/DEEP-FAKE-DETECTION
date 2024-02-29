import 'dart:convert';
import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class API extends ChangeNotifier {
  bool isLoading = false;
  File? image, video;
  String cameraMode = "gallery";
  Future<void> getImage() async {
    if (cameraMode == "gallery") {
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(source: ImageSource.gallery);

      if (pickedFile != null) {
        image = File(pickedFile.path);
      } else {
        Fluttertoast.showToast(msg: "No image selected");
      }
    } else if (cameraMode == "camera") {
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(source: ImageSource.camera);
      if (pickedFile != null) {
        image = File(pickedFile.path);
      } else {
        Fluttertoast.showToast(msg: "No image selected");
      }
    } else {
      Fluttertoast.showToast(msg: "Please select an input");
    }
    _sendImage();
    notifyListeners();
  }

  Future<void> getVideo() async {
    FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['mp4', 'mov'],
    ).then((value) {
      if (value != null) {
        video = File(value.files.single.path!);
        detectVideo(video!.path);
      } else {
        Fluttertoast.showToast(msg: "No video selected");
      }
      notifyListeners();
    });
  }

  String predictedClass = "None";
  String _result = '';

  Future<void> detectVideo(String pathVideo) async {
    print('Detecting Deepfake');
    isLoading = true;
    notifyListeners();
    try {
      // Replace the URL with your Flask API endpoint
      String apiUrl = 'http://192.168.43.163:9000/predict_video';

      // Replace 'file' with the key used in the Flask API for the file upload
      var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
      request.files.add(await http.MultipartFile.fromPath('file', pathVideo));

      var response = await request.send();

      if (response.statusCode == 200) {
        String responseBody = await response.stream.bytesToString();
        Map<String, dynamic> result = json.decode(responseBody);

        // Assuming the result contains 'real' and 'fake' keys
        double threshold = 0.5;

        _result =
            'Real Confidence: ${result['real']}\nFake Confidence: ${result['fake']}';

        if (result['real'] > threshold) {
          predictedClass = 'Real';
          isLoading = false;
          notifyListeners();
        } else {
          predictedClass = 'Fake';
          isLoading = false;
          notifyListeners();
        }
      } else {
        _result = 'Error: ${response.reasonPhrase}';
      }
    } catch (e) {
      _result = 'Error: $e';
    }
  }

  Future<void> _sendImage() async {
    if (image == null) {
      Fluttertoast.showToast(msg: "Please select an image");
      return;
    }

    String apiUrl = "http://192.168.43.163:8000/predict_photo";

    var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
    request.files.add(await http.MultipartFile.fromPath('image', image!.path));
    predictedClass = "";
    try {
      var response = await request.send();
      if (response.statusCode == 200) {
        var responseBody = await response.stream.bytesToString();
        double threshold = 0.5;
        var res = jsonDecode(responseBody);

        if (res['real'] > threshold) {
          predictedClass = 'Real';
          notifyListeners();
        } else {
          predictedClass = 'Fake';
          notifyListeners();
        }
      } else {
        Fluttertoast.showToast(msg: "Error: ${response.statusCode}");

        predictedClass = "None";
        notifyListeners();
      }
    } catch (e) {
      Fluttertoast.showToast(msg: "Error: $e");
    }
    notifyListeners();
  }
}
